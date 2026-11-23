"""Differentiable PAE-logit reranker. All inputs are live, deployable features.

For each pose: PAE bin softmax -> shared 64->64->64 pair encoder ->
mean/max/attention pooling -> 192->128 pose embedding.
Across poses: one 4-head self-attention block with no positional embedding ->
128->64->1 residual correction to differentiable chain-pair ipTM.

Variable numbers of poses and variable token dimensions are supported. No
reference-coordinate, DockQ, or native-chain mapping inputs are accepted.
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def antibody_antigen_pair_mask(
    asym_id: Tensor,
    antibody_asym_ids: Sequence[int],
    antigen_asym_ids: Sequence[int],
    has_frame: Tensor | None = None,
) -> Tensor:
    """Directed AB<->AG token pairs using MODEL chain IDs, never reference labels.

    Supply antigen IDs of the interface actually scored, not all antigen copies.
    Symmetric copies need an inference-time, label-free convention; OST's
    per-pose reference-to-model map may depend on ground truth and is forbidden.
    The row axis is N_frame (can differ from N_token).
    """
    if asym_id.ndim != 1:
        raise ValueError("asym_id must have shape [T]")
    if not antibody_asym_ids or not antigen_asym_ids:
        raise ValueError("antibody and antigen chain ID lists must be nonempty")
    if set(antibody_asym_ids) & set(antigen_asym_ids):
        raise ValueError("antibody and antigen chain IDs overlap")
    ab = torch.isin(asym_id, torch.as_tensor(list(antibody_asym_ids), device=asym_id.device))
    ag = torch.isin(asym_id, torch.as_tensor(list(antigen_asym_ids), device=asym_id.device))
    if has_frame is not None:
        if has_frame.shape != asym_id.shape:
            raise ValueError("has_frame must have shape [T]")
        ab_rows, ag_rows = ab[has_frame.bool()], ag[has_frame.bool()]
    else:
        ab_rows, ag_rows = ab, ag
    mask = (ab_rows[:, None] & ag[None, :]) | (ag_rows[:, None] & ab[None, :])
    if not mask.any():
        raise ValueError("no scored antibody-antigen pairs remain in this crop")
    return mask


def _pose_list(value: Tensor | Sequence[Tensor], name: str, ndim: int) -> list[Tensor]:
    if isinstance(value, Tensor):
        if value.ndim != ndim + 1:
            raise ValueError(f"{name} must have shape [K,...] with {ndim} trailing dims")
        return list(value.unbind(0))
    poses = list(value)
    if not poses:
        raise ValueError(f"{name} cannot be empty")
    for i, item in enumerate(poses):
        if not isinstance(item, Tensor) or item.ndim != ndim:
            raise ValueError(f"{name}[{i}] must be a {ndim}D tensor")
    return poses


class PAEPoseReranker(nn.Module):
    """Scores K poses jointly; K and crop length are variable at each call.

    The encoder indexes masked pairs BEFORE softmax to avoid materializing
    full softmax grids. By default ALL AB/AG pairs contribute: truncating
    them would otherwise change the scientific input silently. An explicit
    max_pairs cap is available as a separately tested memory ablation.
    Chunking itself does NOT remove autograd's saved activations.
    """

    def __init__(
        self,
        *,
        pae_bins: int = 64,
        pair_dim: int = 64,
        pose_dim: int = 128,
        attention_heads: int = 4,
        pair_chunk_size: int = 2048,
        max_pairs: int | None = None,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if pae_bins != 64:
            raise ValueError("this architecture assumes 64 PAE bins")
        if pair_chunk_size <= 0 or (max_pairs is not None and max_pairs <= 0) or pose_dim % attention_heads:
            raise ValueError("positive chunk/optional max_pairs and pose_dim divisible by heads required")
        self.pae_bins = pae_bins
        self.pair_chunk_size = pair_chunk_size
        self.max_pairs = max_pairs
        self.phi = nn.Sequential(
            nn.Linear(pae_bins, pair_dim), nn.SiLU(), nn.Linear(pair_dim, pair_dim)
        )
        self.pair_attention = nn.Linear(pair_dim, 1)
        self.pose_projection = nn.Linear(3 * pair_dim, pose_dim)
        self.cross_pose_attention = nn.MultiheadAttention(
            embed_dim=pose_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(pose_dim)
        self.pose_ffn = nn.Sequential(
            nn.Linear(pose_dim, 2 * pose_dim), nn.SiLU(), nn.Linear(2 * pose_dim, pose_dim)
        )
        self.norm2 = nn.LayerNorm(pose_dim)
        self.score_head = nn.Sequential(
            nn.Linear(pose_dim, 64), nn.SiLU(), nn.Linear(64, 1)
        )
        # At initialization the scorer is EXACTLY the supplied ipTM baseline.
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

    def _encode_pose(self, logits: Tensor, pair_mask: Tensor) -> Tensor:
        if logits.ndim != 3 or logits.shape[-1] != self.pae_bins:
            raise ValueError(f"PAE logits must have shape [N_frame,N_token,{self.pae_bins}]")
        if pair_mask.shape != logits.shape[:2]:
            raise ValueError("PAE logits and pair mask must have the same [N_frame,N_token] shape")
        if logits.device != pair_mask.device:
            raise ValueError("logits and pair mask must share device")
        rows, cols = torch.where(pair_mask.bool())
        count = rows.numel()
        if count == 0:
            raise ValueError("a pose has no valid AB/AG residue pairs")
        # Only use a pair-count limit when the experiment EXPLICITLY requests
        # it. Uniform subsampling by flattened index may discard informative
        # contacts; report it as a separate ablation, never the default.
        if self.max_pairs is not None and count > self.max_pairs:
            inds = torch.linspace(0, count - 1, steps=self.max_pairs,
                                  device=rows.device).long()
            rows, cols = rows[inds], cols[inds]
            count = self.max_pairs

        mean_sum: Tensor | None = None
        pooled_max: Tensor | None = None
        attn_lse: Tensor | None = None
        attn_mean: Tensor | None = None
        for start in range(0, count, self.pair_chunk_size):
            sl = slice(start, start + self.pair_chunk_size)
            p = F.softmax(logits[rows[sl], cols[sl]].float(), dim=-1)
            h = self.phi(p).float()
            chunk_sum = h.sum(dim=0)
            chunk_max = h.amax(dim=0)
            mean_sum = chunk_sum if mean_sum is None else mean_sum + chunk_sum
            pooled_max = chunk_max if pooled_max is None else torch.maximum(pooled_max, chunk_max)
            weights = self.pair_attention(h).squeeze(-1)
            chunk_lse = torch.logsumexp(weights, dim=0)
            chunk_mean = (torch.softmax(weights, dim=0).unsqueeze(-1) * h).sum(dim=0)
            if attn_lse is None:
                attn_lse, attn_mean = chunk_lse, chunk_mean
            else:
                combined = torch.logaddexp(attn_lse, chunk_lse)
                attn_mean = (attn_mean * torch.exp(attn_lse - combined)
                             + chunk_mean * torch.exp(chunk_lse - combined))
                attn_lse = combined
        assert mean_sum is not None and pooled_max is not None and attn_mean is not None
        pooled = torch.cat((mean_sum / count, pooled_max, attn_mean), dim=-1)
        return self.pose_projection(pooled.to(self.pose_projection.weight.dtype))

    def encode_pose(self, pae_logits: Tensor, pair_mask: Tensor) -> Tensor:
        """Per-pose [F,T,64] -> [pose_dim], usable in inference streaming."""
        return self._encode_pose(pae_logits, pair_mask)

    def score_embeddings(self, embeddings: Tensor, iptm_scores: Tensor) -> Tensor:
        """Score compressed [K,pose_dim] features from any number of poses.

        At inference compute per-pose embeddings under no_grad, discard each
        large PAE tensor, then rerank ALL 80/150 embeddings simultaneously.
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.pose_projection.out_features:
            raise ValueError("embeddings must have shape [K,pose_dim]")
        if iptm_scores.shape != (embeddings.shape[0],):
            raise ValueError("iptm_scores must have shape [K]")
        if embeddings.device != iptm_scores.device:
            raise ValueError("embeddings and ipTM must share device")
        x = embeddings.unsqueeze(0)
        cross, _ = self.cross_pose_attention(x, x, x, need_weights=False)
        x = self.norm1(x + cross)
        x = self.norm2(x + self.pose_ffn(x))
        delta = self.score_head(x).squeeze(0).squeeze(-1)
        return iptm_scores + delta.to(iptm_scores.dtype)

    def forward(
        self,
        pae_logits: Tensor | Sequence[Tensor],
        pair_masks: Tensor | Sequence[Tensor],
        iptm_scores: Tensor,
    ) -> Tensor:
        """[K,F,T,64] and [K,F,T] or lists with different crops -> [K].

        Feed CURRENT differentiable ipTM from the SAME head/logits. The reranker
        receives no DockQ or true/native-structure data.
        """
        logits = _pose_list(pae_logits, "pae_logits", 3)
        masks = _pose_list(pair_masks, "pair_masks", 2)
        k = len(logits)
        if len(masks) != k or iptm_scores.shape != (k,):
            raise ValueError("the number of logits, masks and ipTM scores must match")
        if any(t.device != iptm_scores.device for t in logits):
            raise ValueError("PAE logits and ipTM scores must share device")
        embeddings = torch.stack(
            [self._encode_pose(t, m) for t, m in zip(logits, masks)], dim=0
        )
        return self.score_embeddings(embeddings, iptm_scores)
