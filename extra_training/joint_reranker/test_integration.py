"""CPU gate for end-to-end integration callback."""

import unittest

import torch
from torch import nn, functional as F

from extra_training.joint_reranker.integration import PosePrediction, joint_group_loss
from extra_training.joint_reranker.model import PAEPoseReranker, antibody_antigen_pair_mask


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.head = nn.Linear(3, 64)
        self.model = PAEPoseReranker(attention_dropout=0).eval()
        self.asym = torch.tensor([0, 0, 1, 1, 2, 2])
        self.mask = antibody_antigen_pair_mask(self.asym, (0, 1), (2,))

    def predict(self, pose):
        x = self.head(pose)
        ce = F.cross_entropy(x[self.mask], torch.zeros(int(self.mask.sum()), dtype=torch.long))
        score = x.softmax(dim=-1)[self.mask, 0].mean()
        return PosePrediction(x, self.mask, score, ce)

    def test_two_live_losses_reach_head_and_reranker(self):
        poses = [torch.randn(6, 6, 3) for _ in range(4)]
        dockq = torch.tensor([0.01, 0.41, 0.12, 0.87])
        loss = joint_group_loss(self.model, poses, self.predict, dockq, alpha=1.0, temperature=0.5)
        self.assertTrue(loss.ranking_active)
        self.assertEqual(tuple(loss.scores.shape), (4,))
        loss.total.backward()
        self.assertGreater(self.head.weight.grad.norm().item(), 0)
        self.assertGreater(self.model.score_head[-1].weight.grad.norm().item(), 0)

    def test_all_bad_still_supervises_pae(self):
        poses = [torch.randn(6, 6, 3) for _ in range(3)]
        loss = joint_group_loss(self.model, poses, self.predict, torch.zeros(3))
        self.assertFalse(loss.ranking_active)
        loss.total.backward()
        self.assertGreater(self.head.weight.grad.norm().item(), 0)

    def test_detach_rejected(self):
        def broken(pose):
            x = self.head(pose)
            return PosePrediction(x.detach(), self.mask, x.softmax(-1).mean(), x.square().mean())
        with self.assertRaisesRegex(ValueError, "detached"):
            joint_group_loss(self.model, [torch.randn(6,6,3) for _ in range(2)],
                             broken, torch.tensor([0.1, 0.9]))


if __name__ == "__main__":
    unittest.main()
