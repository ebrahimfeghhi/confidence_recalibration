#!/usr/bin/env python
"""THE FULL RUN SPEC AS A TABLE, DERIVED FROM THE ARGV THE LAUNCHER WOULD ACTUALLY PASS.

    run_spec_table.py --dry <dry.log> [--ref <launch.log|args.json>] [--label L] [--save DIR]

WHY IT EXISTS, and why it reads the argv instead of the launcher's banner. Three runs in this
project were launched with an objective nobody intended: a duplicate SKIP whose empty value won,
a `#` inside a backslash-continued env block that split one command into two, and -- caught at
DRY on 2026-11-23 -- `${PAE_BALANCE_ACCEPT:-0.23}` turning the 36x good/bad reweighting ON while
reproducing a run that never had it. Every one of them passed `bash -n`, and every one was
invisible in the launcher's own summary lines, which print what the author MEANT. The argv block
is what the trainer receives, so a table built from it cannot disagree with the run.

IT IS A DIFF WHENEVER THERE IS SOMETHING TO DIFF AGAINST. --ref takes the launch log of the run
being continued or compared with; every flag that differs is marked, so "everything identical
except X" is a claim the table either supports or refutes. A reference whose argv cannot be found
is an error, not a silently empty diff column.

THE TABLE IS SAVED, not just printed: --save writes it under training/run_specs/ with the DRY
log beside it, so the spec that was approved can be compared with what the run actually recorded
in its args.json afterwards.
"""
import argparse, json, os, re, sys, time


def argv_from(path):
    """The argv block, from a DRY log, a real launch log, or a run's args.json."""
    if path.endswith(".json"):
        j = json.load(open(path))
        a = j.get("args", j)
        out = {}
        for k, v in a.items():
            if v in (None, False, "", [], 0.0) and k not in ("seed",):
                continue
            out["--" + k.replace("_", "-")] = True if v is True else str(v)
        return out, "args.json"
    txt = open(path).read()
    # THE LONGEST CONTIGUOUS INDENTED BLOCK, not the lines right after the banner.
    #
    # launch_pae_crop_arm.sh prints "argv the trainer would receive:" at line 528 and the actual
    # `printf '    %s\n' "${ARGS[@]}"` at line 653, with ~40 gate and `say` lines in between --
    # so taking the indented lines that FOLLOW the banner captured a 30-flag fragment and silently
    # omitted --resume, --pair-dropout, --train-only, --expect-trainable, --pae-split-accept and
    # every runtime flag. A spec table that drops the resume path and the dropout is worse than no
    # table, because it reads as complete. The real argv is the longest unbroken run of indented
    # lines in the file, which is unambiguous in both DRY logs and real launch logs.
    # THE REAL COMMAND LINE WINS OVER THE PREVIEW BLOCK.
    #
    # A completed run's launch log holds BOTH: an indented preview printed partway through the
    # launcher, and the full `phase_k_train.py ...` line that was actually executed. Reading the
    # preview made 29 flags -- --resume, --pair-dropout, --train-only, the whole runtime set --
    # look like they had been ADDED relative to the reference, when both runs passed them. A diff
    # that reports 29 spurious changes is unusable for the thing it exists for. The executed line
    # is the complete truth, so it is preferred whenever it exists; a DRY log has no such line and
    # falls back to the preview, which is complete there by construction.
    toks = None
    m = re.search(r"phase_k_train\.py ((?:--|\S)[^\n]*)", txt)
    if m:
        toks = m.group(1).split()
    else:
        blocks = re.findall(r"(?:^ {4}\S[^\n]*\n)+", txt, re.M)
        if blocks:
            best = max(blocks, key=lambda b: b.count("\n"))
            toks = [l[4:] for l in best.splitlines()]
    toks = [t.strip().strip("'") for t in (toks or []) if t.strip()]
    if not any(t.startswith("--") for t in toks):
        toks = None
    if not toks:
        sys.exit(f"STOP: no argv block found in {path}. A reference without an argv is not a "
                 f"reference -- pass the launch log that printed one, or the run's args.json.")
    d, i = {}, 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                d[t] = toks[i + 1]; i += 2
            else:
                d[t] = True; i += 1
        else:
            i += 1
    return d, "argv block"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", required=True, help="DRY=1 log of the run about to launch")
    ap.add_argument("--ref", help="launch log or args.json of the run this continues/compares to")
    ap.add_argument("--ref-label", default="reference")
    ap.add_argument("--label", default="this run")
    ap.add_argument("--save", default="training/run_specs")
    a = ap.parse_args()

    new, src = argv_from(a.dry)
    ref, rsrc = (argv_from(a.ref) if a.ref else ({}, None))

    lines = []
    w = lines.append
    w(f"# RUN SPEC -- {a.label}")
    w("")
    w(f"- generated {time.strftime('%Y-%m-%d %-I:%M %p %Z')}")
    w(f"- derived from the argv block of `{a.dry}` ({src}), not from any banner or summary")
    if a.ref:
        w(f"- compared against `{a.ref}` ({rsrc}), shown as **{a.ref_label}**")
    w("")
    keys = sorted(set(new) | set(ref))
    diffs = [k for k in keys if new.get(k) != ref.get(k)]
    if a.ref:
        w(f"**{len(diffs)} flag(s) differ from {a.ref_label}.** "
          + ("Everything else is identical." if diffs else "Nothing differs."))
        w("")
        w(f"| flag | {a.ref_label} | {a.label} | |")
        w("|---|---|---|---|")
        for k in diffs:
            w(f"| `{k}` | {fmt(ref.get(k))} | **{fmt(new.get(k))}** | CHANGED |")
        w("")
    # GROUPED, ONE FLAG PER ROW. The first version packed five flags into a cell to keep the
    # table short, and the result was unreadable -- which defeats the point of a spec that has to
    # be checked by eye before a launch. The groups are matched from the flag NAME, so a flag
    # added later lands in a section without anyone maintaining a list, and anything unmatched
    # shows up under "other" rather than being dropped.
    GROUPS = [
        ("Resume & horizon",   ("resume", "epochs", "warmup", "lr-", "lr", "seed")),
        ("Data & crop",        ("crop", "manifest", "fold", "indices", "pae-scored", "skip-b",
                                "manifest-skip", "require-replay", "minimal-items", "nsample",
                                "epoch-cover", "epoch-balance", "alignment-preflight")),
        ("Objective",          ("pae-only", "alpha", "rank", "rel-", "pae-split", "pae-balance",
                                "pae-tm", "l2-sp", "drop-pde")),
        ("Architecture",       ("pose", "frame-direction", "train-only", "expect-trainable",
                                "lora", "lean", "skip-dead")),
        ("Regularisation",     ("pair-dropout", "clip-norm")),
        ("Evaluation",         ("train-eval", "eval-", "extra-eval")),
        ("Runtime",            ("accum", "prefetch", "trunk-memmap", "blocks-per-ckpt",
                                "ckpt-token", "mem-log", "dist-", "max-rss", "preload",
                                "save-every", "allow-code-drift", "out")),
    ]

    def group_of(flag):
        """PREFIX MATCH ONLY, longest pattern first.

        Substring matching put --rank-cross-seed under "Resume & horizon" because the pattern
        "seed" appeared inside it. Prefixes are unambiguous, and sorting the patterns longest-first
        means --train-eval-seed-base matches "train-eval" rather than a shorter accidental prefix.
        """
        name = flag.lstrip("-")
        best = None
        for title, pats in GROUPS:
            for x in pats:
                if name.startswith(x) and (best is None or len(x) > best[1]):
                    best = (title, len(x))
        return best[0] if best else "other"

    seen = set()
    w(f"## Full spec -- {len(new)} flags")
    for title, _ in GROUPS + [("other", ())]:
        rows = [k for k in sorted(new) if group_of(k) == title and k not in seen]
        if not rows:
            continue
        seen.update(rows)
        w("")
        w(f"### {title}")
        w("")
        w("| flag | value |" + (f" vs {a.ref_label} |" if a.ref else ""))
        w("|---|---|" + ("---|" if a.ref else ""))
        for k in rows:
            row = f"| `{k}` | {fmt(new[k])} |"
            if a.ref:
                row += (" same |" if ref.get(k) == new[k]
                        else f" **was {fmt(ref.get(k))}** |")
            w(row)
    dropped = [k for k in sorted(ref) if k not in new]
    if dropped:
        w("")
        w("### Passed by the reference and NOT here")
        w("")
        w(f"| flag | {a.ref_label} |")
        w("|---|---|")
        for k in dropped:
            w(f"| `{k}` | {fmt(ref[k])} |")
    text = "\n".join(lines) + "\n"
    print(text)

    if a.save:
        os.makedirs(a.save, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = re.sub(r"[^A-Za-z0-9_.-]+", "_", a.label)[:48]
        out = os.path.join(a.save, f"{stamp}_{base}.md")
        open(out, "w").write(text)
        import shutil
        shutil.copy(a.dry, out.replace(".md", ".dry.log"))
        print(f"saved {out}\nsaved {out.replace('.md', '.dry.log')}")


def fmt(v):
    if v is None:
        return "_not passed_"
    if v is True:
        return "on"
    return f"`{v}`"


if __name__ == "__main__":
    main()
