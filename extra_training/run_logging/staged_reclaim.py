#!/usr/bin/env python
"""Reclaim disk in stages, verifying at every step. Dry-run by default.

THE THING REVIEW CAUGHT. Moving files to a quarantine directory on the SAME filesystem reclaims
nothing -- the blocks are still charged to the user's quota, because quota is per uid over the
whole device, not per directory. /home is one ext4 volume with usrquota, so a quarantine on it is
a rename and the quota number does not move. Any plan that budgets folding against
"space freed by quarantining" is budgeting against space it does not have.

SO QUARANTINE IS FOR REVERSIBILITY, NOT FOR SPACE, and this tool keeps the two apart:

  stage 1  INVENTORY   record every target file with size, mtime and sha256 of the first 1 MiB,
                       plus the quota reading before anything moves. Written to the journal.
                       A new inventory INVALIDATES any prior validate/quarantine/verify state.
  stage 2  VALIDATE    derive the expected deletion set INDEPENDENTLY -- from the seed manifest,
                       or from the trim list -- and require it to equal the target list exactly.
                       Refuses missing, extra, duplicate and out-of-tree paths by name. Quarantine
                       will not run without a passing validate.
  stage 3  QUARANTINE  rename targets under --quarantine. Reversible, reclaims NOTHING. The
                       journal is written after EVERY move, so a crash leaves a journal that knows
                       exactly what moved and the restore path still works. Refuses to start while
                       any live process holds a file under the target tree. On re-run it
                       RECONCILES first -- a file already sitting in quarantine but not yet
                       journalled is adopted only if its size AND first-MiB digest match the
                       inventory. Ends by asserting COMPLETENESS: every inventoried file in
                       quarantine, zero errors, or the stage fails and verify/unlink refuse.
  stage 4  VERIFY      run the caller's verification command. It must exit 0. This is where
                       "the retained manifest still resolves" is proven, with the data still
                       recoverable by rename.
  stage 5  UNLINK      delete the quarantined files. THIS is the step that moves the quota, and
                       the tool reads the quota before and after and refuses to report success
                       unless the delta is within --tolerance of the inventoried bytes.

A missing stage-4 delta means something else holds the blocks -- an open file descriptor on a
deleted inode is the usual cause, and the space returns only when the holder exits. The tool says
so rather than reporting space it cannot see.

Usage:
  python -I run_logging/staged_reclaim.py --targets <file> --journal <path> --stage inventory
  ... --stage quarantine --quarantine training/.reclaim_q
  ... --stage verify --verify-cmd "python -I run_logging/verify_val_manifest.py"
  ... --stage unlink --i-have-verified
"""
import argparse, collections, hashlib, json, os, re, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def quota_kib():
    """The user's charged blocks, from quota(1). None if unreadable.

    Verified against this box's real output, which is three lines with the device row as
    `/dev/nvme1n1p1 2663921024 0 3221225472 ...`. Raised in review: a parser that silently returns
    None must never let the irreversible stage proceed, because the shortfall check is the only
    thing standing between a failed unlink and a false "space is available" report. --stage unlink
    now REFUSES to start without a baseline reading.
    """
    try:
        out = subprocess.run(["quota", "-v"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for ln in out.splitlines():
        parts = ln.split()
        if len(parts) >= 3 and parts[0].startswith("/dev/"):
            try:
                return int(parts[1])
            except ValueError:
                continue
    return None


def head_digest(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()[:16]


def readers_clear(a):
    """No live process may hold a file under the tree being moved.

    Raised in review. Quarantine is a rename, which does not disturb an already-open descriptor --
    but the NEXT open by that process fails, and a trainer part-way through an epoch will either
    die or write a partial checkpoint. The user has accepted losing the current arm, but it must
    be STOPPED first rather than broken mid-write.

    This reads /proc/<pid>/fd directly. It never pattern-matches a command line and never signals
    anything; it reports and lets the caller decide.
    """
    roots = sorted({os.path.join(ROOT, os.path.dirname(r["rel"])) for r in J["files"]})
    prefixes = sorted({os.path.join(ROOT, r["rel"]).rsplit("/", 2)[0] for r in J["files"]})
    holders = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        fdd = f"/proc/{pid}/fd"
        try:
            fds = os.listdir(fdd)
        except OSError:
            continue
        for fd in fds:
            try:
                tgt = os.readlink(os.path.join(fdd, fd))
            except OSError:
                continue
            if any(tgt.startswith(pre) for pre in prefixes):
                holders.setdefault(pid, tgt)
                break
    if holders:
        print("  LIVE READERS under the target tree:")
        for pid, tgt in sorted(holders.items()):
            try:
                cl = open(f"/proc/{pid}/cmdline").read().replace("\0", " ")[:90]
            except OSError:
                cl = "?"
            print(f"    pid {pid}: {tgt[:70]}")
            print(f"              {cl}")
        return False
    print(f"  no live process holds anything under {len(prefixes)} target prefix(es)")
    return True


ap = argparse.ArgumentParser()
ap.add_argument("--targets", required=True, help="one path per line, relative to extra_training")
ap.add_argument("--journal", required=True)
ap.add_argument("--stage", required=True,
                choices=["inventory", "validate", "quarantine", "verify", "unlink", "restore"])
ap.add_argument("--expect", default="",
                help="REQUIRED by --stage validate. 'val-manifest:<path>' derives the expected "
                     "deletion set as the complement of each structure's selected seeds; "
                     "'train-trim:<pdb list>' derives it as every unit but the lowest-numbered "
                     "one of each listed structure. The derived set must EXACTLY equal the target "
                     "list -- no missing, no extra, no duplicates, nothing outside the expected "
                     "trunk directories.")
ap.add_argument("--trunks", default="", help="the trunk root the targets must live under")
ap.add_argument("--journal-every", type=int, default=50, help="fsync the journal every N moves")
ap.add_argument("--quarantine", default="training/.reclaim_quarantine")
ap.add_argument("--verify-cmd", default="")
ap.add_argument("--tolerance", type=float, default=0.05, help="allowed quota-delta shortfall")
ap.add_argument("--i-have-verified", action="store_true", help="required for --stage unlink")
a = ap.parse_args()

JP = os.path.join(ROOT, a.journal)
J = json.load(open(JP)) if os.path.exists(JP) else {"stages": {}, "files": []}


def save(fsync=False):
    os.makedirs(os.path.dirname(JP), exist_ok=True)
    tmp = JP + ".tmp"
    with open(tmp, "w") as f:
        json.dump(J, f, indent=1)
        if fsync:
            f.flush()
            os.fsync(f.fileno())
    os.replace(tmp, JP)


if a.stage == "inventory":
    paths = [l.strip() for l in open(a.targets) if l.strip()]
    files, total = [], 0
    for rel in paths:
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            print(f"  SKIP (not a file): {rel}")
            continue
        sz = os.path.getsize(p)
        files.append({"rel": rel, "bytes": sz, "mtime": os.path.getmtime(p),
                      "head_sha256_1MiB": head_digest(p)})
        total += sz
    J["files"] = files
    # A NEW INVENTORY INVALIDATES ANY PRIOR VERIFICATION. Raised in review: otherwise a journal
    # carrying a passing verify from an earlier, different target set would authorise an unlink of
    # the new one.
    J["stages"].pop("verify", None)
    J["stages"].pop("quarantine", None)
    J["stages"].pop("validate", None)
    J["stages"]["inventory"] = {"at": time.time(), "n": len(files), "bytes": total,
                                "quota_kib_before": quota_kib()}
    save()
    print(f"inventoried {len(files)} file(s), {total/1024**3:.1f} GiB")
    print(f"  quota before: {J['stages']['inventory']['quota_kib_before']} KiB")
    print("  NOTE: nothing has moved. Quarantine does not reclaim quota; only unlink does.")

elif a.stage == "validate":
    # THE MOST IMPORTANT GATE. Raised in review: every other check proves the RETAINED data
    # survives. None of them proves the deletion list contains only what was meant. A list with
    # one stray path deletes something nobody inspected, and the verifiers would still pass.
    #
    # So the expected set is DERIVED INDEPENDENTLY here, from the manifest or the drop list, and
    # must be EXACTLY equal to the target list. Not a subset, not a superset. Missing, extra,
    # duplicate, and out-of-tree paths are each refused by name rather than skipped.
    if not J.get("files"):
        sys.exit("STOP: run --stage inventory first")
    if not a.expect:
        sys.exit("STOP: --expect is required. 'val-manifest:<path>' or 'train-trim:<pdb list>'.")
    kind, _, arg = a.expect.partition(":")
    if not arg:
        sys.exit(f"STOP: --expect {a.expect!r} has no argument after the colon")
    trunks = a.trunks or ("validation/fold/trunks" if kind == "val-manifest"
                          else "training/fold/trunks")
    TR = os.path.join(ROOT, trunks)
    expected = set()
    if kind == "val-manifest":
        man = json.load(open(os.path.join(ROOT, arg)))
        for pdb, rec in man["structures"].items():
            keep = set(rec["seeds"])
            d = os.path.join(TR, pdb)
            if not os.path.isdir(d):
                sys.exit(f"STOP: manifest names {pdb} but {trunks}/{pdb} does not exist")
            for f in os.listdir(d):
                m = re.search(r"_(\d+)\.npz$", f)
                if m and int(m.group(1)) not in keep:
                    expected.add(os.path.relpath(os.path.join(d, f), ROOT))
        detail = (f"complement of the {man['k']} selected seeds for "
                  f"{len(man['structures'])} structure(s), manifest digest {man.get('digest')}")
    elif kind == "train-trim":
        pdbs = [l.strip() for l in open(arg) if l.strip()]
        for pdb in pdbs:
            d = os.path.join(TR, pdb)
            if not os.path.isdir(d):
                continue
            us = sorted((int(re.search(r"_(\d+)\.npz$", f).group(1)), f)
                        for f in os.listdir(d) if re.search(r"_(\d+)\.npz$", f))
            for _sd, f in us[1:]:                     # keep the LOWEST seed, drop the rest
                expected.add(os.path.relpath(os.path.join(d, f), ROOT))
        detail = f"every unit but the lowest-numbered one, for {len(pdbs)} listed structure(s)"
    else:
        sys.exit(f"STOP: --expect kind {kind!r} is not val-manifest or train-trim")

    targets = [r["rel"] for r in J["files"]]
    tset = set(targets)
    dupes = collections.Counter(targets)
    dupes = {k: v for k, v in dupes.items() if v > 1}
    outside = sorted(t for t in tset if not os.path.abspath(os.path.join(ROOT, t)).startswith(TR + os.sep))
    extra = sorted(tset - expected)
    missing = sorted(expected - tset)
    print(f"validate: {len(targets)} target(s) against an independently derived set of "
          f"{len(expected)}")
    print(f"  derivation: {detail}")
    print(f"  tree: {trunks}")
    bad = False
    for nm, lst in (("DUPLICATE in the target list", sorted(dupes)),
                    ("OUTSIDE the expected trunk tree", outside),
                    ("EXTRA -- in the list, not expected", extra),
                    ("MISSING -- expected, not in the list", missing)):
        if lst:
            bad = True
            print(f"  FAIL  {nm}: {len(lst)}")
            for x in lst[:6]:
                print(f"          {x}")
            if len(lst) > 6:
                print(f"          ... and {len(lst)-6} more")
        else:
            print(f"  PASS  no {nm.split(' --')[0].split(' in')[0].lower()} entries")
    J["stages"]["validate"] = {"at": time.time(), "ok": not bad, "expect": a.expect,
                               "n_targets": len(targets), "n_expected": len(expected),
                               "trunks": trunks}
    save(fsync=True)
    print(f"\n{'VALIDATE PASSED -- exact match' if not bad else 'VALIDATE FAILED'}")
    sys.exit(0 if not bad else 1)

elif a.stage == "quarantine":
    if not J.get("files"):
        sys.exit("STOP: run --stage inventory first")
    if not J["stages"].get("validate", {}).get("ok"):
        sys.exit("STOP: no passing --stage validate in the journal. The target list must be "
                 "proven to contain only the intended files before anything is moved.")
    if not readers_clear(a):
        sys.exit("STOP: a live process holds a file under the target tree. Moving it would break "
                 "that process mid-read. Stop it first, then re-run.")
    q = os.path.join(ROOT, a.quarantine)

    # RECONCILE FIRST. Raised in review: there is a crash window between shutil.move() returning
    # and the journal recording it, so a restart can find a file sitting in quarantine that the
    # journal still believes is at its source. Adopting it requires PROOF it is the same file, so
    # the inventoried size and the first-MiB digest must both match -- otherwise it is a different
    # file that happens to share a path and must not be silently claimed.
    adopted = mismatched = 0
    for rec in J["files"]:
        if rec.get("quarantined"):
            continue
        src = os.path.join(ROOT, rec["rel"])
        dst = os.path.join(q, rec["rel"])
        if os.path.isfile(src) or not os.path.isfile(dst):
            continue
        if os.path.getsize(dst) != rec["bytes"] or head_digest(dst) != rec["head_sha256_1MiB"]:
            rec["quarantine_error"] = "found in quarantine but size/digest differ from inventory"
            mismatched += 1
            continue
        rec["quarantined"] = True
        rec["quarantine_path"] = os.path.relpath(dst, ROOT)
        rec["adopted_by_reconcile"] = True
        # A STALE ERROR MUST NOT OUTLIVE THE CONDITION. Found by test: a file that errored on one
        # run, was then restored and successfully moved on the next, still carried the old
        # "source vanished" string -- so completeness failed with 3 of 3 moved. An error is a
        # statement about the CURRENT state and is cleared the moment the file is correctly placed.
        rec.pop("quarantine_error", None)
        adopted += 1
    if adopted or mismatched:
        save(fsync=True)
        print(f"  reconciled: adopted {adopted} already-moved file(s), "
              f"{mismatched} with a size/digest mismatch")

    moved = 0
    # PERSIST AFTER EVERY MOVE. Raised in review: saving only after the loop meant a crash partway
    # through thousands of renames left a journal that did not know which files had moved, so the
    # restore path -- the entire point of quarantining -- could not run. Journal writes are
    # atomic via os.replace, and fsync'd every --journal-every moves so a power loss cannot lose
    # more than that many records.
    for i, rec in enumerate(J["files"]):
        src = os.path.join(ROOT, rec["rel"])
        dst = os.path.join(q, rec["rel"])
        if rec.get("quarantined"):
            continue
        if not os.path.isfile(src):
            # NOT A SILENT SKIP. Raised in review: recording an error and carrying on let the
            # stage finish "successfully" with a partial move set, and the downstream verifiers
            # only check that the RETAINED files survive -- so a partial deletion could proceed.
            # This is now an error that the completeness check below refuses to pass.
            rec["quarantine_error"] = "source vanished and not found in quarantine"
            save()
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        rec["quarantined"] = True
        rec["quarantine_path"] = os.path.relpath(dst, ROOT)
        rec.pop("quarantine_error", None)          # see the note in the reconcile pass
        moved += 1
        if moved % a.journal_every == 0:
            save(fsync=True)
        else:
            save()
    # COMPLETENESS IS AN INVARIANT, not a summary. Every inventoried file must now be in
    # quarantine and none may carry an error. Anything else means a PARTIAL move set, and a
    # partial move set must never reach verify or unlink.
    n = len(J["files"])
    inq = sum(1 for r in J["files"] if r.get("quarantined"))
    errs = [(r["rel"], r["quarantine_error"]) for r in J["files"] if r.get("quarantine_error")]
    complete = (inq == n) and not errs
    J["stages"]["quarantine"] = {"at": time.time(), "moved": moved, "adopted": adopted,
                                 "in_quarantine": inq, "inventoried": n,
                                 "errors": len(errs), "complete": complete,
                                 "quota_kib_after": quota_kib()}
    save(fsync=True)
    print(f"quarantined {moved} file(s) under {a.quarantine} ({adopted} adopted on reconcile)")
    print(f"  in quarantine {inq} of {n} inventoried, {len(errs)} error(s)")
    print(f"  quota now: {J['stages']['quarantine']['quota_kib_after']} KiB")
    print("  EXPECTED TO BE UNCHANGED -- a same-filesystem rename reclaims no quota.")
    if not complete:
        for rel, why in errs[:8]:
            print(f"    {rel}: {why}")
        if len(errs) > 8:
            print(f"    ... and {len(errs)-8} more")
        sys.exit(f"STOP: quarantine is INCOMPLETE -- {inq} of {n} moved, {len(errs)} error(s). "
                 f"Verify and unlink will refuse. Investigate, then re-run this stage; already "
                 f"moved files are adopted by the reconcile pass.")
    print("  COMPLETE: every inventoried file is in quarantine with no errors.")

elif a.stage == "verify":
    _q = J["stages"].get("quarantine") or {}
    if not _q.get("complete"):
        sys.exit(f"STOP: quarantine is not complete ({_q.get('in_quarantine')} of "
                 f"{_q.get('inventoried')} moved, {_q.get('errors')} error(s)). Verifying a "
                 f"partial move set would authorise a partial deletion.")
    if not a.verify_cmd:
        sys.exit("STOP: --verify-cmd is required; the point of this stage is an external check")
    print(f"running: {a.verify_cmd}")
    rc = subprocess.run(a.verify_cmd, shell=True, cwd=ROOT).returncode
    J["stages"]["verify"] = {"at": time.time(), "cmd": a.verify_cmd, "returncode": rc}
    save()
    print(f"\nverification exit {rc} -- {'OK, unlink may proceed' if rc == 0 else 'FAILED, do not unlink'}")
    sys.exit(rc)

elif a.stage == "unlink":
    if not a.i_have_verified:
        sys.exit("STOP: --i-have-verified is required. This is the irreversible stage.")
    _q = J["stages"].get("quarantine") or {}
    if not _q.get("complete"):
        sys.exit(f"STOP: quarantine is not complete ({_q.get('in_quarantine')} of "
                 f"{_q.get('inventoried')} moved, {_q.get('errors')} error(s)). Refusing a "
                 f"partial deletion.")
    v = J["stages"].get("verify")
    if not v or v.get("returncode") != 0:
        sys.exit("STOP: no passing verify stage in the journal. Run --stage verify first.")
    # AND the set must still be fully accounted for at unlink time, not merely when quarantine ran
    _n = len(J["files"])
    _inq = sum(1 for r in J["files"] if r.get("quarantined"))
    if _inq != _n:
        sys.exit(f"STOP: {_inq} of {_n} files are marked quarantined now; the journal changed "
                 f"since the quarantine stage. Re-run quarantine to reconcile.")
    before = quota_kib()
    if before is None:
        sys.exit("STOP: quota -v could not be read, so the reclaimed space cannot be verified. "
                 "Refusing the irreversible stage rather than deleting first and discovering "
                 "afterwards that the check is unavailable.")
    freed = 0
    for rec in J["files"]:
        if not rec.get("quarantined"):
            continue
        p = os.path.join(ROOT, rec["quarantine_path"])
        if os.path.isfile(p):
            os.unlink(p)
            freed += rec["bytes"]
            rec["unlinked"] = True
    after = quota_kib()
    delta = (before - after) * 1024 if (before is not None and after is not None) else None
    J["stages"]["unlink"] = {"at": time.time(), "bytes_expected": freed,
                             "quota_kib_before": before, "quota_kib_after": after,
                             "quota_bytes_delta": delta}
    save()
    print(f"unlinked {freed/1024**3:.1f} GiB of inventoried data")
    if delta is None:
        print("  quota unreadable; cannot confirm the space was actually released")
        sys.exit(2)
    print(f"  quota delta: {delta/1024**3:.1f} GiB")
    short = freed - delta
    if short > freed * a.tolerance:
        print(f"  SHORTFALL {short/1024**3:.1f} GiB -- the blocks are still charged. Usual cause")
        print("  is an open descriptor on a deleted inode; the space returns when the holder")
        print("  exits. DO NOT budget folding against this space until the delta matches.")
        sys.exit(3)
    print("  delta matches the inventory; the space is genuinely available.")

elif a.stage == "restore":
    n = 0
    for rec in J["files"]:
        if rec.get("unlinked"):
            print(f"  CANNOT RESTORE (already unlinked): {rec['rel']}")
            continue
        if not rec.get("quarantined"):
            continue
        src = os.path.join(ROOT, rec["quarantine_path"])
        dst = os.path.join(ROOT, rec["rel"])
        if os.path.isfile(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            rec["quarantined"] = False
            n += 1
    J["stages"]["restore"] = {"at": time.time(), "restored": n}
    save()
    print(f"restored {n} file(s)")
