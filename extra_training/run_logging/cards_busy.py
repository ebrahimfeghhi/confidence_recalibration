# Count processes occupying the trainer's cards that are NOT the three GPU holders.
# Reads GPU occupancy, never command-line text: the string a text check looks for is present in
# the checking command itself, which is how the queued launch refused against nothing.
import subprocess, sys
HOLDERS = {306042, 306045, 306050}
out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout
apps = []
for line in out.strip().splitlines():
    if not line.strip():
        continue
    pid, uuid = [x.strip() for x in line.split(",")[:2]]
    apps.append((int(pid), uuid))
ours = {u for p, u in apps if p in HOLDERS}
intruders = [(p, u) for p, u in apps if p not in HOLDERS and u in ours]
for p, u in intruders:
    print(f"  occupying {u}: pid {p}", file=sys.stderr)
print(len(intruders))
