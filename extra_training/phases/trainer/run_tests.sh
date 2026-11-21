#!/usr/bin/env bash
# The trainer copy's CPU test suite.
#
# RUN FROM extra_training/, NOT FROM HERE. Several fixtures carry RELATIVE paths exactly as the
# live arm passes them (--rel-labels training/fold/itm_labels_extended), so from any other
# directory config.py correctly refuses them and the test reports a failure that is really a cwd.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=/home/ebrahim/Protenix/.venv/bin/python
fail=0
for t in phases/trainer/test_*.py; do
  printf "  %-36s " "$(basename "$t")"
  if LAYERNORM_TYPE=openfold PYTHONPATH=phases/trainer:phases $PY "$t" >/tmp/.tt 2>&1; then
    echo PASS
  else
    echo FAIL; sed 's/^/      /' /tmp/.tt | grep -E "FAIL|STOP" | tail -3; fail=1
  fi
done
exit $fail
