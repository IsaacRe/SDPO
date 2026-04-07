#!/bin/bash
set -u

while true; do
  clear
  date -u
  echo

  LATEST=$(ls -dt /tmp/ray/session_* 2>/dev/null | head -n1)
  if [[ -z "${LATEST:-}" ]]; then
    echo "no ray session found"
    sleep 2
    continue
  fi

  echo "session: $LATEST"
  echo

  echo "latest reward progress:"
  grep -h '\[reward\] completed' "$LATEST"/logs/worker-*.out 2>/dev/null | tail -n 20
  echo

  echo "latest training lines:"
  grep -hE 'Training Progress|step:' "$LATEST"/logs/worker-* "$LATEST"/logs/*.err 2>/dev/null | tail -n 20
  echo

  echo "gpu:"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

  sleep 5
done

