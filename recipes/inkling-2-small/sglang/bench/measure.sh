#!/bin/bash
# measure.sh — send the timing workload to the running fleet and print the [timers] lines it produced (rank 0).
#   ./measure.sh <label>     (prefill: a ~15.6K-token prompt; decode: a 300-token coding answer)
#   Needs a boot with -e SPARK_TIMERS=1 (NCCL_EXTRA="-e SPARK_TIMERS=1"; a test window, not a serving default).
# MIT License, Copyright (c) 2026 Aiden Le.
S=$(cd "$(dirname "$0")" && pwd)/logs; mkdir -p $S; L=$S/measure-$1-$(date +%H%M%S).log
since=$(date -u +%Y-%m-%dT%H:%M:%S)
python3 - > $S/long.json <<'PY'
import json, random
random.seed(0)
w = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
print(json.dumps({"model": "inkling-2-small", "messages": [{"role": "user", "content": " ".join(random.choice(w) for _ in range(12000)) + "\n\nHow many words above? One number."}], "max_tokens": 1}))
PY
t=$(curl -s -m900 -o /dev/null -w '%{time_total}' 127.0.0.1:8210/v1/chat/completions -H 'Content-Type: application/json' -d @$S/long.json)
echo "prefill request: ${t}s" | tee $L
t=$(curl -s -m900 -o $S/decode.json -w '%{time_total}' 127.0.0.1:8210/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"inkling-2-small","messages":[{"role":"user","content":"Write a Python function that checks whether a string is a palindrome, ignoring case and punctuation. Include a short docstring and examples."}],"max_tokens":'${DECODE_TOKENS:-300}',"temperature":0}')
echo "decode request: ${t}s, $(python3 -c "import json;print(json.load(open('$S/decode.json'))['usage']['completion_tokens'])") tokens" | tee -a $L
. "${FLEET_ENV:-$(dirname "$0")/../launch/fleet.env}"; docker logs --since $since inkling2 2>&1 | grep "\[timers\]" >> $L; for r in 1 2 3; do ssh -o BatchMode=yes ${USERS[$r]}@${NODES[$r]} "docker logs --since $since inkling2 2>&1 | grep \"\\[timers\\]\"" >> $L; done
echo "timer lines: $(grep -c '\[timers\]' $L) -> $L"
