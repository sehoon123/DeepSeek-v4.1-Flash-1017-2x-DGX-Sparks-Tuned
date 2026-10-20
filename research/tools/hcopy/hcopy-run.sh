#!/usr/bin/env bash
# H-COPY validation with the dev-built tf-dsv41-lanes (production stopped; the trap restores it).
# usage: hcopy-run.sh OUTDIR    (phases: phase-a.txt, then make-b.py, then phase-b.txt; line: name|req|parallel|rank0 flags|env K=V ...)
set -u
OUT=$1; mkdir -p "$OUT"; cd "$(dirname "$0")"
IMG=tensorfold-dsv41:exl3-tp2-da8e1eb22cc6; W=worker
BIN0=/tmp/dsv-tf-hotfix/zig-out-dev/bin/tf-dsv41-lanes; BIN1=/tmp/dsv-tf-dev/zig-out-dev/bin/tf-dsv41-lanes
D=/hf/tensorfold-dsv41/6d54be4f-ablit-0bdc2036; KIT=/hf/tensorfold-dsv41/kit-abe07c7d/kit
EN=/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170; HCAS=rocep1s0f1,roceP2p1s0f1
RUN="--gpus all --ipc=host --network host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --device /dev/infiniband"
NCCL="-e NCCL_IB_GID_INDEX=5 -e NCCL_IB_HCA=$HCAS -e NCCL_NET_PLUGIN=spcx -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e TF_RDMA_GID_INDEX=5"
BASEENV="-e TF_DS_REPLAY_FLOOR=1 -e TF_DS_STREAMS=4 -e HF_HUB_OFFLINE=1 -e TF_DS_CACHE_PLACEMENT=1 -e TF_DS_PREFILL_INTERLEAVE_MS=250 -e TF_DS_ADAPT_K=1 -e TF_DS_ADAPT_PERIOD=8 -e TF_DS_ADAPT_ACCEPT=2 -e TF_DS_ADAPT_MAX_LIVE=2"
COMMON="--pool 1048576 --streams 4 --drafts 1 --graphs 1 --side 1 --prefetch 1 --aio 1 --rdma $HCAS"
log(){ echo "[$(date +%T)] $*" | tee -a "$OUT/harness.log"; }
cleanup(){ docker rm -f dsv41-hc-r0 >/dev/null 2>&1; ssh -o BatchMode=yes $W 'docker rm -f dsv41-hc-r1 >/dev/null 2>&1'; }
restore(){ [ -n "${WD:-}" ] && kill $WD 2>/dev/null; cleanup; log "restoring production"; systemctl --user reset-failed dsv41-1017.service 2>/dev/null; systemctl --user start dsv41-1017.service; systemctl --user start dsv41-1017-auth-proxy.service
  for i in $(seq 1 120); do curl -sf --max-time 3 http://127.0.0.1:8000/health >/dev/null && { log "production healthy"; return; }; sleep 3; done; log "PRODUCTION NOT HEALTHY"; }
runphase(){ while IFS='|' read -r -u 3 name req par x0 envs; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  E=""; for kv in $envs; do E+=" -e $kv"; done
  log "run $name req=$req par=$par r0='$x0' env='$envs'"
  ssh -n -o BatchMode=yes $W "docker run -d --name dsv41-hc-r1 $RUN -v dsv41-1017-exl3-hf:/hf:ro -v \$HOME/.cache/dsv41-tensorfold/engram-1bae0e10:/engram:ro -v \$HOME/.cache/dsv41-tensorfold/rank-cache-6d54be4f-ablit-0bdc2036:/rank-cache:ro -v $BIN1:/opt/tensorfold/bin/tf-dsv41-lanes:ro $BASEENV $E $NCCL $IMG tf-dsv41-lanes $D/checkpoint /rank-cache 1 2 10.10.10.1 29620 $KIT --engram /engram --token-map $D/dsv41_token_map.json $COMMON" >/dev/null || { log "rank1 start failed"; continue; }
  timeout 1200 docker run --name dsv41-hc-r0 $RUN -v /home/USER/.cache/huggingface:/hf:ro -v "$PWD:/work:ro" -v $BIN0:/opt/tensorfold/bin/tf-dsv41-lanes:ro $BASEENV $E $NCCL -e TENSORFOLD_CUDA_KERNELS=$KIT/aot $IMG \
    tf-dsv41-lanes $D/checkpoint $D/rank-cache 0 2 10.10.10.1 29620 $KIT --requests /work/$req --parallel $par --engram $EN --token-map $D/dsv41_token_map.json $COMMON $x0 < /dev/null > "$OUT/$name.r0.jsonl" 2> "$OUT/$name.r0.err"; rc=$?
  log "rank0 rc=$rc"; docker rm -f dsv41-hc-r0 >/dev/null 2>&1
  for i in $(seq 1 30); do ssh -n -o BatchMode=yes $W "docker inspect -f '{{.State.Running}}' dsv41-hc-r1" 2>/dev/null | grep -q true || break; sleep 2; done
  ssh -n -o BatchMode=yes $W "docker logs dsv41-hc-r1" > "$OUT/$name.r1.log" 2>&1; ssh -n -o BatchMode=yes $W 'docker rm -f dsv41-hc-r1 >/dev/null 2>&1'
  python3 summarize.py "$OUT/$name.r0.jsonl" | tee -a "$OUT/harness.log"
done 3< "$1"; }
python3 - <<'PY' || { echo busy; exit 2; }
import json,urllib.request
h=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5)); assert not (h.get('live') or {}).get('connections',0)
PY
WD=""; trap restore EXIT
log "stopping production"; systemctl --user stop dsv41-1017-auth-proxy.service; systemctl --user stop dsv41-1017.service; sleep 3; cleanup
( while sleep 1; do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); b=$(ssh -o BatchMode=yes $W "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null || echo 99)
  echo "$(date +%T) $a $b" >> "$OUT/mem.log"; if (( a < 3 || b < 3 )); then echo "$(date +%T) WATCHDOG a=$a b=$b" >> "$OUT/harness.log"; docker kill dsv41-hc-r0; ssh -o BatchMode=yes $W 'docker kill dsv41-hc-r1'; fi; done ) &
WD=$!
runphase phase-a.txt
python3 make-b.py "$OUT" > "$OUT/make-b.out" 2>&1 || { log "make-b failed: $(tail -n 2 $OUT/make-b.out)"; exit 1; }; cat "$OUT/make-b.out" | tee -a "$OUT/harness.log"
runphase phase-b.txt
