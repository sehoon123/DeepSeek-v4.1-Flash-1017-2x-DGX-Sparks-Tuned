#!/usr/bin/env bash
# Offline rank-0 lane profiles with tf-dsv41-lanes (production stopped while it runs; restored by the trap).
# usage: profile-run.sh OUTDIR "name|reqset|parallel|extra rank0 flags|extra rank1 flags|policy(1=A2P8,0=none)" ...
set -u
OUT=$1; shift; mkdir -p "$OUT"; cd "$(dirname "$0")"
IMG=tensorfold-dsv41:exl3-tp2-da8e1eb22cc6; W=worker
D=/hf/tensorfold-dsv41/6d54be4f-ablit-0bdc2036; KIT=/hf/tensorfold-dsv41/kit-abe07c7d/kit
EN=/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
HCAS=rocep1s0f1,roceP2p1s0f1
RUN="--gpus all --ipc=host --network host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --device /dev/infiniband"
NCCL="-e NCCL_IB_GID_INDEX=5 -e NCCL_IB_HCA=$HCAS -e NCCL_NET_PLUGIN=spcx -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e TF_RDMA_GID_INDEX=5"
BASEENV="-e TF_DS_REPLAY_FLOOR=1 -e TF_DS_STREAMS=4 -e HF_HUB_OFFLINE=1 -e TF_DS_CACHE_PLACEMENT=1 -e TF_DS_PREFILL_INTERLEAVE_MS=250"
POL="-e TF_DS_ADAPT_K=1 -e TF_DS_ADAPT_PERIOD=8 -e TF_DS_ADAPT_ACCEPT=2 -e TF_DS_ADAPT_MAX_LIVE=2"
log(){ echo "[$(date +%T)] $*" | tee -a "$OUT/harness.log"; }
cleanup(){ docker rm -f dsv41-prof-r0 >/dev/null 2>&1; ssh -o BatchMode=yes $W 'docker rm -f dsv41-prof-r1 >/dev/null 2>&1'; }
restore(){ kill $WD 2>/dev/null; cleanup; log "restoring production"; systemctl --user reset-failed dsv41-1017.service 2>/dev/null; systemctl --user start dsv41-1017.service; systemctl --user start dsv41-1017-auth-proxy.service
  for i in $(seq 1 120); do curl -sf --max-time 3 http://127.0.0.1:8000/health >/dev/null && { log "production healthy"; return; }; sleep 3; done; log "PRODUCTION NOT HEALTHY"; }
python3 - <<'PY' || { echo busy; exit 2; }
import json,urllib.request
h=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5)); assert not (h.get('live') or {}).get('connections',0)
PY
WD=""; trap restore EXIT
log "stopping production"; systemctl --user stop dsv41-1017-auth-proxy.service; systemctl --user stop dsv41-1017.service; sleep 3; cleanup
# watchdog: kill the profile containers if either node's MemAvailable < 3 GiB
( while sleep 1; do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); b=$(ssh -o BatchMode=yes $W "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null || echo 99)
  echo "$(date +%T) $a $b" >> "$OUT/mem.log"; if (( a < 3 || b < 3 )); then echo "$(date +%T) WATCHDOG kill a=$a b=$b" >> "$OUT/harness.log"; docker kill dsv41-prof-r0; ssh -o BatchMode=yes $W 'docker kill dsv41-prof-r1'; fi; done ) &
WD=$!
for spec in "$@"; do
  IFS='|' read -r name set par x0 x1 pol <<< "$spec"
  P=""; [[ "$pol" == 1 ]] && P="$POL"
  FOLLOW="--pool 1048576 --streams 4 --drafts 1 --graphs 1 --side 1 --prefetch 1 --aio 1 --profile 1 $x1"
  log "run $name: set=$set parallel=$par r0='$x0' r1='$x1' policy=$pol"
  ssh -o BatchMode=yes $W "docker run -d --name dsv41-prof-r1 $RUN -v dsv41-1017-exl3-hf:/hf:ro -v \$HOME/.cache/dsv41-tensorfold/engram-1bae0e10:/engram:ro -v \$HOME/.cache/dsv41-tensorfold/rank-cache-6d54be4f-ablit-0bdc2036:/rank-cache:ro $BASEENV $P $NCCL $IMG tf-dsv41-lanes $D/checkpoint /rank-cache 1 2 10.10.10.1 29620 $KIT --engram /engram --token-map $D/dsv41_token_map.json $FOLLOW --rdma $HCAS" >/dev/null || { log "rank1 start failed"; continue; }
  timeout 900 docker run --name dsv41-prof-r0 $RUN -v /home/USER/.cache/huggingface:/hf:ro -v "$PWD:/work:ro" $BASEENV $P $NCCL -e TENSORFOLD_CUDA_KERNELS=$KIT/aot $IMG \
    tf-dsv41-lanes $D/checkpoint $D/rank-cache 0 2 10.10.10.1 29620 $KIT --requests /work/req-$set.json --parallel $par --pool 1048576 --streams 4 --drafts 1 --graphs 1 --side 1 --prefetch 1 --aio 1 --rdma $HCAS --engram $EN --token-map $D/dsv41_token_map.json --profile 1 $x0 > "$OUT/$name.r0.jsonl" 2> "$OUT/$name.r0.err"; rc=$?
  log "rank0 rc=$rc"; docker rm -f dsv41-prof-r0 >/dev/null 2>&1
  for i in $(seq 1 30); do ssh -o BatchMode=yes $W "docker inspect -f '{{.State.Running}}' dsv41-prof-r1" 2>/dev/null | grep -q true || break; sleep 2; done
  ssh -o BatchMode=yes $W "docker logs dsv41-prof-r1" > "$OUT/$name.r1.log" 2>&1; ssh -o BatchMode=yes $W 'docker rm -f dsv41-prof-r1 >/dev/null 2>&1'
  grep -h '"profile"\|"clock\|error' "$OUT/$name.r0.jsonl" | cut -c1-600 | tee -a "$OUT/harness.log"
done
