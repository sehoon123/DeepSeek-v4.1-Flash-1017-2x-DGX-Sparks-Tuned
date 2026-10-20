#!/usr/bin/env bash
# usage: calib-run.sh OUTDIR SPEC [DEADLINE_S]   SPEC line: name|req|parallel|rank0 flags|rank1 flags|env K=V ...  (@NAME@ -> run name)
set -u
OUT=$1; SPEC=$2; DEADLINE=${3:-3000}; mkdir -p "$OUT"; cd "$(dirname "$(readlink -f "$0")")"; T0=$(date +%s)
IMG=tensorfold-dsv41:exl3-tp2-c28f5fb6b9c2; W=worker
BIN0=/tmp/dsv-tf-hotfix/zig-out-dev/bin/tf-dsv41-lanes; BIN1=/tmp/dsv-tf-dev/zig-out-dev/bin/tf-dsv41-lanes
D=/hf/tensorfold-dsv41/6d54be4f-ablit-0bdc2036; KIT=/hf/tensorfold-dsv41/kit-abe07c7d/kit
EN=/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170; HCAS=rocep1s0f1,roceP2p1s0f1
RUN="--gpus all --ipc=host --network host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --device /dev/infiniband"
NCCL="-e NCCL_IB_GID_INDEX=5 -e NCCL_IB_HCA=$HCAS -e NCCL_NET_PLUGIN=spcx -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e TF_RDMA_GID_INDEX=5"
BASEENV="-e TF_DS_REPLAY_FLOOR=1 -e TF_DS_STREAMS=4 -e HF_HUB_OFFLINE=1 -e TF_DS_PREFILL_INTERLEAVE_MS=250 -e TF_DS_COPY_ROWS=16 -e TF_DS_COPY_SKIP=1"
COMMON="--pool 1048576 --streams 4 --drafts 1 --graphs 1 --side 1 --prefetch 1 --aio 1 --rdma $HCAS"
log(){ echo "[$(date +%T)] $*" | tee -a "$OUT/harness.log"; }
cleanup(){ docker rm -f dsv41-cb-r0 >/dev/null 2>&1; ssh -n -o BatchMode=yes $W 'docker rm -f dsv41-cb-r1 >/dev/null 2>&1'; }
restore(){ [ -n "${WD:-}" ] && kill $WD 2>/dev/null; cleanup; log "restoring production"; systemctl --user reset-failed dsv41-1017.service 2>/dev/null; systemctl --user start dsv41-1017.service; systemctl --user start dsv41-1017-auth-proxy.service
  for i in $(seq 1 120); do curl -sf --max-time 3 http://127.0.0.1:8000/health >/dev/null && { log "production healthy ($(( $(date +%s) - T0 )) s window)"; return; }; sleep 3; done; log "PRODUCTION NOT HEALTHY"; }
python3 - <<'PY' || { echo busy; exit 2; }
import json,urllib.request
h=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5)); assert not (h.get('live') or {}).get('connections',0)
PY
WD=""; trap restore EXIT
timeout 20 ssh -n -o BatchMode=yes $W 'mkdir -p /tmp/nsys-out'; log "stopping production"; systemctl --user stop dsv41-1017-auth-proxy.service; systemctl --user stop dsv41-1017.service; sleep 3; cleanup
( while sleep 1; do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); b=$(ssh -n -o BatchMode=yes $W "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null || echo 99)
  echo "$(date +%T) $a $b" >> "$OUT/mem.log"; if (( a < 3 || b < 3 )); then echo "$(date +%T) WATCHDOG a=$a b=$b" >> "$OUT/harness.log"; docker kill dsv41-cb-r0; ssh -n -o BatchMode=yes $W 'docker kill dsv41-cb-r1'; fi; done ) &
WD=$!
while IFS= read -r -u 3 line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  if [[ "$line" == '!'* ]]; then log "cmd ${line:1}"; bash -c "${line:1}" >> "$OUT/harness.log" 2>&1 || log "CMD FAILED"; continue; fi
  IFS='|' read -r name req par x0 x1 envs maxs <<< "$line"; maxs=${maxs:-900}
  if (( $(date +%s) - T0 > DEADLINE )); then log "deadline: skipping $name"; continue; fi
  envs=${envs//@NAME@/$name}; P0=""; M0=""; P1=""; M1=""; CP=""; re=' CPUSET=([^ ]+) '
  if [[ " $envs " =~ $re ]]; then CP="--cpuset-cpus=${BASH_REMATCH[1]}"; envs=${envs/CPUSET=${BASH_REMATCH[1]}/}; fi
  if [[ " $envs " == *" NSYS1=1 "* ]]; then
    envs=${envs//NSYS1=1/}; NS1=/opt/nvidia/nsight-systems/2025.3.2
    M1="-v $NS1:/nsys:ro -v /tmp/nsys-out:/nsysout"; P1="/nsys/target-linux-sbsa-armv8/nsys profile -o /nsysout/$name-r1 --force-overwrite=true --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --stats=false"
  fi
  if [[ " $envs " == *" NSYS=1 "* ]]; then
    envs=${envs//NSYS=1/}; NS=/opt/nvidia/nsight-systems/2025.3.2
    NSOPT="--force-overwrite=true --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --stats=false"
    M0="-v $NS:/nsys:ro"; M1="-v $NS:/nsys:ro -v /tmp/nsys-out:/nsysout"
    P0="/nsys/target-linux-sbsa-armv8/nsys profile -o /out/$name-r0 $NSOPT"; P1="/nsys/target-linux-sbsa-armv8/nsys profile -o /nsysout/$name-r1 $NSOPT"
  fi
  E=""; for kv in $envs; do E+=" -e $kv"; done
  log "run $name req=$req par=$par r0='$x0' r1='$x1' env='$envs' cpu='$CP' max=$maxs"
  ssh -n -o BatchMode=yes $W "docker run -d --name dsv41-cb-r1 $RUN $CP -v dsv41-1017-exl3-hf:/hf:ro -v \$HOME/.cache/dsv41-tensorfold/engram-1bae0e10:/engram:ro -v \$HOME/.cache/dsv41-tensorfold/rank-cache-6d54be4f-ablit-0bdc2036:/rank-cache:ro -v $BIN1:/opt/tensorfold/bin/tf-dsv41-lanes:ro $BASEENV $E $NCCL $M1 $IMG $P1 tf-dsv41-lanes $D/checkpoint /rank-cache 1 2 10.10.10.1 29620 $KIT --engram /engram --token-map $D/dsv41_token_map.json $COMMON $x1" >/dev/null || { log "rank1 start failed"; continue; }
  timeout $maxs docker run --name dsv41-cb-r0 $RUN $CP -v /home/USER/.cache/huggingface:/hf:ro -v "$PWD:/work:ro" -v "$OUT:/out" -v $BIN0:/opt/tensorfold/bin/tf-dsv41-lanes:ro $BASEENV $E $NCCL -e TENSORFOLD_CUDA_KERNELS=$KIT/aot $M0 $IMG $P0 \
    tf-dsv41-lanes $D/checkpoint $D/rank-cache 0 2 10.10.10.1 29620 $KIT --requests /work/$req --parallel $par --engram $EN --token-map $D/dsv41_token_map.json $COMMON $x0 < /dev/null > "$OUT/$name.r0.jsonl" 2> "$OUT/$name.r0.err" &
  RP=$!; t1=$(date +%s)
  while kill -0 $RP 2>/dev/null; do
    sleep 5
    if (( $(date +%s) - t1 > 300 )) && ! grep -q '"loaded_s"' "$OUT/$name.r0.jsonl" 2>/dev/null; then
      log "STUCK $name: no engine output after 300 s, killing both ranks"; docker kill dsv41-cb-r0 >/dev/null 2>&1; ssh -n -o BatchMode=yes $W 'docker kill dsv41-cb-r1 >/dev/null 2>&1'; break
    fi
  done
  wait $RP; rc=$?
  log "rank0 rc=$rc"; docker rm -f dsv41-cb-r0 >/dev/null 2>&1
  for i in $(seq 1 90); do ssh -n -o BatchMode=yes $W "docker inspect -f '{{.State.Running}}' dsv41-cb-r1" 2>/dev/null | grep -q true || break; sleep 2; done
  ssh -n -o BatchMode=yes $W "docker logs dsv41-cb-r1" > "$OUT/$name.r1.log" 2>&1; ssh -n -o BatchMode=yes $W 'docker rm -f dsv41-cb-r1 >/dev/null 2>&1'
  python3 summarize.py "$OUT/$name.r0.jsonl" | tee -a "$OUT/harness.log"
done 3< "$SPEC"
