#!/bin/bash
# start.sh's rank 1 watch loop (the text between "( exec 8>&-" and "done ) &" in foreground()), run with stubs: a failed
# ssh (255) or one "not running" reading leaves rank 0 alone; two "not running" readings in a row stop it (exit 1).
# Usage: watch_rank1_test.sh RECIPE_DIR
set -u
R=$1
loop=$(awk '/\( exec 8>&-/{f=1} f{print} /done \) &/{if(f) exit}' "$R/start.sh" | sed '1s/^ *( exec 8>&-.*$/(/; $s/done ) &/done )/')
[[ -n "$loop" ]] || { echo "FAIL: loop not found"; exit 1; }
source "$R/scripts/nodes.sh"
CONTAINER_NAME=c
fails=0
run() {   # run NAME EXPECTED_EXIT EXPECTED_STOPS SEQUENCE...: worker answers in turn: 255, true, false or gone
  local name=$1 want=$2 stops_want=$3; shift 3
  printf '%s\n' "$@" > seq; : > stops; echo 0 > i; echo 0 > here
  worker() { local k; k=$(($(cat i) + 1)); echo $k > i; local a; a=$(sed -n "${k}p" seq)
    case $a in 255) return 255 ;; true) echo true ;; false) echo false ;; gone) return 1 ;; *) return 255 ;; esac; }
  running_here() { local k; k=$(($(cat here) + 1)); echo $k > here; (( k <= $(wc -l < seq) )); }
  docker() { echo "$*" >> stops; }
  warn() { :; }
  worker_host() { echo w; }
  running_worker() { [[ "$(worker "$1" docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }   # start.sh's own, for the loop before bundle 14
  sleep() { :; }
  eval "$loop"; local got=$?
  local stops; stops=$(wc -l < stops)
  if [[ $got == "$want" && $stops == "$stops_want" ]]; then echo "ok   $name (exit $got, stops $stops)"
  else echo "FAIL $name: exit $got (want $want), stops $stops (want $stops_want)"; fails=$((fails + 1)); fi
}
run "a failed ssh between running readings" 0 0 true 255 true 255 true
run "one not-running reading" 0 0 true false true true
run "a failed ssh between two not-running readings" 0 0 false 255 true
run "two not-running readings" 1 1 true false false true
run "rank 1's container gone twice" 1 1 gone gone
exit $fails
