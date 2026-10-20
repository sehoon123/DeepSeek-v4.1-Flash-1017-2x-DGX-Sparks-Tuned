#!/usr/bin/env bash
# RDMA write latency floor between the nodes (perftest ib_write_lat, CPU polling), per rail, while the engine is stopped.
O=$1
for spec in "rocep1s0f1 10.10.10.2 18561" "roceP2p1s0f1 10.10.11.2 18562"; do
  set -- $spec; dev=$1; ip=$2; port=$3
  timeout 25 ssh -n -o BatchMode=yes -o ConnectTimeout=8 worker "nohup setsid timeout 150 ib_write_lat -d $dev -x 5 -a -n 2000 -F -p $port > /tmp/lat-$dev-srv.log 2>&1 < /dev/null & echo server-started"
  sleep 3
  timeout 150 ib_write_lat -d $dev -x 5 -a -n 2000 -F -p $port $ip > $O/lat-$dev.txt 2>&1; echo "$dev client rc=$?"
  sleep 2
done
