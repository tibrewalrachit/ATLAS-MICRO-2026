"""Lightweight serving simulator (Part XXI).

Event-driven continuous batching on one Fabrik device, batch cap 4.
Decode-step latency comes from the analytical model, re-evaluated whenever
batch composition changes (latency depends on B and, weakly, on context mix:
we use the max context among active sessions for attention terms — worst-case
within the step, labeled).

Reports p50/p95/p99 TPOT, per-user TPS, aggregate TPS, utilization, queueing.
"""
import heapq, random, statistics, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "models"))
sys.path.insert(0, os.path.join(HERE, "..", "analytical"))
from model import decode_step


class ServingSim:
    def __init__(self, workload_builder, hw, batch_cap=4, reuse="statistical"):
        self.wb = workload_builder
        self.hw = hw
        self.cap = batch_cap
        self.reuse = reuse
        self._lat_cache = {}

    def _step_latency(self, batch, ctx_bucket):
        key = (batch, ctx_bucket)
        if key not in self._lat_cache:
            wl = self.wb(ctx_bucket)
            self._lat_cache[key] = decode_step(wl, self.hw, batch, self.reuse).latency_s
        return self._lat_cache[key]

    @staticmethod
    def _bucket(ctx):
        for b in (8192, 32768, 131072, 524288, 1048576):
            if ctx <= b:
                return b
        return 1048576

    def run(self, arrival_rate_per_s, mean_output_tokens, context_dist,
            sim_seconds=200.0, seed=0):
        rng = random.Random(seed)
        t = 0.0
        arrivals = []
        while t < sim_seconds:
            t += rng.expovariate(arrival_rate_per_s)
            ctx = context_dist(rng)
            out = max(1, int(rng.expovariate(1.0 / mean_output_tokens)))
            arrivals.append((t, ctx, out))
        queue = []          # waiting sessions
        active = []         # [ctx, remaining, tpots, enqueue_time, start_time]
        now = 0.0
        ai = 0
        tpot_all, per_user_tps, queue_delays = [], [], []
        busy_time = 0.0
        while ai < len(arrivals) or active or queue:
            if not active and not queue:
                now = max(now, arrivals[ai][0])
            while ai < len(arrivals) and arrivals[ai][0] <= now:
                queue.append([arrivals[ai][1], arrivals[ai][2], [], arrivals[ai][0], None])
                ai += 1
            while queue and len(active) < self.cap:
                s = queue.pop(0)
                s[4] = now
                queue_delays.append(now - s[3])
                active.append(s)
            if not active:
                continue
            B = len(active)
            ctxb = self._bucket(max(s[0] for s in active))
            lat = self._step_latency(B, ctxb)
            now += lat
            busy_time += lat
            done = []
            for s in active:
                s[0] += 1; s[1] -= 1
                s[2].append(lat)
                if s[1] <= 0:
                    done.append(s)
            for s in done:
                active.remove(s)
                tpot_all.extend(s[2])
                gen_time = sum(s[2])
                per_user_tps.append(len(s[2]) / gen_time if gen_time > 0 else 0.0)
        def pct(v, p):
            v = sorted(v); return v[min(len(v) - 1, int(p / 100 * len(v)))] if v else None
        total_tokens = len(tpot_all)
        return {
            "sessions": len(per_user_tps), "tokens": total_tokens,
            "p50_tpot_ms": pct(tpot_all, 50) * 1e3, "p95_tpot_ms": pct(tpot_all, 95) * 1e3,
            "p99_tpot_ms": pct(tpot_all, 99) * 1e3,
            "p50_tps_user": pct(per_user_tps, 50), "p95_low_tps_user": pct(per_user_tps, 5),
            "aggregate_tps": total_tokens / busy_time if busy_time else 0.0,
            "device_busy_frac": busy_time / max(now, 1e-9),
            "p50_queue_ms": pct(queue_delays, 50) * 1e3, "p99_queue_ms": pct(queue_delays, 99) * 1e3,
        }
