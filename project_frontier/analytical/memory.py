"""Memory-system model (Part IX/XI).

Effective bandwidth = peak x efficiency(access pattern). Efficiencies are
*parameters with provenance*: initialized from literature-typical values
(assumption_level=assumed), to be replaced by Ramulator-measured values
(assumption_level=cycle_calibrated) produced by the Part XI experiments.
Never conflated with peak BW in any reported number.
"""
from dataclasses import dataclass, field

# access-pattern efficiency: fraction of peak BW achieved
DEFAULT_EFF = {
    "seq_stream": 0.85,     # long weight streams, open-row hits dominate
    "block_gather": 0.65,   # >=576B..128KB contiguous chunks, scattered rows
    "random_gather": 0.25,  # fine-grained scatter (KV slot writes, embeds)
    "scan": 0.85,           # index scans are sequential
    "on_chip": 1.0,
}
EFF_RANGES = {  # low / base / high for uncertainty analysis (Part XXXIV)
    "seq_stream": (0.70, 0.85, 0.93),
    "block_gather": (0.45, 0.65, 0.80),
    "random_gather": (0.10, 0.25, 0.45),
    "scan": (0.70, 0.85, 0.93),
}


@dataclass
class MemorySystem:
    peak_bw: float                     # bytes/s
    capacity: float                    # bytes
    eff: dict = field(default_factory=lambda: dict(DEFAULT_EFF))
    calibration_source: str = "assumed_defaults"

    def time_for(self, bytes_by_pattern: dict) -> float:
        """Seconds to move the given bytes, pattern-aware. Patterns share the
        same channels; times add (worst case, no inter-pattern overlap)."""
        return sum(b / (self.peak_bw * self.eff[p])
                   for p, b in bytes_by_pattern.items() if b > 0 and p != "on_chip")

    def achieved_bw(self, bytes_by_pattern: dict) -> float:
        t = self.time_for(bytes_by_pattern)
        tot = sum(b for p, b in bytes_by_pattern.items() if p != "on_chip")
        return tot / t if t > 0 else 0.0
