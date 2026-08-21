"""Validation anchors for the PULP HWPE cycle models (<=10% rule)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pulp_sim"))
import redmule_model as rm
import neureka_model as nk


def test_redmule_paper_anchor():
    u = rm.utilization(96, 96, 96, allow_swap=False)
    assert abs(u - 0.994) < 0.01, u          # paper: 99.4% @ 96^3


def test_redmule_swap_mapping():
    assert rm.utilization(1, 2048, 4096) > 0.95
    assert rm.utilization(1, 2048, 4096, allow_swap=False) < 0.1


def test_neureka_bit_serial_and_spatial():
    assert nk.effective_macs_per_cycle(1, 4) == 256          # B=1: one PE
    assert nk.effective_macs_per_cycle(36, 4) == 9216        # full array
    assert nk.gemv_cycles(4096, 2048, 1, 2) < nk.gemv_cycles(4096, 2048, 1, 8)


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
