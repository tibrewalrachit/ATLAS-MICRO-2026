"""Re-export of the shared operator representation (kept under models/ so the
DAG builders and ATLAS extension share one definition)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from workload_common import Op, Engine, AccessPattern, DecodeWorkload, BYTES  # noqa
