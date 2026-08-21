"""MiniMax M3 entry point for the analytical simulator (explicit-path load)."""
import importlib.util, os
_p = os.path.join(os.path.dirname(__file__), "..", "models", "minimax_m3", "dag.py")
_spec = importlib.util.spec_from_file_location("m3_dag", _p)
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
build_decode_dag, total_params, kv_capacity_per_user = \
    _m.build_decode_dag, _m.total_params, _m.kv_capacity_per_user
