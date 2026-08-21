"""V4-Flash entry point for the analytical simulator (loads by explicit path
to avoid shadowing the models/deepseek_v4_flash package name)."""
import importlib.util, os
_p = os.path.join(os.path.dirname(__file__), "..", "models", "deepseek_v4_flash", "dag.py")
_spec = importlib.util.spec_from_file_location("dsv4_dag", _p)
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
build_decode_dag, total_params, kv_capacity_per_user = \
    _m.build_decode_dag, _m.total_params, _m.kv_capacity_per_user
