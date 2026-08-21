# ASSUMPTIONS — PULP tile on CUBE

| id | assumption | low / base / high | rationale |
|---|---|---|---|
| P1 | Tile frequency 1 GHz | 0.8 / 1.0 / 1.2 GHz | RedMulE GF22 silicon corners span ~0.47-1.19 GHz; advanced node assumed for Fabrik. Swept. |
| P2 | Snitch FP16 SIMD rate 8 FLOP/clk/core | 4 / 8 / 8 | ISA flags (xfvec, xfdotp, zfh) prove capability; sustained rate inferred, not measured. |
| P3 | RedMulE scaled variant (16x32 = 512 CE) | 48 / 512 / 768 CE | Parametrically legal (WIDTH <= HEIGHTxPIPE_REGS with HEIGHT=16); no silicon reference at this size; timing closure unproven. |
| P4 | FP8 rate = FP16 rate on RedMulE | exact | Sourced (cast-to-FP16 architecture) - listed here because it is a *disadvantage* vs Tensix FP8 and central to the comparison. |
| P5 | 4 PULP tiles per CUBE stack | 2 / 4 / 8 | Matches 64-tile/16-stack brief design point; area not modeled. |
| P6 | ATLAS power fields carried from stock test chip | n/a | This study compares latency/utilization only; PULP-tile energy needs per-block numbers from the papers (future work). |
| P7 | v4flash_proxy MLA mapping (kv_lora=448, rope=64) | n/a | Proxy reproduces V4's 512B/token/layer latent read + 64-head geometry; indexer/compressor NOT modeled in proxy (exact path exists analytically in models/deepseek_v4_flash). |
