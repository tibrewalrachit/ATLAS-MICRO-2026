"""TCO model (Part XXIII) + NRE amortization (Part XXIV). Formulas per spec."""
from dataclasses import dataclass

@dataclass
class TCOParams:
    hardware_cost_usd: float = 25000.0
    lifetime_years: float = 4.0
    system_power_kw: float = 0.8
    pue: float = 1.25
    electricity_usd_kwh: float = 0.08
    fleet_utilization: float = 0.6
    other_hourly_usd: float = 0.15      # network/maintenance/host share

def cost_per_1m_tokens(aggregate_tps: float, p: TCOParams) -> dict:
    capex_h = p.hardware_cost_usd / (8760 * p.lifetime_years)
    power_h = p.system_power_kw * p.pue * p.electricity_usd_kwh
    total_h = capex_h + power_h + p.other_hourly_usd
    tokens_h = 3600 * aggregate_tps * p.fleet_utilization
    return {"capex_per_hour": capex_h, "power_per_hour": power_h,
            "total_hourly_cost": total_h,
            "cost_per_1m_output_tokens": total_h / tokens_h * 1e6 if tokens_h else float("inf")}

def effective_card_cost(manufacturing_cost: float, nre_usd: float, deployed_units: int) -> float:
    return manufacturing_cost + nre_usd / deployed_units
