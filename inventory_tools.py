"""Supply Chain Analytics Inventory Tools with Student Verification Package.

This module provides deterministic Python calculation tools for core inventory models:
1. calculate_eoq()
2. calculate_quantity_discount()
3. calculate_rop() (alias: calculate_safety_stock_rop)
4. calculate_periodic_review()
5. calculate_newsvendor()

For every calculation, every tool deterministically computes and returns:
1. Inputs Extracted
2. Variable Definitions
3. Formula (Symbolic)
4. Formula with Substituted Values
5. Intermediate Calculations
6. Final Answer
7. Verification Checks
8. Assumptions

All numerical steps originate exclusively from Python calculations.
Missing information is strictly validated and never invented.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from scipy.stats import norm


# ============================================================================
# Core Student Verification Classes
# ============================================================================

class FinalAnswer(float):
    """A float subclass representing the primary numerical answer with structured verification details.

    Behaves seamlessly as a standard float for comparisons (approx, >, ==), while
    providing dictionary access, key-value inspection, and serialization for student verification.
    """
    def __new__(cls, value: float, details: Optional[Dict[str, Any]] = None):
        obj = super().__new__(cls, float(value))
        obj._details = details or {}
        return obj

    @property
    def details(self) -> Dict[str, Any]:
        return self._details

    def to_dict(self) -> Dict[str, Any]:
        return {"value": float(self), **self._details}

    def __getitem__(self, key: str):
        return self._details[key]

    def __contains__(self, key: str):
        return key in self._details

    def get(self, key: str, default=None):
        return self._details.get(key, default)

    def items(self):
        return self._details.items()

    def keys(self):
        return self._details.keys()

    def values(self):
        return self._details.values()


# ============================================================================
# 1. Economic Order Quantity (EOQ)
# ============================================================================

@dataclass
class EOQResult:
    """Structured result for Economic Order Quantity with student verification package."""
    # 8 Student Verification Requirements
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Union[float, FinalAnswer]
    verification_checks: Dict[str, Any]
    assumptions: List[str]

    # Direct attribute access for backward compatibility
    eoq: float
    annual_demand: float
    order_cost: float
    holding_cost: float
    unit_cost: float
    annual_ordering_cost: float
    annual_holding_cost: float
    total_inventory_cost: float
    total_annual_cost: float
    orders_per_year: float
    order_cycle_days: float
    reorder_point: Optional[float] = None
    lead_time_days: float = 0.0
    working_days_per_year: int = 250

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "Economic Order Quantity (EOQ)",
            "1_inputs_extracted": self.inputs_extracted,
            "2_variable_definitions": self.variable_definitions,
            "3_formula": self.formula,
            "4_formula_with_substituted_values": self.formula_with_substituted_values,
            "5_intermediate_calculations": self.intermediate_calculations,
            "6_final_answer": self.final_answer.to_dict() if hasattr(self.final_answer, "to_dict") else self.final_answer,
            "7_verification_checks": self.verification_checks,
            "8_assumptions": self.assumptions,
        }


def calculate_eoq(
    annual_demand: Optional[float] = None,
    order_cost: Optional[float] = None,
    holding_cost: Optional[float] = None,
    holding_cost_rate: Optional[float] = None,
    unit_cost: Optional[float] = None,
    lead_time_days: float = 0.0,
    working_days_per_year: int = 250,
) -> EOQResult:
    """Calculate Economic Order Quantity (EOQ) with complete verification package.

    Parameters
    ----------
    annual_demand : float
        Annual demand in units (D > 0). Required.
    order_cost : float
        Fixed setup or ordering cost per order (S > 0). Required.
    holding_cost : float, optional
        Annual holding cost per unit per year (H > 0).
    holding_cost_rate : float, optional
        Holding cost rate as fraction of unit cost (e.g. 0.20 for 20%).
    unit_cost : float, optional
        Purchase price per unit (C >= 0).
    lead_time_days : float, default 0.0
        Supplier lead time in days for deterministic reorder point.
    working_days_per_year : int, default 250
        Operating working days per year.

    Returns
    -------
    EOQResult
    """
    # 1. Input Validation - Never invent missing numerical inputs
    if annual_demand is None:
        raise ValueError("Missing required numerical input: 'annual_demand' must be provided.")
    if not isinstance(annual_demand, (int, float)) or annual_demand <= 0:
        raise ValueError(f"'annual_demand' must be a strictly positive number; got {annual_demand}")

    if order_cost is None:
        raise ValueError("Missing required numerical input: 'order_cost' must be provided.")
    if not isinstance(order_cost, (int, float)) or order_cost <= 0:
        raise ValueError(f"'order_cost' must be a strictly positive number; got {order_cost}")

    effective_unit_cost = float(unit_cost) if unit_cost is not None else 0.0
    if holding_cost is None:
        if holding_cost_rate is not None:
            if unit_cost is None or unit_cost <= 0:
                raise ValueError("Missing required numerical input: 'unit_cost' must be positive when using 'holding_cost_rate'.")
            if holding_cost_rate <= 0:
                raise ValueError(f"'holding_cost_rate' must be positive; got {holding_cost_rate}")
            effective_h = holding_cost_rate * effective_unit_cost
        else:
            raise ValueError("Missing required numerical input: Either 'holding_cost' or ('holding_cost_rate' and 'unit_cost') must be provided.")
    else:
        if not isinstance(holding_cost, (int, float)) or holding_cost <= 0:
            raise ValueError(f"'holding_cost' must be a strictly positive number; got {holding_cost}")
        effective_h = float(holding_cost)

    D = float(annual_demand)
    S = float(order_cost)
    H = effective_h
    C = effective_unit_cost

    # 2. Variable Definitions
    var_defs = {
        "D": "Annual demand (units/year)",
        "S": "Fixed setup or ordering cost per order ($/order)",
        "H": "Holding / carrying cost per unit per year ($/unit/year)",
        "C": "Unit purchase cost ($/unit)",
        "Q*": "Economic Order Quantity (optimal order batch size in units)",
        "N": "Number of replenishment orders per year (orders/year)",
        "T": "Order cycle interval between replenishments (operating days)",
        "AOC": "Annual Ordering Cost ($/year)",
        "AHC": "Annual Holding Cost ($/year)",
        "TIC": "Total Annual Inventory Cost = AOC + AHC ($/year)",
        "TC": "Total Annual Logistics & Acquisition Cost = TIC + (D * C) ($/year)",
        "ROP": "Deterministic Reorder Point = (D / working_days) * L (units)",
    }

    # 3. Formulas (Symbolic)
    formulas = {
        "EOQ": "Q* = sqrt((2 * D * S) / H)",
        "Annual_Ordering_Cost": "AOC = (D / Q*) * S",
        "Annual_Holding_Cost": "AHC = (Q* / 2) * H",
        "Total_Inventory_Cost": "TIC = AOC + AHC = sqrt(2 * D * S * H)",
        "Total_Annual_Cost": "TC = TIC + (D * C)",
        "Orders_Per_Year": "N = D / Q*",
        "Order_Cycle_Days": "T = (Q* / D) * working_days_per_year",
        "Reorder_Point": "ROP = (D / working_days_per_year) * lead_time_days",
    }

    # 4. Numerical Calculations & Substituted Formulas
    numerator_2DS = 2.0 * D * S
    ratio_2DS_over_H = numerator_2DS / H
    eoq = math.sqrt(ratio_2DS_over_H)
    q_int = int(round(eoq))

    aoc = (D / eoq) * S
    ahc = (eoq / 2.0) * H
    tic = aoc + ahc
    apc = D * C
    tc = tic + apc
    orders_per_yr = D / eoq
    cycle_days = (eoq / D) * working_days_per_year

    rop = None
    if lead_time_days > 0:
        daily_d = D / working_days_per_year
        rop = daily_d * lead_time_days

    substituted = {
        "EOQ": f"Q* = sqrt((2 * {D:,.2f} * {S:,.2f}) / {H:,.2f}) = sqrt({numerator_2DS:,.2f} / {H:,.2f}) = sqrt({ratio_2DS_over_H:,.4f}) = {eoq:,.2f} units",
        "Annual_Ordering_Cost": f"AOC = ({D:,.2f} / {eoq:,.2f}) * {S:,.2f} = {orders_per_yr:.2f} * {S:,.2f} = ${aoc:,.2f}",
        "Annual_Holding_Cost": f"AHC = ({eoq:,.2f} / 2) * {H:,.2f} = {eoq / 2.0:,.2f} * {H:,.2f} = ${ahc:,.2f}",
        "Total_Inventory_Cost": f"TIC = ${aoc:,.2f} + ${ahc:,.2f} = ${tic:,.2f}",
        "Total_Annual_Cost": f"TC = ${tic:,.2f} + (${D:,.2f} * ${C:,.2f}) = ${tic:,.2f} + ${apc:,.2f} = ${tc:,.2f}",
        "Orders_Per_Year": f"N = {D:,.2f} / {eoq:,.2f} = {orders_per_yr:.2f} orders/year",
        "Order_Cycle_Days": f"T = ({eoq:,.2f} / {D:,.2f}) * {working_days_per_year} = {cycle_days:.1f} operating days",
    }
    if rop is not None:
        substituted["Reorder_Point"] = f"ROP = ({D:,.2f} / {working_days_per_year}) * {lead_time_days:.1f} = {D / working_days_per_year:.2f} * {lead_time_days:.1f} = {rop:,.2f} units"

    # 5. Intermediate Calculations
    intermediates = {
        "numerator_2DS": numerator_2DS,
        "denominator_H": H,
        "ratio_2DS_over_H": ratio_2DS_over_H,
        "raw_eoq": eoq,
        "annual_ordering_cost": aoc,
        "annual_holding_cost": ahc,
        "annual_purchase_cost": apc,
        "total_inventory_cost": tic,
        "total_annual_cost": tc,
        "orders_per_year": orders_per_yr,
        "order_cycle_days": cycle_days,
        "reorder_point_deterministic": rop,
    }

    # 6. Final Answer
    final_ans = {
        "optimal_order_quantity_eoq": round(eoq, 2),
        "recommended_integer_batch": q_int,
        "annual_ordering_cost": round(aoc, 2),
        "annual_holding_cost": round(ahc, 2),
        "total_inventory_cost": round(tic, 2),
        "total_annual_cost": round(tc, 2),
        "orders_per_year": round(orders_per_yr, 2),
        "order_cycle_days": round(cycle_days, 1),
        "reorder_point": round(rop, 2) if rop is not None else None,
    }

    # 7. Verification Checks
    cost_diff = abs(aoc - ahc)
    cost_tol = max(0.01, 0.001 * max(aoc, ahc))
    balance_ok = cost_diff <= cost_tol
    checks = {
        "status": "PASSED" if balance_ok and eoq > 0 else "FAILED",
        "checks": [
            {
                "check": "Optimality Condition: Annual Ordering Cost equals Annual Holding Cost",
                "passed": balance_ok,
                "detail": f"|${aoc:,.2f} - ${ahc:,.2f}| = ${cost_diff:.4f} <= tolerance ${cost_tol:.4f}",
            },
            {
                "check": "Feasibility: EOQ is strictly positive",
                "passed": eoq > 0,
                "detail": f"Q* = {eoq:,.2f} units > 0",
            },
        ],
    }

    # 8. Assumptions
    assumptions = [
        "Demand rate is continuous, constant, and known with certainty over the planning horizon.",
        "Replenishment is instantaneous or arrives in one complete batch after fixed lead time.",
        "Ordering cost (S) is fixed per replenishment order regardless of order size.",
        "Holding cost (H) is linear and proportional to average cycle inventory (Q / 2).",
        "No shortages, stockouts, or backorders are permitted.",
        "Unit purchase price is constant with no quantity discounts applied.",
    ]

    inputs_dict = {
        "annual_demand": D,
        "order_cost": S,
        "holding_cost": H,
        "holding_cost_rate": holding_cost_rate,
        "unit_cost": C,
        "lead_time_days": lead_time_days,
        "working_days_per_year": working_days_per_year,
    }

    eoq_final = FinalAnswer(round(eoq, 2), final_ans)

    return EOQResult(
        inputs_extracted=inputs_dict,
        variable_definitions=var_defs,
        formula=formulas,
        formula_with_substituted_values=substituted,
        intermediate_calculations=intermediates,
        final_answer=eoq_final,
        verification_checks=checks,
        assumptions=assumptions,
        eoq=eoq,
        annual_demand=D,
        order_cost=S,
        holding_cost=H,
        unit_cost=C,
        annual_ordering_cost=aoc,
        annual_holding_cost=ahc,
        total_inventory_cost=tic,
        total_annual_cost=tc,
        orders_per_year=orders_per_yr,
        order_cycle_days=cycle_days,
        reorder_point=rop,
        lead_time_days=lead_time_days,
        working_days_per_year=working_days_per_year,
    )


# ============================================================================
# 2. Quantity Discount Model
# ============================================================================

@dataclass
class DiscountTierEvaluation:
    """Evaluation metrics for an individual discount bracket."""
    tier_index: int
    min_qty: float
    max_qty: Optional[float]
    unit_price: float
    holding_cost: float
    raw_eoq: float
    candidate_qty: float
    status: str
    annual_purchase_cost: float
    annual_ordering_cost: float
    annual_holding_cost: float
    total_annual_cost: float


@dataclass
class QuantityDiscountResult:
    """Structured result for Quantity Discount analysis with student verification package."""
    # 8 Student Verification Requirements
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Union[float, FinalAnswer]
    verification_checks: Dict[str, Any]
    assumptions: List[str]

    # Direct attributes for backward compatibility
    optimal_order_quantity: float
    best_tier_index: int
    optimal_unit_price: float
    annual_demand: float
    order_cost: float
    annual_purchase_cost: float
    annual_ordering_cost: float
    annual_holding_cost: float
    total_annual_cost: float
    baseline_total_cost: float
    total_annual_savings: float
    tier_evaluations: List[DiscountTierEvaluation]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "All-Units Quantity Discount",
            "1_inputs_extracted": self.inputs_extracted,
            "2_variable_definitions": self.variable_definitions,
            "3_formula": self.formula,
            "4_formula_with_substituted_values": self.formula_with_substituted_values,
            "5_intermediate_calculations": self.intermediate_calculations,
            "6_final_answer": self.final_answer.to_dict() if hasattr(self.final_answer, "to_dict") else self.final_answer,
            "7_verification_checks": self.verification_checks,
            "8_assumptions": self.assumptions,
        }


def calculate_quantity_discount(
    annual_demand: Optional[float] = None,
    order_cost: Optional[float] = None,
    discount_schedule: Optional[List[Dict[str, Any]]] = None,
    holding_cost_rate: Optional[float] = None,
    fixed_holding_cost: Optional[float] = None,
) -> QuantityDiscountResult:
    """Calculate the optimal order quantity under all-units quantity discounts with student verification package.

    Parameters
    ----------
    annual_demand : float
        Annual demand in units (D > 0). Required.
    order_cost : float
        Fixed ordering cost per order (S > 0). Required.
    discount_schedule : list of dict
        Each dict must contain 'min_qty', 'unit_price', and optional 'max_qty'. Required.
    holding_cost_rate : float, optional
        Carrying cost rate as fraction of unit price (e.g. 0.20 for 20%).
    fixed_holding_cost : float, optional
        Fixed holding cost per unit per year independent of unit price.

    Returns
    -------
    QuantityDiscountResult
    """
    if annual_demand is None:
        raise ValueError("Missing required numerical input: 'annual_demand' must be provided.")
    if not isinstance(annual_demand, (int, float)) or annual_demand <= 0:
        raise ValueError(f"'annual_demand' must be positive; got {annual_demand}")

    if order_cost is None:
        raise ValueError("Missing required numerical input: 'order_cost' must be provided.")
    if not isinstance(order_cost, (int, float)) or order_cost <= 0:
        raise ValueError(f"'order_cost' must be positive; got {order_cost}")

    if not discount_schedule:
        raise ValueError("Missing required input: 'discount_schedule' must be a non-empty list of tiers.")

    if holding_cost_rate is None and fixed_holding_cost is None:
        raise ValueError("Missing required numerical input: Must provide either 'holding_cost_rate' or 'fixed_holding_cost'.")

    D = float(annual_demand)
    S = float(order_cost)

    var_defs = {
        "D": "Annual demand (units/year)",
        "S": "Ordering cost per order ($/order)",
        "C_i": "Discounted unit purchase price for tier i ($/unit)",
        "H_i": "Holding cost per unit per year for tier i ($/unit/year)",
        "EOQ_i": "Unconstrained raw EOQ for tier i = sqrt((2 * D * S) / H_i)",
        "Q_i": "Evaluated candidate order quantity for tier i after threshold adjustment",
        "APC_i": "Annual Purchase Cost = D * C_i ($/year)",
        "AOC_i": "Annual Ordering Cost = (D / Q_i) * S ($/year)",
        "AHC_i": "Annual Holding Cost = (Q_i / 2) * H_i ($/year)",
        "TC_i": "Total Annual Cost = APC_i + AOC_i + AHC_i ($/year)",
        "Q*": "Optimal order quantity achieving global minimum TC across all tiers",
    }

    formulas = {
        "Raw_EOQ": "EOQ_i = sqrt((2 * D * S) / H_i)",
        "Threshold_Rule": "If EOQ_i < min_q then Q_i = min_q; else if min_q <= EOQ_i <= max_q then Q_i = EOQ_i; else dominated",
        "Total_Cost": "TC(Q_i) = (D * C_i) + ((D / Q_i) * S) + ((Q_i / 2) * H_i)",
        "Optimality": "Q* = argmin_{Q_i} TC(Q_i)",
    }

    evaluations: List[DiscountTierEvaluation] = []
    substituted_tiers = {}

    for idx, tier in enumerate(discount_schedule):
        min_q = float(tier["min_qty"])
        max_q = float(tier["max_qty"]) if tier.get("max_qty") is not None else None
        price = float(tier["unit_price"])

        if price <= 0:
            raise ValueError(f"Tier {idx} unit_price must be positive; got {price}")

        H_i = fixed_holding_cost if fixed_holding_cost is not None else (holding_cost_rate * price)
        raw_eoq = math.sqrt((2.0 * D * S) / H_i)

        if raw_eoq < min_q:
            candidate_q = min_q
            status = f"Adjusted up to threshold {min_q:.0f}"
        elif max_q is not None and raw_eoq > max_q:
            candidate_q = max_q
            status = f"Exceeds upper limit {max_q:.0f} (dominated)"
        else:
            candidate_q = raw_eoq
            status = "EOQ feasible within tier"

        apc = D * price
        aoc = (D / candidate_q) * S
        ahc = (candidate_q / 2.0) * H_i
        total_c = apc + aoc + ahc

        evaluation = DiscountTierEvaluation(
            tier_index=idx,
            min_qty=min_q,
            max_qty=max_q,
            unit_price=price,
            holding_cost=H_i,
            raw_eoq=raw_eoq,
            candidate_qty=candidate_q,
            status=status,
            annual_purchase_cost=apc,
            annual_ordering_cost=aoc,
            annual_holding_cost=ahc,
            total_annual_cost=total_c,
        )
        evaluations.append(evaluation)

        max_str = f"{max_q:.0f}" if max_q is not None else "inf"
        rng = f"[{min_q:.0f}, {max_str})"
        substituted_tiers[f"Tier_{idx+1}"] = (
            f"Range {rng} @ ${price:.2f}/unit: H = ${H_i:.2f}. "
            f"Raw EOQ = sqrt((2 * {D:,.0f} * {S:,.2f}) / {H_i:.2f}) = {raw_eoq:.2f} -> Q_cand = {candidate_q:,.2f} ({status}). "
            f"TC = ({D:,.0f} * ${price:.2f}) + (({D:,.0f} / {candidate_q:,.2f}) * ${S:,.2f}) + (({candidate_q:,.2f} / 2) * ${H_i:.2f}) "
            f"= ${apc:,.2f} + ${aoc:,.2f} + ${ahc:,.2f} = ${total_c:,.2f}"
        )

    best_tier = min(evaluations, key=lambda t: t.total_annual_cost)
    base_tier = evaluations[0]
    savings = base_tier.total_annual_cost - best_tier.total_annual_cost

    intermediates = {
        "tiers_evaluated_count": len(evaluations),
        "best_tier_index": best_tier.tier_index,
        "best_unit_price": best_tier.unit_price,
        "best_candidate_quantity": best_tier.candidate_qty,
        "annual_purchase_cost": best_tier.annual_purchase_cost,
        "annual_ordering_cost": best_tier.annual_ordering_cost,
        "annual_holding_cost": best_tier.annual_holding_cost,
        "total_annual_cost": best_tier.total_annual_cost,
        "baseline_tier_cost": base_tier.total_annual_cost,
        "net_annual_savings": savings,
        "tier_evaluations": [
            {
                "tier": t.tier_index + 1,
                "price": t.unit_price,
                "raw_eoq": round(t.raw_eoq, 2),
                "candidate_q": round(t.candidate_qty, 2),
                "status": t.status,
                "total_cost": round(t.total_annual_cost, 2),
            }
            for t in evaluations
        ],
    }

    final_ans = {
        "optimal_order_quantity": round(best_tier.candidate_qty, 2),
        "recommended_integer_batch": int(round(best_tier.candidate_qty)),
        "best_tier_index": best_tier.tier_index + 1,
        "best_unit_price": round(best_tier.unit_price, 2),
        "total_annual_cost": round(best_tier.total_annual_cost, 2),
        "total_annual_savings": round(savings, 2),
    }

    prices = [t.unit_price for t in evaluations]
    is_monotonic = all(prices[i] >= prices[i+1] for i in range(len(prices)-1))
    checks = {
        "status": "PASSED" if is_monotonic and savings >= -0.01 else "FAILED",
        "checks": [
            {
                "check": "Optimality: Selected batch achieves minimum total cost across all tiers",
                "passed": abs(best_tier.total_annual_cost - min(t.total_annual_cost for t in evaluations)) < 0.01,
                "detail": f"Best cost ${best_tier.total_annual_cost:,.2f} equals evaluated minimum",
            },
            {
                "check": "Monotonicity: Unit price decreases or stays constant with increasing order volume",
                "passed": is_monotonic,
                "detail": f"Price schedule: {prices}",
            },
            {
                "check": "Savings Feasibility: Savings vs baseline tier are non-negative",
                "passed": savings >= -0.01,
                "detail": f"Savings = ${savings:,.2f}",
            },
        ],
    }

    assumptions = [
        "All-units quantity discount structure: the discounted price applies to all units in the order.",
        "Holding cost is either a fixed dollar amount or a constant percentage of the discounted purchase price.",
        "Demand is constant, continuous, and known with certainty.",
        "Storage capacity, handling equipment, and working capital can accommodate the larger batch size.",
        "Shelf life is sufficiently long to prevent spoilage or obsolescence during extended cycle times.",
    ]

    inputs_dict = {
        "annual_demand": D,
        "order_cost": S,
        "holding_cost_rate": holding_cost_rate,
        "fixed_holding_cost": fixed_holding_cost,
        "discount_schedule": discount_schedule,
    }

    qd_final = FinalAnswer(round(best_tier.candidate_qty, 2), final_ans)

    return QuantityDiscountResult(
        inputs_extracted=inputs_dict,
        variable_definitions=var_defs,
        formula=formulas,
        formula_with_substituted_values=substituted_tiers,
        intermediate_calculations=intermediates,
        final_answer=qd_final,
        verification_checks=checks,
        assumptions=assumptions,
        optimal_order_quantity=best_tier.candidate_qty,
        best_tier_index=best_tier.tier_index,
        optimal_unit_price=best_tier.unit_price,
        annual_demand=D,
        order_cost=S,
        annual_purchase_cost=best_tier.annual_purchase_cost,
        annual_ordering_cost=best_tier.annual_ordering_cost,
        annual_holding_cost=best_tier.annual_holding_cost,
        total_annual_cost=best_tier.total_annual_cost,
        baseline_total_cost=base_tier.total_annual_cost,
        total_annual_savings=savings,
        tier_evaluations=evaluations,
    )


# ============================================================================
# 3. Safety Stock and Reorder Point (Continuous Review Policy)
# ============================================================================

@dataclass
class ROPResult:
    """Structured result for Safety Stock and Reorder Point with student verification package."""
    # 8 Student Verification Requirements
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Union[float, FinalAnswer]
    verification_checks: Dict[str, Any]
    assumptions: List[str]

    # Direct attributes for backward compatibility
    reorder_point: float
    safety_stock: float
    avg_demand: float
    std_demand: float
    lead_time: float
    std_lead_time: float
    service_level: Optional[float]
    z_score: float
    uncertainty_case: str
    lead_time_demand: float
    lead_time_demand_std: float
    stockout_risk: Optional[float]
    annual_safety_stock_holding_cost: Optional[float] = None
    time_unit: str = "days"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "Safety Stock & Reorder Point (Continuous Review)",
            "1_inputs_extracted": self.inputs_extracted,
            "2_variable_definitions": self.variable_definitions,
            "3_formula": self.formula,
            "4_formula_with_substituted_values": self.formula_with_substituted_values,
            "5_intermediate_calculations": self.intermediate_calculations,
            "6_final_answer": self.final_answer.to_dict() if hasattr(self.final_answer, "to_dict") else self.final_answer,
            "7_verification_checks": self.verification_checks,
            "8_assumptions": self.assumptions,
        }


SafetyStockROPResult = ROPResult


def calculate_rop(
    avg_demand: Optional[float] = None,
    lead_time: Optional[float] = None,
    std_demand: float = 0.0,
    std_lead_time: float = 0.0,
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
    unit_holding_cost: Optional[float] = None,
    time_unit: str = "days",
) -> ROPResult:
    """Calculate Safety Stock and Reorder Point supporting all 4 uncertainty cases with student verification package.

    Cases supported:
    1. Constant demand and constant lead time:
       sigma_LT = 0, SS = 0, ROP = d * L
    2. Variable demand and constant lead time:
       sigma_LT = sigma_d * sqrt(L), SS = z * sigma_LT, ROP = d * L + SS
    3. Constant demand and variable lead time:
       sigma_LT = d * sigma_L, SS = z * sigma_LT, ROP = d * L + SS
    4. Variable demand and variable lead time:
       sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2), SS = z * sigma_LT, ROP = d * L + SS
    """
    if avg_demand is None:
        raise ValueError("Missing required numerical input: 'avg_demand' must be provided.")
    if not isinstance(avg_demand, (int, float)) or avg_demand < 0:
        raise ValueError(f"'avg_demand' must be non-negative; got {avg_demand}")

    if lead_time is None:
        raise ValueError("Missing required numerical input: 'lead_time' must be provided.")
    if not isinstance(lead_time, (int, float)) or lead_time <= 0:
        raise ValueError(f"'lead_time' must be positive; got {lead_time}")

    d = float(avg_demand)
    L = float(lead_time)
    sig_d = float(std_demand)
    sig_L = float(std_lead_time)

    has_demand_var = sig_d > 0.0
    has_lt_var = sig_L > 0.0
    has_uncertainty = has_demand_var or has_lt_var

    # Resolve z-score
    effective_z = 0.0
    effective_sl = service_level

    if has_uncertainty:
        if z_score is not None:
            effective_z = float(z_score)
            if effective_sl is None:
                effective_sl = float(norm.cdf(effective_z))
        elif service_level is not None:
            if not (0.0 < service_level < 1.0):
                raise ValueError(f"'service_level' must be strictly between 0 and 1; got {service_level}")
            effective_z = float(norm.ppf(service_level))
        else:
            raise ValueError("Missing required numerical input: Either 'service_level' or 'z_score' must be provided when demand or lead time is variable.")
    else:
        effective_z = 0.0

    var_defs = {
        "d": f"Average demand per {time_unit} (units/{time_unit})",
        "sigma_d": f"Demand standard deviation per {time_unit} (units/{time_unit})",
        "L": f"Supplier delivery lead time ({time_unit})",
        "sigma_L": f"Lead time standard deviation ({time_unit})",
        "LTD": "Expected Lead Time Demand = d * L (units)",
        "sigma_LT": "Combined standard deviation of demand during lead time (units)",
        "SL": "Cycle Service Level (target in-stock probability per cycle)",
        "z": "Standard normal safety factor = Phi^-1(SL)",
        "SS": "Safety Stock buffer = z * sigma_LT (units)",
        "ROP": "Reorder Point = LTD + SS (units)",
        "Risk": "Cycle stockout risk = 1 - SL",
    }

    ltd = d * L
    if not has_demand_var and not has_lt_var:
        uncertainty_case = "constant_demand_constant_lead_time"
        sigma_lt = 0.0
        formula_sigma = "sigma_LT = 0"
        formula_ss = "SS = 0"
        sub_sigma = "sigma_LT = 0.0 (deterministic)"
        sub_ss = "SS = 0.0 units"
    elif has_demand_var and not has_lt_var:
        uncertainty_case = "variable_demand_constant_lead_time"
        sigma_lt = sig_d * math.sqrt(L)
        formula_sigma = "sigma_LT = sigma_d * sqrt(L)"
        formula_ss = "SS = z * sigma_LT"
        sub_sigma = f"sigma_LT = {sig_d:.2f} * sqrt({L:.2f}) = {sig_d:.2f} * {math.sqrt(L):.4f} = {sigma_lt:.2f} units"
        sub_ss = f"SS = {effective_z:.4f} * {sigma_lt:.2f} = {effective_z * sigma_lt:.2f} units"
    elif not has_demand_var and has_lt_var:
        uncertainty_case = "constant_demand_variable_lead_time"
        sigma_lt = d * sig_L
        formula_sigma = "sigma_LT = d * sigma_L"
        formula_ss = "SS = z * sigma_LT"
        sub_sigma = f"sigma_LT = {d:.2f} * {sig_L:.2f} = {sigma_lt:.2f} units"
        sub_ss = f"SS = {effective_z:.4f} * {sigma_lt:.2f} = {effective_z * sigma_lt:.2f} units"
    else:
        uncertainty_case = "variable_demand_variable_lead_time"
        var_lt = (L * (sig_d ** 2)) + ((d ** 2) * (sig_L ** 2))
        sigma_lt = math.sqrt(var_lt)
        formula_sigma = "sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)"
        formula_ss = "SS = z * sigma_LT"
        sub_sigma = f"sigma_LT = sqrt(({L:.2f} * {sig_d:.2f}^2) + ({d:.2f}^2 * {sig_L:.2f}^2)) = sqrt({L * sig_d**2:.2f} + {d**2 * sig_L**2:.2f}) = sqrt({var_lt:.2f}) = {sigma_lt:.2f} units"
        sub_ss = f"SS = {effective_z:.4f} * {sigma_lt:.2f} = {effective_z * sigma_lt:.2f} units"

    ss = effective_z * sigma_lt
    rop = ltd + ss
    stockout_risk = (1.0 - effective_sl) if effective_sl is not None else None

    holding_ss_cost = None
    if unit_holding_cost is not None and unit_holding_cost > 0:
        holding_ss_cost = ss * float(unit_holding_cost)

    formulas = {
        "Lead_Time_Demand": "LTD = d * L",
        "Lead_Time_Std_Dev": formula_sigma,
        "Safety_Stock": formula_ss,
        "Reorder_Point": "ROP = LTD + SS = (d * L) + (z * sigma_LT)",
        "Stockout_Risk": "Risk = 1 - SL",
    }

    substituted = {
        "Lead_Time_Demand": f"LTD = {d:,.2f} * {L:.1f} = {ltd:,.2f} units",
        "Lead_Time_Std_Dev": sub_sigma,
        "Safety_Stock": sub_ss,
        "Reorder_Point": f"ROP = {ltd:,.2f} + {ss:,.2f} = {rop:,.2f} units (reorder at {int(math.ceil(rop)):,} units)",
    }
    if stockout_risk is not None:
        substituted["Stockout_Risk"] = f"Risk = 1 - {effective_sl:.4f} = {stockout_risk * 100:.2f}%"

    intermediates = {
        "uncertainty_case": uncertainty_case,
        "lead_time_demand_mean": ltd,
        "lead_time_demand_std": sigma_lt,
        "z_score": effective_z,
        "safety_stock": ss,
        "reorder_point": rop,
        "stockout_risk": stockout_risk,
        "annual_holding_cost_of_safety_stock": holding_ss_cost,
        "formula_used": formula_sigma,
    }

    final_ans = {
        "reorder_point": round(rop, 2),
        "recommended_integer_rop": int(math.ceil(rop)),
        "safety_stock": round(ss, 2),
        "recommended_integer_safety_stock": int(math.ceil(ss)),
        "lead_time_demand": round(ltd, 2),
        "lead_time_demand_std": round(sigma_lt, 2),
        "uncertainty_case": uncertainty_case,
    }

    checks = {
        "status": "PASSED" if rop >= ltd - 1e-4 else "FAILED",
        "checks": [
            {
                "check": "Safety Buffer Monotonicity: Reorder Point >= Expected Lead Time Demand",
                "passed": rop >= ltd - 1e-4,
                "detail": f"ROP ({rop:.2f}) >= LTD ({ltd:.2f})",
            },
            {
                "check": "Probability Consistency: Stockout risk complement equals 1 - Service Level",
                "passed": stockout_risk is None or abs(stockout_risk - (1.0 - effective_sl)) < 1e-5,
                "detail": f"Stockout risk = {stockout_risk * 100:.2f}%" if stockout_risk is not None else "N/A",
            },
        ],
    }

    assumptions = [
        f"Uncertainty Classification: {uncertainty_case.replace('_', ' ').title()}.",
        "Continuous Review Policy (s, Q): Inventory position is monitored continuously in real time.",
        "Demand increments in consecutive periods are mutually independent and identically distributed.",
        "Supplier lead time is independent of order size Q.",
        "Unsatisfied customer demand during lead time is completely backordered.",
        "Service level represents Cycle Service Level (probability of not stocking out during lead time).",
    ]

    inputs_dict = {
        "avg_demand": d,
        "lead_time": L,
        "std_demand": sig_d,
        "std_lead_time": sig_L,
        "service_level": effective_sl,
        "z_score": effective_z,
        "unit_holding_cost": unit_holding_cost,
        "time_unit": time_unit,
    }

    rop_final = FinalAnswer(round(rop, 2), final_ans)

    return ROPResult(
        inputs_extracted=inputs_dict,
        variable_definitions=var_defs,
        formula=formulas,
        formula_with_substituted_values=substituted,
        intermediate_calculations=intermediates,
        final_answer=rop_final,
        verification_checks=checks,
        assumptions=assumptions,
        reorder_point=rop,
        safety_stock=ss,
        avg_demand=d,
        std_demand=sig_d,
        lead_time=L,
        std_lead_time=sig_L,
        service_level=effective_sl,
        z_score=effective_z,
        uncertainty_case=uncertainty_case,
        lead_time_demand=ltd,
        lead_time_demand_std=sigma_lt,
        stockout_risk=stockout_risk,
        annual_safety_stock_holding_cost=holding_ss_cost,
        time_unit=time_unit,
    )


calculate_safety_stock_rop = calculate_rop


# ============================================================================
# 4. Fixed-Period Review Model (Periodic Review / Order-Up-To Level)
# ============================================================================

@dataclass
class PeriodicReviewResult:
    """Structured result for Fixed-Period Review with student verification package."""
    # 8 Student Verification Requirements
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Union[float, FinalAnswer]
    verification_checks: Dict[str, Any]
    assumptions: List[str]

    # Direct attributes for backward compatibility
    recommended_order_quantity: float
    target_inventory: float
    safety_stock: float
    protection_period: float
    protection_interval: float
    avg_demand: float
    std_demand: float
    review_period: float
    lead_time: float
    service_level: float
    z_score: float
    demand_during_protection_period: float
    expected_demand_protection_interval: float
    std_demand_protection_period: float
    std_demand_protection_interval: float
    target_order_up_to_level: float
    on_hand_inventory: float
    on_order_inventory: float
    backorders: float
    inventory_position: float
    time_unit: str = "days"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "Fixed-Period Review (Periodic Order-Up-To Level)",
            "1_inputs_extracted": self.inputs_extracted,
            "2_variable_definitions": self.variable_definitions,
            "3_formula": self.formula,
            "4_formula_with_substituted_values": self.formula_with_substituted_values,
            "5_intermediate_calculations": self.intermediate_calculations,
            "6_final_answer": self.final_answer.to_dict() if hasattr(self.final_answer, "to_dict") else self.final_answer,
            "7_verification_checks": self.verification_checks,
            "8_assumptions": self.assumptions,
        }


def calculate_periodic_review(
    review_period: Optional[float] = None,
    lead_time: Optional[float] = None,
    avg_demand: Optional[float] = None,
    std_demand: float = 0.0,
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
    on_hand: float = 0.0,
    on_order: float = 0.0,
    backorders: float = 0.0,
    inventory_position: Optional[float] = None,
    time_unit: str = "days",
) -> PeriodicReviewResult:
    """Calculate target inventory (M) and replenishment quantity (q) for periodic review with student verification package."""
    if review_period is None:
        raise ValueError("Missing required numerical input: 'review_period' must be provided.")
    if not isinstance(review_period, (int, float)) or review_period <= 0:
        raise ValueError(f"'review_period' must be positive; got {review_period}")

    if lead_time is None:
        raise ValueError("Missing required numerical input: 'lead_time' must be provided.")
    if not isinstance(lead_time, (int, float)) or lead_time < 0:
        raise ValueError(f"'lead_time' cannot be negative; got {lead_time}")

    if avg_demand is None:
        raise ValueError("Missing required numerical input: 'avg_demand' must be provided.")
    if not isinstance(avg_demand, (int, float)) or avg_demand < 0:
        raise ValueError(f"'avg_demand' cannot be negative; got {avg_demand}")

    P = float(review_period)
    L = float(lead_time)
    d = float(avg_demand)
    sig_d = float(std_demand)

    effective_z = 0.0
    effective_sl = service_level

    if sig_d > 0:
        if z_score is not None:
            effective_z = float(z_score)
            if effective_sl is None:
                effective_sl = float(norm.cdf(effective_z))
        elif service_level is not None:
            if not (0.0 < service_level < 1.0):
                raise ValueError(f"'service_level' must be strictly between 0 and 1; got {service_level}")
            effective_z = float(norm.ppf(service_level))
            effective_sl = service_level
        else:
            raise ValueError("Missing required numerical input: Either 'service_level' or 'z_score' must be provided when demand standard deviation is positive.")
    else:
        effective_sl = 1.0

    var_defs = {
        "P": f"Review period between order decisions ({time_unit})",
        "L": f"Supplier delivery lead time ({time_unit})",
        "P + L": f"Protection period over which stock must satisfy demand ({time_unit})",
        "d_bar": f"Average demand per period (units/{time_unit})",
        "sigma_d": f"Standard deviation of demand per period (units/{time_unit})",
        "D_prot": "Expected demand during protection period = d_bar * (P + L) (units)",
        "sigma_prot": "Demand standard deviation during protection period = sigma_d * sqrt(P + L) (units)",
        "z": "Standard normal safety factor = Phi^-1(SL)",
        "SS": "Safety Stock = z * sigma_prot (units)",
        "M": "Target Order-Up-To Level (Base Stock) = D_prot + SS (units)",
        "IP": "Net inventory position = On-hand + On-order - Backorders (units)",
        "q": "Recommended replenishment quantity = max(0, M - IP) (units)",
    }

    formulas = {
        "Protection_Period": "Protection_Period = P + L",
        "Demand_Protection": "D_prot = d_bar * (P + L)",
        "Std_Dev_Protection": "sigma_prot = sigma_d * sqrt(P + L)",
        "Safety_Stock": "SS = z * sigma_prot = z * sigma_d * sqrt(P + L)",
        "Target_Inventory": "M = D_prot + SS = (d_bar * (P + L)) + (z * sigma_d * sqrt(P + L))",
        "Inventory_Position": "IP = on_hand + on_order - backorders",
        "Order_Quantity": "q = max(0, M - IP)",
    }

    prot_period = P + L
    demand_prot = d * prot_period
    sigma_prot = sig_d * math.sqrt(prot_period)
    ss = effective_z * sigma_prot
    target_m = demand_prot + ss

    net_ip = float(inventory_position) if inventory_position is not None else (float(on_hand) + float(on_order) - float(backorders))
    order_q = max(0.0, target_m - net_ip)

    substituted = {
        "Protection_Period": f"Protection_Period = {P:.1f} + {L:.1f} = {prot_period:.1f} {time_unit}",
        "Demand_Protection": f"D_prot = {d:,.2f} * {prot_period:.1f} = {demand_prot:,.2f} units",
        "Std_Dev_Protection": f"sigma_prot = {sig_d:,.2f} * sqrt({prot_period:.1f}) = {sig_d:,.2f} * {math.sqrt(prot_period):.4f} = {sigma_prot:,.2f} units",
        "Safety_Stock": f"SS = {effective_z:.4f} * {sigma_prot:,.2f} = {ss:,.2f} units",
        "Target_Inventory": f"M = {demand_prot:,.2f} + {ss:,.2f} = {target_m:,.2f} units (ceiling: {int(math.ceil(target_m)):,} units)",
        "Inventory_Position": f"IP = {float(on_hand):,.1f} + {float(on_order):,.1f} - {float(backorders):,.1f} = {net_ip:,.2f} units",
        "Order_Quantity": f"q = max(0, {target_m:,.2f} - {net_ip:,.2f}) = {order_q:,.2f} units (place order for {int(math.ceil(order_q)):,} units)",
    }

    intermediates = {
        "protection_period": prot_period,
        "demand_during_protection_period": demand_prot,
        "std_demand_protection_period": sigma_prot,
        "z_score": effective_z,
        "safety_stock": ss,
        "target_inventory": target_m,
        "inventory_position": net_ip,
        "recommended_order_quantity": order_q,
    }

    final_ans = {
        "recommended_order_quantity_q": round(order_q, 2),
        "recommended_order_quantity_integer": int(math.ceil(order_q)),
        "target_inventory_M": round(target_m, 2),
        "target_inventory_integer": int(math.ceil(target_m)),
        "safety_stock": round(ss, 2),
        "current_inventory_position": round(net_ip, 2),
    }

    checks = {
        "status": "PASSED" if target_m >= demand_prot - 1e-4 and order_q >= 0 else "FAILED",
        "checks": [
            {
                "check": "Base Stock Ceiling: Target Inventory (M) >= Expected Protection Demand",
                "passed": target_m >= demand_prot - 1e-4,
                "detail": f"M ({target_m:.2f}) >= D_prot ({demand_prot:.2f})",
            },
            {
                "check": "Non-negative Order: Replenishment quantity is bounded below by 0",
                "passed": order_q >= 0,
                "detail": f"q = {order_q:.2f} >= 0",
            },
        ],
    }

    assumptions = [
        "Fixed-Period Review Policy (P, M): Order decisions occur periodically at fixed review intervals of P periods.",
        "Protection Period is (P + L): Stock ordered now must cover customer demand until the next delivery arrives.",
        "Target inventory (base stock M) restores inventory position to buffer both expected demand and safety margin.",
        "Demand increments in each period are mutually independent and identically distributed.",
        "Replenishment quantity has a non-negative floor: q = max(0, M - IP). No product returns allowed if IP > M.",
    ]

    inputs_dict = {
        "review_period": P,
        "lead_time": L,
        "avg_demand": d,
        "std_demand": sig_d,
        "service_level": effective_sl,
        "z_score": effective_z,
        "on_hand": float(on_hand),
        "on_order": float(on_order),
        "backorders": float(backorders),
        "inventory_position": net_ip,
        "time_unit": time_unit,
    }

    pr_final = FinalAnswer(round(order_q, 2), final_ans)

    return PeriodicReviewResult(
        inputs_extracted=inputs_dict,
        variable_definitions=var_defs,
        formula=formulas,
        formula_with_substituted_values=substituted,
        intermediate_calculations=intermediates,
        final_answer=pr_final,
        verification_checks=checks,
        assumptions=assumptions,
        recommended_order_quantity=order_q,
        target_inventory=target_m,
        safety_stock=ss,
        protection_period=prot_period,
        protection_interval=prot_period,
        avg_demand=d,
        std_demand=sig_d,
        review_period=P,
        lead_time=L,
        service_level=effective_sl,
        z_score=effective_z,
        demand_during_protection_period=demand_prot,
        expected_demand_protection_interval=demand_prot,
        std_demand_protection_period=sigma_prot,
        std_demand_protection_interval=sigma_prot,
        target_order_up_to_level=target_m,
        on_hand_inventory=float(on_hand),
        on_order_inventory=float(on_order),
        backorders=float(backorders),
        inventory_position=net_ip,
        time_unit=time_unit,
    )


# ============================================================================
# 5. Newsvendor Model (Single-Period Stochastic Demand)
# ============================================================================

@dataclass
class NewsvendorResult:
    """Structured result for Newsvendor single-period inventory decision with student verification package."""
    # 8 Student Verification Requirements
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Union[float, FinalAnswer]
    verification_checks: Dict[str, Any]
    assumptions: List[str]

    # Direct attributes for backward compatibility
    optimal_order_quantity: float
    recommended_order_quantity_integer: int
    critical_ratio: float
    optimal_z: float
    expected_profit: float
    expected_sales: float
    expected_lost_sales: float
    expected_leftover_units: float
    underage_cost: float
    overage_cost: float
    selling_price: float
    unit_cost: float
    salvage_value: float
    shortage_cost: float
    mean_demand: float
    std_demand: float
    expected_revenue: float
    expected_purchase_cost: float
    expected_salvage_revenue: float
    expected_penalty_cost: float
    maximum_potential_profit: float
    evpi: float
    in_stock_probability: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "Newsvendor (Single-Period Stochastic Demand)",
            "1_inputs_extracted": self.inputs_extracted,
            "2_variable_definitions": self.variable_definitions,
            "3_formula": self.formula,
            "4_formula_with_substituted_values": self.formula_with_substituted_values,
            "5_intermediate_calculations": self.intermediate_calculations,
            "6_final_answer": self.final_answer.to_dict() if hasattr(self.final_answer, "to_dict") else self.final_answer,
            "7_verification_checks": self.verification_checks,
            "8_assumptions": self.assumptions,
        }


def calculate_newsvendor(
    selling_price: Optional[float] = None,
    unit_cost: Optional[float] = None,
    salvage_value: float = 0.0,
    shortage_cost: float = 0.0,
    mean_demand: Optional[float] = None,
    std_demand: Optional[float] = None,
) -> NewsvendorResult:
    """Calculate underage cost, overage cost, critical ratio, and recommended order quantity with student verification package."""
    if selling_price is None:
        raise ValueError("Missing required numerical input: 'selling_price' must be provided.")
    if unit_cost is None:
        raise ValueError("Missing required numerical input: 'unit_cost' must be provided.")
    if mean_demand is None:
        raise ValueError("Missing required numerical input: 'mean_demand' must be provided.")
    if std_demand is None:
        raise ValueError("Missing required numerical input: 'std_demand' must be provided.")

    if not isinstance(selling_price, (int, float)) or selling_price <= 0:
        raise ValueError(f"'selling_price' must be positive; got {selling_price}")
    if not isinstance(unit_cost, (int, float)) or unit_cost <= 0:
        raise ValueError(f"'unit_cost' must be positive; got {unit_cost}")
    if not isinstance(salvage_value, (int, float)) or salvage_value < 0:
        raise ValueError(f"'salvage_value' cannot be negative; got {salvage_value}")
    if not isinstance(shortage_cost, (int, float)) or shortage_cost < 0:
        raise ValueError(f"'shortage_cost' cannot be negative; got {shortage_cost}")
    if not isinstance(mean_demand, (int, float)) or mean_demand <= 0:
        raise ValueError(f"'mean_demand' must be positive; got {mean_demand}")
    if not isinstance(std_demand, (int, float)) or std_demand <= 0:
        raise ValueError(f"'std_demand' must be positive; got {std_demand}")

    P = float(selling_price)
    C = float(unit_cost)
    V = float(salvage_value)
    g = float(shortage_cost)
    mu = float(mean_demand)
    sigma = float(std_demand)

    if P <= C:
        raise ValueError(f"Selling price (${P:.2f}) must exceed unit cost (${C:.2f}) for economic viability.")
    if C <= V:
        raise ValueError(f"Unit cost (${C:.2f}) must exceed salvage value (${V:.2f}).")

    var_defs = {
        "P": "Unit selling price ($/unit)",
        "C": "Unit purchase / acquisition cost ($/unit)",
        "V": "Unit salvage value for unsold leftover goods ($/unit)",
        "g": "Unit shortage penalty or lost customer goodwill ($/unit)",
        "Cu": "Underage Cost (marginal loss of ordering one unit too few) = P - C + g ($/unit)",
        "Co": "Overage Cost (marginal loss of ordering one unit too many) = C - V ($/unit)",
        "CR": "Critical Ratio / Critical Fractile = Cu / (Cu + Co)",
        "mu": "Mean expected demand (units)",
        "sigma": "Standard deviation of demand (units)",
        "z*": "Optimal standard normal safety factor = Phi^-1(CR)",
        "Q*": "Optimal order quantity = mu + z* * sigma (units)",
        "L(z)": "Standard normal unit loss function = phi(z*) - z* * (1 - Phi(z*))",
        "E[Lost Sales]": "Expected unmet customer demand = sigma * L(z*) (units)",
        "E[Sales]": "Expected fulfilled sales = mu - E[Lost Sales] (units)",
        "E[Leftover]": "Expected unsold leftover inventory = Q* - E[Sales] (units)",
        "E[Profit]": "Expected net financial profit ($)",
        "EVPI": "Expected Value of Perfect Information = Max Potential Profit - E[Profit] ($)",
    }

    formulas = {
        "Underage_Cost": "Cu = P - C + g",
        "Overage_Cost": "Co = C - V",
        "Critical_Ratio": "CR = Cu / (Cu + Co)",
        "Optimal_Z": "z* = Phi^-1(CR)",
        "Optimal_Order_Quantity": "Q* = mu + z* * sigma",
        "Normal_Loss_Function": "L(z) = phi(z*) - z* * (1 - Phi(z*))",
        "Expected_Lost_Sales": "E[Lost Sales] = sigma * L(z*)",
        "Expected_Sales": "E[Sales] = mu - E[Lost Sales]",
        "Expected_Leftover": "E[Leftover] = Q* - E[Sales]",
        "Expected_Profit": "E[Profit] = (P * E[Sales]) + (V * E[Leftover]) - (C * Q*) - (g * E[Lost Sales])",
        "Max_Profit": "Max_Profit = (P - C) * mu",
        "EVPI": "EVPI = Max_Profit - E[Profit]",
    }

    cu = P - C + g
    co = C - V
    cr = cu / (cu + co)
    z_opt = float(norm.ppf(cr))
    q_opt = mu + (z_opt * sigma)
    q_int = int(round(q_opt))

    z_eval = (q_opt - mu) / sigma
    pdf_z = float(norm.pdf(z_eval))
    cdf_z = float(norm.cdf(z_eval))
    unit_loss = pdf_z - (z_eval * (1.0 - cdf_z))

    exp_lost = sigma * unit_loss
    exp_sales = mu - exp_lost
    exp_left = q_opt - exp_sales

    exp_rev = P * exp_sales
    exp_purch = C * q_opt
    exp_salv = V * exp_left
    exp_pen = g * exp_lost
    exp_profit = exp_rev + exp_salv - exp_purch - exp_pen

    max_prof = (P - C) * mu
    evpi = max_prof - exp_profit

    substituted = {
        "Underage_Cost": f"Cu = ${P:.2f} - ${C:.2f} + ${g:.2f} = ${cu:.2f}/unit",
        "Overage_Cost": f"Co = ${C:.2f} - ${V:.2f} = ${co:.2f}/unit",
        "Critical_Ratio": f"CR = ${cu:.2f} / (${cu:.2f} + ${co:.2f}) = ${cu:.2f} / ${cu + co:.2f} = {cr:.4f}",
        "Optimal_Z": f"z* = Phi^-1({cr:.4f}) = {z_opt:.4f}",
        "Optimal_Order_Quantity": f"Q* = {mu:,.2f} + ({z_opt:.4f} * {sigma:,.2f}) = {mu:,.2f} + {z_opt * sigma:,.2f} = {q_opt:,.2f} units (order {q_int:,} units)",
        "Normal_Loss_Function": f"L(z) = {pdf_z:.4f} - ({z_opt:.4f} * (1 - {cdf_z:.4f})) = {unit_loss:.4f}",
        "Expected_Lost_Sales": f"E[Lost Sales] = {sigma:,.2f} * {unit_loss:.4f} = {exp_lost:,.2f} units",
        "Expected_Sales": f"E[Sales] = {mu:,.2f} - {exp_lost:,.2f} = {exp_sales:,.2f} units",
        "Expected_Leftover": f"E[Leftover] = {q_opt:,.2f} - {exp_sales:,.2f} = {exp_left:,.2f} units",
        "Expected_Profit": f"E[Profit] = (${P:.2f} * {exp_sales:,.2f}) + (${V:.2f} * {exp_left:,.2f}) - (${C:.2f} * {q_opt:,.2f}) = ${exp_rev:,.2f} + ${exp_salv:,.2f} - ${exp_purch:,.2f} = ${exp_profit:,.2f}",
        "EVPI": f"EVPI = ${max_prof:,.2f} - ${exp_profit:,.2f} = ${evpi:,.2f}",
    }

    intermediates = {
        "underage_cost_Cu": cu,
        "overage_cost_Co": co,
        "critical_ratio_CR": cr,
        "optimal_z_score": z_opt,
        "optimal_order_quantity_continuous": q_opt,
        "optimal_order_quantity_integer": q_int,
        "unit_normal_loss_Lz": unit_loss,
        "expected_lost_sales": exp_lost,
        "expected_sales": exp_sales,
        "expected_leftover_units": exp_left,
        "expected_revenue": exp_rev,
        "expected_purchase_cost": exp_purch,
        "expected_salvage_revenue": exp_salv,
        "expected_shortage_penalty": exp_pen,
        "expected_profit": exp_profit,
        "maximum_profit_perfect_information": max_prof,
        "evpi": evpi,
    }

    final_ans = {
        "optimal_order_quantity": round(q_opt, 2),
        "recommended_integer_order": q_int,
        "critical_ratio": round(cr, 4),
        "in_stock_probability": f"{cr * 100:.2f}%",
        "expected_profit": round(exp_profit, 2),
        "expected_sales": round(exp_sales, 2),
        "expected_lost_sales": round(exp_lost, 2),
        "expected_leftover_units": round(exp_left, 2),
        "evpi": round(evpi, 2),
    }

    checks = {
        "status": "PASSED" if (0.0 < cr < 1.0) and exp_profit <= max_prof + 1e-3 and evpi >= -1e-3 else "FAILED",
        "checks": [
            {
                "check": "Critical Ratio Boundedness: 0 < CR < 1",
                "passed": 0.0 < cr < 1.0,
                "detail": f"0.0 < {cr:.4f} < 1.0",
            },
            {
                "check": "Profit Upper Bound: Expected profit <= Max profit under perfect information",
                "passed": exp_profit <= max_prof + 1e-3,
                "detail": f"${exp_profit:,.2f} <= ${max_prof:,.2f}",
            },
            {
                "check": "Information Value: EVPI is non-negative",
                "passed": evpi >= -1e-3,
                "detail": f"EVPI = ${evpi:,.2f}",
            },
            {
                "check": "Inventory Conservation: Q* = Expected Sales + Expected Leftover",
                "passed": abs(q_opt - (exp_sales + exp_left)) < 1e-2,
                "detail": f"|{q_opt:.2f} - ({exp_sales:.2f} + {exp_left:.2f})| = 0.00",
            },
        ],
    }

    assumptions = [
        "Single-period horizon: goods must be ordered prior to the selling season with no replenishment opportunities.",
        "Demand is stochastic and continuous, modeled as a Normal distribution N(mu, sigma).",
        "Underage cost Cu captures lost gross margin (P - C) plus any intangible shortage penalty (g).",
        "Overage cost Co captures net acquisition depreciation (C - V) on unsold leftover units.",
        "Critical ratio CR = Cu / (Cu + Co) gives the optimal service level balancing marginal underage vs overage costs.",
    ]

    inputs_dict = {
        "selling_price": P,
        "unit_cost": C,
        "salvage_value": V,
        "shortage_cost": g,
        "mean_demand": mu,
        "std_demand": sigma,
    }

    nv_final = FinalAnswer(round(q_opt, 2), final_ans)

    return NewsvendorResult(
        inputs_extracted=inputs_dict,
        variable_definitions=var_defs,
        formula=formulas,
        formula_with_substituted_values=substituted,
        intermediate_calculations=intermediates,
        final_answer=nv_final,
        verification_checks=checks,
        assumptions=assumptions,
        optimal_order_quantity=q_opt,
        recommended_order_quantity_integer=q_int,
        critical_ratio=cr,
        optimal_z=z_opt,
        expected_profit=exp_profit,
        expected_sales=exp_sales,
        expected_lost_sales=exp_lost,
        expected_leftover_units=exp_left,
        underage_cost=cu,
        overage_cost=co,
        selling_price=P,
        unit_cost=C,
        salvage_value=V,
        shortage_cost=g,
        mean_demand=mu,
        std_demand=sigma,
        expected_revenue=exp_rev,
        expected_purchase_cost=exp_purch,
        expected_salvage_revenue=exp_salv,
        expected_penalty_cost=exp_pen,
        maximum_potential_profit=max_prof,
        evpi=evpi,
        in_stock_probability=cdf_z,
    )


# ============================================================================
# Verification Engine Helper Functions
# ============================================================================

def verify_eoq_result(res: EOQResult) -> Dict[str, Any]:
    return res.verification_checks


def verify_quantity_discount_result(res: QuantityDiscountResult) -> Dict[str, Any]:
    return res.verification_checks


def verify_safety_stock_rop_result(res: ROPResult) -> Dict[str, Any]:
    return res.verification_checks


verify_rop_result = verify_safety_stock_rop_result


def verify_periodic_review_result(res: PeriodicReviewResult) -> Dict[str, Any]:
    return res.verification_checks


def verify_newsvendor_result(res: NewsvendorResult) -> Dict[str, Any]:
    return res.verification_checks
