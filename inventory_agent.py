"""Supply Chain Analytics Inventory Decision Agent.

This module implements the end-to-end Decision Agent architecture:
Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain

All numerical calculations are delegated strictly to Python functions in `inventory_tools.py`.
The agent features advanced natural-language extraction recognizing common inventory terminology,
synonyms, and contextual phrasing, ensuring no parameter is reported missing when present in
the customer's question, while never inventing missing numerical inputs.
"""

from __future__ import annotations
import re
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import inventory_tools as tools


# ============================================================================
# Response Data Structures
# ============================================================================

@dataclass
class AgentResponse:
    """Structured response representing the full agent decision pipeline."""
    query: str
    model_id: str
    model_name: str
    confidence: float
    identification_rationale: str
    extracted_inputs: Dict[str, Any]
    missing_parameters: List[str]
    tool_name: str
    raw_result: Any
    result_dict: Dict[str, Any]
    verification: Dict[str, Any]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "pipeline": {
                "step_1_identify_model": {
                    "model_id": self.model_id,
                    "model_name": self.model_name,
                    "confidence": f"{self.confidence * 100:.1f}%",
                    "rationale": self.identification_rationale,
                },
                "step_2_extract_inputs": {
                    "extracted": self.extracted_inputs,
                    "missing_parameters": self.missing_parameters,
                },
                "step_3_select_python_tool": self.tool_name,
                "step_4_calculate": self.result_dict,
                "step_5_verify": self.verification,
                "step_6_explain": self.explanation,
            }
        }


# ============================================================================
# Helper Extraction Functions for Common Inventory Terminology & Synonyms
# ============================================================================

def _clean_num(raw: str) -> Optional[float]:
    """Strip currency symbols, commas, and formatting from a numerical string."""
    try:
        cleaned = raw.replace(",", "").replace("$", "").replace("€", "").replace("£", "").strip()
        return float(cleaned)
    except (ValueError, AttributeError):
        return None


def extract_ordering_cost(text: str) -> Optional[float]:
    """Extract ordering cost supporting all common synonyms:
    - ordering cost
    - order cost
    - setup cost
    - setup expense
    - cost per order
    - cost of placing an order
    - ordering/setup cost
    - ordering and setup cost
    - cost to place an order
    - order placement cost
    - order fee
    - procurement cost per order
    """
    # 1. Term followed by value
    pattern1 = (
        r"(?:ordering\s+cost|order\s+cost|setup\s+cost|setup\s+expense|cost\s+per\s+order|"
        r"cost\s+of\s+placing\s+an\s+order|ordering\/setup\s+cost|ordering\s+and\s+setup\s+cost|"
        r"cost\s+to\s+place\s+an\s+order|order\s+placement\s+cost|order\s+fee|procurement\s+cost\s+per\s+order|"
        r"replenishment\s+cost|order\s+processing\s+cost)"
        r"\s*(?:of|is|:|=)?\s*\$?([0-9,]+(?:\.[0-9]+)?)"
    )
    m = re.search(pattern1, text, re.IGNORECASE)
    if m:
        return _clean_num(m.group(1))

    # 2. Value followed by contextual descriptor
    pattern2 = r"\$([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s+order|to\s+place\s+(?:an\s+)?order|per\s+setup|to\s+order|per\s+replenishment)"
    m2 = re.search(pattern2, text, re.IGNORECASE)
    if m2:
        return _clean_num(m2.group(1))

    return None


def extract_holding_cost(text: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract holding cost distinguishing fixed dollar cost vs. percentage carrying rate.

    Returns (fixed_holding_cost, holding_cost_rate)
    """
    fixed_h = None
    rate_h = None

    # 1. Percentage holding / carrying cost
    m_rate = re.search(
        r"(?:percentage\s+holding\s+cost|holding\s+cost\s+rate|carrying\s+cost\s+rate|"
        r"carrying\s+cost\s+percentage|annual\s+carrying\s+rate|holding\s+rate|carrying\s+rate|"
        r"carrying\s+charge|carrying\s+cost|holding\s+cost)"
        r"\s*(?:of|is|:|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        text,
        re.IGNORECASE,
    )
    if not m_rate:
        m_rate = re.search(
            r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:of\s+(?:unit\s+)?(?:cost|price)\s+)?(?:holding|carrying)\s*cost",
            text,
            re.IGNORECASE,
        )

    if m_rate:
        rate_h = _clean_num(m_rate.group(1))
        if rate_h is not None:
            rate_h = rate_h / 100.0

    # Accept explicitly named fractional rates, e.g. "holding cost rate 0.20".
    # A value above 1 without a percent sign is ambiguous; do not guess.
    if rate_h is None:
        m_fraction = re.search(
            r"(?:holding|carrying)\s+(?:cost\s+)?rate\s*(?:of|is|:|=)?\s*(0?\.\d+|1(?:\.0+)?)\b(?!\s*%)",
            text, re.IGNORECASE,
        )
        if m_fraction:
            rate_h = float(m_fraction.group(1))

    # 2. Fixed holding cost ($ per unit per year)
    m_fix = re.search(
        r"(?:fixed\s+holding\s+cost|holding\s+cost|carrying\s+cost|holding\s+expense|carrying\s+expense|"
        r"cost\s+to\s+hold|cost\s+of\s+holding|cost\s+to\s+carry|storage\s+cost|inventory\s+carrying\s+cost)"
        r"\s*(?:of|is|:|=)?\s*\$?([0-9,]+(?:\.[0-9]+)?)(?!\s*%)",
        text,
        re.IGNORECASE,
    )
    if m_fix:
        val = _clean_num(m_fix.group(1))
        # Ensure it's not a percentage rate
        if rate_h is None or "$" in m_fix.group(0) or "per unit" in text[m_fix.end():m_fix.end()+30].lower():
            fixed_h = val

    if fixed_h is None and rate_h is None:
        m_unit_yr = re.search(
            r"\$([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s+unit\s+per\s+year|\/unit\/year|\/unit\s+per\s+year|per\s+unit\s+annually)",
            text,
            re.IGNORECASE,
        )
        if m_unit_yr:
            fixed_h = _clean_num(m_unit_yr.group(1))

    return fixed_h, rate_h


def extract_unit_price_or_cost(text: str) -> Optional[float]:
    """Extract unit purchase price or unit acquisition cost."""
    # Contextual check for purchase cost / unit cost
    m = re.search(
        r"(?:unit\s+purchase\s+price|unit\s+cost|purchase\s+cost|purchase\s+price|item\s+cost|"
        r"unit\s+price|cost\s+per\s+unit|price\s+per\s+unit|acquisition\s+cost|manufacturing\s+cost|"
        r"procurement\s+price)\s*(?:of|is|:|=)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
        text,
        re.IGNORECASE,
    )
    if m:
        return _clean_num(m.group(1))

    m2 = re.search(r"costs?\s+(?:us\s+)?\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:to\s+(?:acquire|produce|make|buy|purchase))", text, re.IGNORECASE)
    if m2:
        return _clean_num(m2.group(1))

    return None


def extract_demands(text: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Read demand mean and variability separately; never confuse lead-time variability."""
    num = r"([0-9][0-9,]*(?:\.[0-9]+)?)"
    def first(patterns):
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return _clean_num(match.group(1))
        return None

    annual = first([
        rf"\b(?:annual|yearly)\s+(?:demand|sales|usage|volume|consumption)\s*(?:of|is|:|=)?\s*{num}",
        rf"\bdemand\s*(?:of|is|:|=)?\s*{num}\s*(?:units\s*(?:/|per)\s*year|units\s+annually)",
    ])
    average = first([
        rf"\b(?:average|mean|expected)\s+(?:daily|weekly|period)?\s*demand\s*(?:is|of|averages|equals|:|=)?\s*{num}",
        rf"\b(?:daily|weekly)\s+demand\s*(?:is|of|averages|equals|:|=)?\s*{num}",
        rf"\bdemand\s*(?:averages|average|mean)\s*{num}",
        rf"\bdemand\s*(?:of|is|:|=)?\s*{num}\s*(?:units\s*(?:/|per)\s*(?:day|week)|units\s+daily)",
        rf"\b(?:mean|expected)\s*(?:demand)?\s*(?:of|is|:|=)\s*{num}",
        rf"\b(?:demand|consumption)\s+rate\s*(?:of|is|:|=)?\s*{num}",
    ])
    # Demand deviation may be expressed in either word order. Exclude lead-time
    # standard deviations by requiring an explicit demand label when available.
    std = first([
        rf"\b(?:standard\s+deviation|std\.?\s*dev\.?|sigma)\s+of\s+(?:daily\s+|weekly\s+)?demand\s*(?:is|of|:|=)?\s*{num}",
        rf"\b(?:daily\s+|weekly\s+)?demand\s+(?:standard\s+deviation|std\.?\s*dev\.?|sigma)\s*(?:is|of|:|=)?\s*{num}",
        rf"\bstandard\s+deviation\s+in\s+demand\s*(?:is|of|:|=)?\s*{num}",
        rf"\b(?:daily|weekly)\s+standard\s+deviation\s*(?:is|of|:|=)?\s*{num}",
    ])
    if std is None:
        for m in re.finditer(rf"\b(?:standard\s+deviation|std\.?\s*dev\.?|sigma)\s*(?:is|of|:|=)?\s*{num}", text, re.IGNORECASE):
            preceding = text[max(0, m.start()-35):m.start()].lower()
            if not re.search(r"lead[\s-]*time\s*$", preceding):
                std = _clean_num(m.group(1))
                break
    # If annual demand not found, check for monthly demand and convert to annual
    if annual is None:
        # Pattern: e.g., "1,000 units per month" or "1000 units/month"
        m_month = re.search(r"([0-9][0-9,]*)(?:\s*units?)?\s*(?:per|/)\s*month", text, re.IGNORECASE)
        if m_month:
            monthly_val = _clean_num(m_month.group(1))
            if monthly_val is not None:
                annual = monthly_val * 12
    return annual, average, std


def extract_lead_times(text: str) -> Tuple[Optional[float], float]:
    """Extract lead-time mean and standard deviation, normalized to calendar days.

    Supports explicit lead-time standard deviations and the contextual form
    "lead time is 3 weeks with a standard deviation of 0.5 weeks".
    """
    num = r"([0-9][0-9,]*(?:\.[0-9]+)?)"
    unit = r"(days?|weeks?|months?)?"

    def in_days(value: str, units: Optional[str]) -> float:
        multiplier = 7 if units and units.lower().startswith("week") else (30 if units and units.lower().startswith("month") else 1)
        return float(value.replace(",", "")) * multiplier

    mean_patterns = [
        rf"\b(?:(?:average|mean|expected|supplier|delivery|replenishment|constant|fixed)\s+)?lead[\s-]*time\s*(?:is|of|equals|averages|:|=)?\s*{num}\s*{unit}",
        rf"\blead[\s-]*time\s+is\s+(?:a\s+)?constant\s+(?:of\s+)?{num}\s*{unit}",
    ]
    mean = None
    mean_match = None
    for pattern in mean_patterns:
        mean_match = re.search(pattern, text, re.IGNORECASE)
        if mean_match:
            mean = in_days(mean_match.group(1), mean_match.group(2))
            break

    std_patterns = [
        rf"\blead[\s-]*time\s+(?:standard\s+deviation|std\.?\s*dev\.?|sigma|variability)\s*(?:is|of|:|=)?\s*{num}\s*{unit}",
        rf"\b(?:standard\s+deviation|std\.?\s*dev\.?|sigma)\s+of\s+lead[\s-]*time\s*(?:is|of|:|=)?\s*{num}\s*{unit}",
    ]
    std = 0.0
    for pattern in std_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            std = in_days(match.group(1), match.group(2))
            return mean, std

    # Contextual lead-time variability: "lead time is 3 weeks with a
    # standard deviation of 0.5 weeks". Do not capture demand variability
    # earlier in the sentence.
    if mean_match:
        following = text[mean_match.end():]
        context = re.match(
            rf"\s*(?:,\s*)?(?:with|and)\s+(?:a\s+)?(?:standard\s+deviation|std\.?\s*dev\.?|sigma)\s*(?:is|of|:|=)?\s*{num}\s*{unit}",
            following, re.IGNORECASE,
        )
        if context:
            std = in_days(context.group(1), context.group(2))
    return mean, std


def extract_service_level(text: str) -> Optional[float]:
    """Extract cycle service level supporting percentages and decimal representations."""
    # 1. Percentage: "95% service level", "98% cycle service level", "in-stock probability 95%"
    m1 = re.search(
        r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:cycle\s+)?(?:service\s+level|target\s+service\s+level|in[\s\-]stock\s+probability|in[\s\-]stock\s+rate|desired\s+service\s+level|service\s+factor)",
        text,
        re.IGNORECASE,
    )
    if m1:
        val = _clean_num(m1.group(1))
        return val / 100.0 if val is not None else None

    # 2. Suffix percentage: "service level of 95%"
    m2 = re.search(
        r"(?:cycle\s+)?(?:service\s+level|target\s+service\s+level|in[\s\-]stock\s+probability|in[\s\-]stock\s+rate|desired\s+service\s+level)\s*(?:of|is|:|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        text,
        re.IGNORECASE,
    )
    if m2:
        val = _clean_num(m2.group(1))
        return val / 100.0 if val is not None else None

    # 3. Decimal: "service level of 0.95"
    m3 = re.search(
        r"(?:cycle\s+)?(?:service\s+level|target\s+service\s+level|in[\s\-]stock\s+probability)\s*(?:of|is|:|=)?\s*(0\.[0-9]+)",
        text,
        re.IGNORECASE,
    )
    if m3:
        return _clean_num(m3.group(1))

    return None


def extract_periodic_review_fields(text: str) -> Dict[str, Any]:
    """Extract review period, on-hand, on-order, backorders, and inventory position."""
    fields: Dict[str, Any] = {}

    # 1. Review Period
    m_p = re.search(
        r"(?:review\s+period|review\s+interval|review\s+cycle|order\s+cycle|replenishment\s+interval|"
        r"reviewed\s+every|checked\s+every|periodically\s+reviewed\s+every)\s*(?:of|is|:|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days|weeks)?",
        text,
        re.IGNORECASE,
    )
    if m_p:
        fields["review_period"] = _clean_num(m_p.group(1))
    else:
        m_p2 = re.search(r"every\s+([0-9]+)\s*(?:days|weeks)", text, re.IGNORECASE)
        if m_p2:
            fields["review_period"] = _clean_num(m_p2.group(1))

    # 2. Inventory Position (explicit)
    m_ip = re.search(
        r"(?:inventory\s+position|net\s+inventory\s+position|current\s+inventory\s+position)\s*(?:of|is|:|=)?\s*([0-9,]+(?:\.[0-9]+)?)",
        text,
        re.IGNORECASE,
    )
    if m_ip:
        fields["inventory_position"] = _clean_num(m_ip.group(1))

    # 3. On-hand inventory
    m_oh = re.search(
        r"(?:on[\s\-]hand\s+inventory|stock\s+on\s+hand|physical\s+stock|current\s+inventory|current\s+stock|on\s+hand)\s*(?:of|is|:|=)?\s*([0-9,]+(?:\.[0-9]+)?)",
        text,
        re.IGNORECASE,
    )
    if m_oh:
        fields["on_hand"] = _clean_num(m_oh.group(1))
    else:
        m_oh2 = re.search(r"([0-9,]+(?:\.[0-9]+)?)\s*(?:units\s+)?(?:currently\s+)?on[\s\-]hand", text, re.IGNORECASE)
        if m_oh2:
            fields["on_hand"] = _clean_num(m_oh2.group(1))

    # 4. On-order inventory
    m_oo = re.search(
        r"(?:on[\s\-]order\s+inventory|in[\s\-]transit\s+inventory|open\s+orders|pending\s+delivery|on\s+order|in\s+transit)\s*(?:of|is|:|=)?\s*([0-9,]+(?:\.[0-9]+)?)",
        text,
        re.IGNORECASE,
    )
    if m_oo:
        fields["on_order"] = _clean_num(m_oo.group(1))
    else:
        m_oo2 = re.search(r"([0-9,]+(?:\.[0-9]+)?)\s*(?:units\s+)?(?:currently\s+)?(?:on[\s\-]order|in[\s\-]transit)", text, re.IGNORECASE)
        if m_oo2:
            fields["on_order"] = _clean_num(m_oo2.group(1))

    # 5. Backorders
    m_bo = re.search(
        r"(?:backorders?|backlog|backordered\s+units?)\s*(?:of|is|:|=)?\s*([0-9,]+(?:\.[0-9]+)?)",
        text,
        re.IGNORECASE,
    )
    if m_bo:
        fields["backorders"] = _clean_num(m_bo.group(1))
    else:
        m_bo2 = re.search(r"([0-9,]+(?:\.[0-9]+)?)\s*(?:units\s+)?(?:backorders?|backordered|in\s+backlog)", text, re.IGNORECASE)
        if m_bo2:
            fields["backorders"] = _clean_num(m_bo2.group(1))

    return fields


def extract_newsvendor_fields(text: str) -> Dict[str, Any]:
    """Extract Newsvendor monetary parameters and demand parameters:
    - selling price
    - purchase cost
    - salvage value
    - underage / shortage cost
    - overage / excess cost
    """
    fields: Dict[str, Any] = {}

    # 1. Selling price
    m_sp = re.search(r"sell(?:s)?(?:\s+[\w\-]+){0,8}\s+for\s+\$?([0-9,]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if m_sp:
        fields["selling_price"] = _clean_num(m_sp.group(1))
    else:
        m_sp2 = re.search(
            r"(?:selling\s+price|retail\s+price|sales\s+price|retail\s+value|customer\s+price)\s*(?:of|is|:|=)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            text,
            re.IGNORECASE,
        )
        if m_sp2:
            fields["selling_price"] = _clean_num(m_sp2.group(1))

    # 2. Purchase / Unit acquisition cost
    m_pc = re.search(
        r"costs?\s+(?:us\s+)?\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:to\s+(?:acquire|produce|make|buy|purchase))?",
        text,
        re.IGNORECASE,
    )
    if m_pc and ("sell" not in text[max(0, m_pc.start()-10):m_pc.start()]):
        fields["unit_cost"] = _clean_num(m_pc.group(1))
    else:
        m_pc2 = re.search(
            r"(?:purchase\s+cost|acquisition\s+cost|cost\s+to\s+acquire|cost\s+per\s+unit|unit\s+cost|manufacturing\s+cost|procurement\s+cost)\s*(?:of|is|:|=)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            text,
            re.IGNORECASE,
        )
        if m_pc2:
            fields["unit_cost"] = _clean_num(m_pc2.group(1))

    # 3. Salvage value
    m_sal = re.search(
        r"(?:unsold\s+(?:units\s+)?)?salvage(?:d)?(?:\s+units)?\s*(?:for|value|price)?\s*(?:of|is|:|=)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
        text,
        re.IGNORECASE,
    )
    if m_sal:
        fields["salvage_value"] = _clean_num(m_sal.group(1))
    else:
        m_sal2 = re.search(
            r"(?:salvage\s+value|scrap\s+value|markdown\s+price|clearance\s+price)\s*(?:of|is|:|=)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            text,
            re.IGNORECASE,
        )
        if m_sal2:
            fields["salvage_value"] = _clean_num(m_sal2.group(1))

    # 4. Underage / Shortage cost (goodwill penalty or direct Cu)
    m_und = re.search(
        r"(?:underage\s+cost|shortage\s+cost|shortage\s+penalty|penalty\s+for\s+stockout|lost\s+goodwill|"
        r"goodwill\s+loss|cost\s+of\s+understocking|penalty)\s*(?:of|is|:|=)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
        text,
        re.IGNORECASE,
    )
    if m_und:
        fields["shortage_cost"] = _clean_num(m_und.group(1))

    # 5. Overage cost (explicit Co if given directly)
    m_ov = re.search(
        r"(?:overage\s+cost|excess\s+cost|cost\s+of\s+overstocking|loss\s+per\s+unsold\s+unit)\s*(?:of|is|:|=)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
        text,
        re.IGNORECASE,
    )
    if m_ov:
        fields["overage_cost"] = _clean_num(m_ov.group(1))

    return fields


# ============================================================================
# Main Inventory Decision Agent
# ============================================================================

class InventoryDecisionAgent:
    """Intelligent Supply Chain Inventory Decision Agent."""

    MODELS = {
        "EOQ": {
            "name": "Economic Order Quantity (EOQ)",
            "tool": "calculate_eoq",
            "verifier": tools.verify_eoq_result,
        },
        "QUANTITY_DISCOUNT": {
            "name": "All-Units Quantity Discount",
            "tool": "calculate_quantity_discount",
            "verifier": tools.verify_quantity_discount_result,
        },
        "SAFETY_STOCK_ROP": {
            "name": "Safety Stock & Reorder Point (Continuous Review)",
            "tool": "calculate_rop",
            "verifier": tools.verify_rop_result,
        },
        "FIXED_PERIOD_REVIEW": {
            "name": "Fixed-Period Review (Periodic Review / Order-Up-To Level)",
            "tool": "calculate_periodic_review",
            "verifier": tools.verify_periodic_review_result,
        },
        "NEWSVENDOR": {
            "name": "Newsvendor Model (Single-Period Perishable Inventory)",
            "tool": "calculate_newsvendor",
            "verifier": tools.verify_newsvendor_result,
        },
    }

    def __init__(self):
        pass

    # ------------------------------------------------------------------------
    # Step 1: Identify Model
    # ------------------------------------------------------------------------
    def identify_model(self, query: str) -> Tuple[str, float, str]:
        """Classify customer question into one of the 5 inventory models."""
        q = query.lower()

        scores = {
            "QUANTITY_DISCOUNT": 0.0,
            "NEWSVENDOR": 0.0,
            "FIXED_PERIOD_REVIEW": 0.0,
            "SAFETY_STOCK_ROP": 0.0,
            "EOQ": 0.0,
        }
        reasons = {k: [] for k in scores}

        # 1. Quantity Discount keywords
        if any(term in q for term in ["discount", "tier", "bracket", "price break", "volume pricing", "quantity break", "all-units", "all units"]):
            scores["QUANTITY_DISCOUNT"] += 4.0
            reasons["QUANTITY_DISCOUNT"].append("Mentions pricing tiers/volume quantity discounts")
        if re.search(r"\$\d+(\.\d+)?\s*(?:for|if|at|per).*\$\d+", q):
            scores["QUANTITY_DISCOUNT"] += 2.5
            reasons["QUANTITY_DISCOUNT"].append("Detected multiple price tier levels")

        # Price schedules may be described without the word "discount".
        # Require multiple prices with quantity thresholds to avoid misclassifying EOQ.
        price_tiers = self._parse_discount_schedule(query)
        if len(price_tiers) >= 2 and len({t["unit_price"] for t in price_tiers}) >= 2:
            scores["QUANTITY_DISCOUNT"] += 7.0
            reasons["QUANTITY_DISCOUNT"].append("Detected multiple unit prices tied to order quantity thresholds")

        # 2. Newsvendor keywords
        if any(term in q for term in ["newsvendor", "newsboy", "single period", "single-period", "perishable", "spoilage", "seasonal"]):
            scores["NEWSVENDOR"] += 5.0
            reasons["NEWSVENDOR"].append("Mentions single-period / perishable / seasonal goods")
        if any(term in q for term in ["salvage", "salvage value", "unsold", "critical fractile", "critical ratio", "underage", "overage", "scrap value"]):
            scores["NEWSVENDOR"] += 4.0
            reasons["NEWSVENDOR"].append("Mentions salvage value, underage, or overage trade-offs")
        if "selling price" in q and ("salvage" in q or "unsold" in q or "penalty" in q or "goodwill" in q or "cost to acquire" in q):
            scores["NEWSVENDOR"] += 3.0

        # 3. Fixed-Period Review keywords
        if any(term in q for term in ["periodic review", "fixed period", "fixed-period", "order-up-to", "order up to", "base stock", "review period", "review interval", "replenishment cycle"]):
            scores["FIXED_PERIOD_REVIEW"] += 5.0
            reasons["FIXED_PERIOD_REVIEW"].append("Mentions periodic review / order-up-to / review period")
        if any(term in q for term in ["on-hand", "on hand", "inventory position", "current inventory", "in transit", "on order", "backorder"]):
            scores["FIXED_PERIOD_REVIEW"] += 2.5
            reasons["FIXED_PERIOD_REVIEW"].append("Mentions current inventory position (on-hand, on-order, or backorders)")
        if re.search(r"every\s+(\d+|week|month|day)", q):
            scores["FIXED_PERIOD_REVIEW"] += 2.0

        # 4. Safety Stock & ROP keywords
        if any(term in q for term in ["safety stock", "reorder point", "rop", "buffer stock", "service level", "cycle service", "stockout risk"]):
            scores["SAFETY_STOCK_ROP"] += 4.5
            reasons["SAFETY_STOCK_ROP"].append("Mentions safety stock / reorder point / service level")
        if "lead time" in q and any(term in q for term in ["std", "deviation", "uncertainty", "sigma", "variance"]):
            scores["SAFETY_STOCK_ROP"] += 2.5
            reasons["SAFETY_STOCK_ROP"].append("Mentions lead time with demand variability")

        # 5. EOQ keywords
        if any(term in q for term in ["eoq", "economic order quantity", "wilson lot size", "optimal batch size", "optimal order quantity"]):
            scores["EOQ"] += 5.0
            reasons["EOQ"].append("Explicitly requests EOQ / Economic Order Quantity")
        if "annual demand" in q and any(term in q for term in ["holding cost", "carrying cost", "order cost", "setup cost", "setup expense", "cost of placing an order"]):
            scores["EOQ"] += 3.5
            reasons["EOQ"].append("Specifies standard EOQ parameters (annual demand, order cost, holding cost)")

        # Disambiguation heuristics
        if scores["FIXED_PERIOD_REVIEW"] > 0 and scores["SAFETY_STOCK_ROP"] > 0:
            if any(term in q for term in ["review period", "every", "order-up-to", "order up to", "base stock"]):
                scores["FIXED_PERIOD_REVIEW"] += 2.0
            else:
                scores["SAFETY_STOCK_ROP"] += 1.0

        if scores["QUANTITY_DISCOUNT"] >= 4.0:
            scores["EOQ"] = max(0.0, scores["EOQ"] - 2.0)

        sorted_models = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_model, top_score = sorted_models[0]

        if top_score == 0:
            top_model = "EOQ"
            top_score = 1.0
            rationale = "Defaulted to Economic Order Quantity (EOQ) based on general inventory ordering intent."
        else:
            rationale = "; ".join(reasons[top_model]) if reasons[top_model] else "Best match based on domain terminology."

        confidence = min(0.99, max(0.65, 0.60 + (top_score * 0.05)))
        return top_model, confidence, rationale

    # ------------------------------------------------------------------------
    # Step 2: Extract Inputs with Synonym Recognition & Missing Parameter Check
    # ------------------------------------------------------------------------
    def extract_inputs(self, query: str, model_id: str) -> Tuple[Dict[str, Any], List[str]]:
        """Extract domain parameters across the entire question.

        Returns (extracted_inputs, missing_required_parameters)
        """
        q = query
        inputs: Dict[str, Any] = {}
        missing: List[str] = []

        # Common Extractions
        order_cost = extract_ordering_cost(q)
        fixed_h, rate_h = extract_holding_cost(q)
        unit_price = extract_unit_price_or_cost(q)
        ann_demand, avg_demand, std_demand = extract_demands(q)
        avg_lt, std_lt = extract_lead_times(q)
        service_level = extract_service_level(q)

        if model_id == "EOQ":
            # Demands
            eff_demand = ann_demand or avg_demand
            inputs["annual_demand"] = eff_demand
            inputs["order_cost"] = order_cost
            inputs["holding_cost"] = fixed_h
            inputs["holding_cost_rate"] = rate_h
            inputs["unit_cost"] = unit_price or 0.0
            inputs["lead_time_days"] = avg_lt or 0.0
            inputs["working_days_per_year"] = 250

            # Check missing
            if eff_demand is None:
                missing.append("annual_demand (e.g. 'annual demand of 12,000 units')")
            if order_cost is None:
                missing.append("order_cost (e.g. 'ordering cost', 'setup cost', 'setup expense', or 'cost of placing an order')")
            if fixed_h is None and (rate_h is None or unit_price is None):
                missing.append("holding_cost (e.g. '$3/unit/year' or 'carrying cost rate of 20% with unit cost')")

        elif model_id == "QUANTITY_DISCOUNT":
            eff_demand = ann_demand or avg_demand
            schedule = self._parse_discount_schedule(q)
            inputs["annual_demand"] = eff_demand
            inputs["order_cost"] = order_cost
            inputs["holding_cost_rate"] = rate_h
            inputs["fixed_holding_cost"] = fixed_h
            inputs["discount_schedule"] = schedule

            if eff_demand is None:
                missing.append("annual_demand")
            if order_cost is None:
                missing.append("order_cost (e.g. 'order cost', 'setup cost')")
            if not schedule:
                missing.append("discount_schedule (e.g. '1-499: $20, 500+: $18')")
            if rate_h is None and fixed_h is None:
                missing.append("holding_cost (e.g. 'carrying cost rate 20%' or 'fixed holding cost $2')")

        elif model_id == "SAFETY_STOCK_ROP":
            eff_avg_demand = avg_demand or (ann_demand / 250.0 if ann_demand else None)
            inputs["avg_demand"] = eff_avg_demand
            inputs["std_demand"] = std_demand if std_demand is not None else 0.0
            inputs["lead_time"] = avg_lt
            inputs["std_lead_time"] = std_lt
            inputs["service_level"] = service_level
            inputs["unit_holding_cost"] = fixed_h
            inputs["time_unit"] = "days"

            if eff_avg_demand is None:
                missing.append("avg_demand (e.g. 'average daily demand of 50 units')")
            if avg_lt is None:
                missing.append("lead_time (e.g. 'lead time of 6 days')")
            # If demand or lead time is variable, service level is required
            if (inputs["std_demand"] > 0 or inputs["std_lead_time"] > 0) and service_level is None:
                missing.append("service_level (e.g. '95% service level')")

        elif model_id == "FIXED_PERIOD_REVIEW":
            pr_fields = extract_periodic_review_fields(q)
            eff_avg_demand = avg_demand or (ann_demand / 250.0 if ann_demand else None)
            inputs["review_period"] = pr_fields.get("review_period")
            inputs["lead_time"] = avg_lt
            inputs["avg_demand"] = eff_avg_demand
            inputs["std_demand"] = std_demand if std_demand is not None else 0.0
            inputs["service_level"] = service_level
            inputs["on_hand"] = pr_fields.get("on_hand", 0.0)
            inputs["on_order"] = pr_fields.get("on_order", 0.0)
            inputs["backorders"] = pr_fields.get("backorders", 0.0)
            inputs["inventory_position"] = pr_fields.get("inventory_position")
            inputs["time_unit"] = "days"

            if inputs["review_period"] is None:
                missing.append("review_period (e.g. 'review period of 7 days' or 'reviewed every 14 days')")
            if inputs["lead_time"] is None:
                missing.append("lead_time (e.g. 'lead time 3 days')")
            if eff_avg_demand is None:
                missing.append("avg_demand (e.g. 'average daily demand of 40 units')")
            if inputs["std_demand"] > 0 and service_level is None:
                missing.append("service_level (e.g. '98% service level')")

        elif model_id == "NEWSVENDOR":
            nv_fields = extract_newsvendor_fields(q)
            eff_mean = avg_demand or ann_demand
            inputs["selling_price"] = nv_fields.get("selling_price")
            inputs["unit_cost"] = nv_fields.get("unit_cost") or unit_price
            inputs["salvage_value"] = nv_fields.get("salvage_value", 0.0)
            inputs["shortage_cost"] = nv_fields.get("shortage_cost", 0.0)
            inputs["mean_demand"] = eff_mean
            inputs["std_demand"] = std_demand

            if inputs["selling_price"] is None:
                missing.append("selling_price (e.g. 'sell ... for $250' or 'retail price $250')")
            if inputs["unit_cost"] is None:
                missing.append("unit_cost / purchase cost (e.g. 'costs us $100 to acquire')")
            if inputs["mean_demand"] is None:
                missing.append("mean_demand (e.g. 'expected demand of 800 units')")
            if inputs["std_demand"] is None:
                missing.append("std_demand (e.g. 'demand standard deviation of 150 units')")

        return inputs, missing

    def _parse_discount_schedule(self, query: str) -> List[Dict[str, Any]]:
        """Read all-units discount brackets in common natural-language formats.

        Supported: '1-499 units at $10', '$10 for orders of 1–499 units',
        '1000 or more units at $9', and 'minimum quantity 500, unit cost $9.50'.
        """
        text = query.replace("–", "-").replace("—", "-").replace("−", "-")
        num = r"[0-9][0-9,]*(?:\.[0-9]+)?"
        price = r"\$\s*(" + num + r")"
        range_q = r"(" + num + r")\s*(?:-|to)\s*(" + num + r")"
        open_q = r"(" + num + r")\s*(?:\+|or\s+more|and\s+up|or\s+above)"
        matches = []

        def add(match, minimum, maximum, unit_price):
            matches.append((match.start(), {
                "min_qty": float(minimum.replace(",", "")),
                "max_qty": float(maximum.replace(",", "")) if maximum is not None else None,
                "unit_price": float(unit_price.replace(",", "")),
            }))

        # Quantity first, e.g. '1-499 units at $10', '1000+ units: $9'.
        for m in re.finditer(
            range_q + r"\s*(?:units?)?\s*(?:at|for|:|=)?\s*" + price,
            text, re.IGNORECASE,
        ):
            add(m, m.group(1), m.group(2), m.group(3))
        for m in re.finditer(
            open_q + r"\s*(?:units?)?\s*(?:at|for|:|=)?\s*" + price,
            text, re.IGNORECASE,
        ):
            add(m, m.group(1), None, m.group(2))

        # Price first, e.g. '$10 for orders of 1-499 units'.
        for m in re.finditer(
            price + r"\s*(?:per\s+unit\s*)?(?:for|at)\s*(?:orders?\s+of\s*)?" + range_q,
            text, re.IGNORECASE,
        ):
            add(m, m.group(2), m.group(3), m.group(1))
        for m in re.finditer(
            price + r"\s*(?:per\s+unit\s*)?(?:for|at)\s*(?:orders?\s+of\s*)?" + open_q,
            text, re.IGNORECASE,
        ):
            add(m, m.group(2), None, m.group(1))

        # Price-first lower bracket: '$12 per unit for orders below 500 units'.
        for m in re.finditer(
            price + r"\s*(?:per\s+unit\s*)?(?:for|at)\s*(?:orders?\s*)?(?:below|under|less\s+than)\s*(" + num + r")\s*(?:units?)?",
            text, re.IGNORECASE,
        ):
            upper = float(m.group(2).replace(",", "")) - 1
            add(m, "1", str(upper), m.group(1))

        # Price-first open bracket: '$11 for orders of 1,000 units or more'.
        for m in re.finditer(
            price + r"\s*(?:per\s+unit\s*)?(?:for|at)\s*(?:orders?\s+of\s*)?(" + num + r")\s*units?\s*(?:or\s+more|and\s+up|or\s+above)",
            text, re.IGNORECASE,
        ):
            add(m, m.group(2), None, m.group(1))

        # Explicit thresholds, e.g. 'minimum quantity 500, unit cost $9.50'.
        for m in re.finditer(
            r"(?:minimum\s+(?:order\s+)?quantity|min(?:imum)?\s+qty)\s*(?:of|:|=)?\s*(" + num +
            r")\s*[,;:]?\s*(?:unit\s+(?:cost|price)|price\s+per\s+unit)\s*(?:of|:|=)?\s*" + price,
            text, re.IGNORECASE,
        ):
            add(m, m.group(1), None, m.group(2))

        # 'under 500 at $10' is an inclusive bracket from quantity 1.
        for m in re.finditer(
            r"under\s*(" + num + r")\s*(?:units?)?\s*(?:at|:|=)\s*" + price,
            text, re.IGNORECASE,
        ):
            add(m, "1", str(float(m.group(1).replace(",", "")) - 1), m.group(2))

        if not matches:
            return []

        # Duplicate patterns may recognize the same bracket. Resolve by threshold.
        tiers_by_min = {}
        for _, tier in matches:
            key = tier["min_qty"]
            if key in tiers_by_min and tiers_by_min[key]["unit_price"] != tier["unit_price"]:
                return []  # conflicting schedule, request clarification
            old = tiers_by_min.get(key)
            if old is None or (old["max_qty"] is None and tier["max_qty"] is not None):
                tiers_by_min[key] = tier

        tiers = [tiers_by_min[k] for k in sorted(tiers_by_min)]
        for i, tier in enumerate(tiers):
            if tier["min_qty"] < 1 or tier["unit_price"] <= 0:
                return []
            if tier["max_qty"] is not None and tier["max_qty"] < tier["min_qty"]:
                return []
            if i + 1 < len(tiers):
                next_min = tiers[i + 1]["min_qty"]
                if tier["max_qty"] is None:
                    tier["max_qty"] = next_min - 1
                elif tier["max_qty"] != next_min - 1:
                    return []  # overlap or gap: do not invent a price
        return tiers

    # ------------------------------------------------------------------------
    # Step 3: Select Python Tool
    # ------------------------------------------------------------------------
    def select_tool(self, model_id: str) -> Tuple[str, Any]:
        """Select calculation function for the model."""
        model_info = self.MODELS.get(model_id, self.MODELS["EOQ"])
        tool_name = model_info["tool"]
        tool_fn = getattr(tools, tool_name)
        return tool_name, tool_fn

    # ------------------------------------------------------------------------
    # Step 4: Calculate
    # ------------------------------------------------------------------------
    def calculate(self, model_id: str, inputs: Dict[str, Any]) -> Any:
        """Execute calculation tool in inventory_tools.py without inventing inputs."""
        _, tool_fn = self.select_tool(model_id)
        # Filter inputs to only non-None arguments accepted by tool_fn
        clean_kwargs = {k: v for k, v in inputs.items() if v is not None}
        return tool_fn(**clean_kwargs)

    # ------------------------------------------------------------------------
    # Step 5: Verify
    # ------------------------------------------------------------------------
    def verify(self, model_id: str, result: Any) -> Dict[str, Any]:
        """Verify calculations using theoretical identities and bounds."""
        verifier = self.MODELS[model_id]["verifier"]
        return verifier(result)

    # ------------------------------------------------------------------------
    # Step 6: Explain
    # ------------------------------------------------------------------------
    def explain(
        self,
        query: str,
        model_id: str,
        inputs: Dict[str, Any],
        result: Any,
        verification: Dict[str, Any],
        missing_params: Optional[List[str]] = None,
    ) -> str:
        """Generate executive-grade supply chain explanation."""
        model_name = self.MODELS[model_id]["name"]
        lines = []

        lines.append(f"### Supply Chain Analytics Executive Decision Brief")
        lines.append(f"**Selected Model:** {model_name}")
        lines.append("")

        if missing_params:
            lines.append("#### ⚠️ Incomplete Information / Missing Parameters")
            lines.append(
                f"The question was classified as **{model_name}**, but one or more mandatory "
                "numerical parameters were missing from the customer question."
            )
            lines.append("")
            lines.append("**Extracted Parameters Found in Query:**")
            for k, v in inputs.items():
                if v is not None:
                    lines.append(f"- `{k}`: {v}")
            lines.append("")
            lines.append("**Required Parameters Still Missing:**")
            for mp in missing_params:
                lines.append(f"- ❌ **{mp}**")
            lines.append("")
            lines.append(
                "**Action Required:** In accordance with our deterministic supply chain analytics policy, "
                "the agent does not invent missing numerical inputs. Please supply the missing parameter(s) above "
                "to execute the exact optimization calculation."
            )
            return "\n".join(lines)

        # 1. Direct Decision Recommendation
        lines.append("#### 1. Core Recommendation & Decision Summary")
        if model_id == "EOQ":
            lines.append(
                f"- **Optimal Order Quantity (Q*):** **{result.eoq:,.2f} units** "
                f"(recommended batch size: **{int(round(result.eoq)):,} units**)."
            )
            lines.append(f"- **Order Frequency:** {result.orders_per_year:.2f} orders per year.")
            lines.append(f"- **Order Cycle Interval:** Every {result.order_cycle_days:.1f} operating days.")
            if result.reorder_point is not None:
                lines.append(f"- **Reorder Point (ROP):** Reorder when inventory drops to **{result.reorder_point:,.1f} units** (lead time: {result.lead_time_days:.0f} days).")
            lines.append(f"- **Total Annual Inventory Cost (Ordering + Holding):** **${result.total_inventory_cost:,.2f}**.")

        elif model_id == "QUANTITY_DISCOUNT":
            lines.append(
                f"- **Optimal Order Quantity:** **{result.optimal_order_quantity:,.2f} units** "
                f"(Tier {result.best_tier_index + 1} qualifying for price **${result.optimal_unit_price:,.2f}/unit**)."
            )
            lines.append(f"- **Total Annual Cost (Purchasing + Logistics):** **${result.total_annual_cost:,.2f}**.")
            lines.append(
                f"- **Net Annual Savings vs Baseline (No Discount):** **${result.total_annual_savings:,.2f}** "
                f"({(result.total_annual_savings / max(1.0, result.baseline_total_cost)) * 100:.2f}% savings)."
            )

        elif model_id == "SAFETY_STOCK_ROP":
            lines.append(f"- **Uncertainty Classification:** `{result.uncertainty_case.replace('_', ' ').title()}`.")
            lines.append(
                f"- **Reorder Point (ROP):** **{result.reorder_point:,.2f} units** "
                f"(order when inventory hits **{int(math.ceil(result.reorder_point)):,} units**)."
            )
            lines.append(
                f"- **Buffer / Safety Stock (SS):** **{result.safety_stock:,.2f} units** "
                f"(recommended buffer: **{int(math.ceil(result.safety_stock)):,} units**)."
            )
            if result.service_level is not None:
                lines.append(f"- **Cycle Service Level:** {result.service_level * 100:.1f}% (Stockout Risk: {result.stockout_risk * 100:.2f}%).")
                lines.append(f"- **Standard Normal Safety Factor (z):** {result.z_score:.4f}.")

        elif model_id == "FIXED_PERIOD_REVIEW":
            lines.append(
                f"- **Target Order-Up-To Level (Base Stock M):** **{result.target_inventory:,.2f} units** "
                f"(target ceiling: **{int(math.ceil(result.target_inventory)):,} units**)."
            )
            lines.append(
                f"- **Recommended Order Quantity Today (q):** **{result.recommended_order_quantity:,.2f} units** "
                f"(place order for **{int(math.ceil(result.recommended_order_quantity)):,} units**)."
            )
            lines.append(f"- **Safety Stock Component:** {result.safety_stock:,.2f} units covering the {result.protection_period:.1f}-{result.time_unit} protection interval (P + L).")
            lines.append(f"- **Current Net Inventory Position:** {result.inventory_position:,.2f} units (On-hand: {result.on_hand_inventory}, On-order: {result.on_order_inventory}, Backorders: {result.backorders}).")

        elif model_id == "NEWSVENDOR":
            lines.append(
                f"- **Optimal Order Quantity (Q*):** **{result.optimal_order_quantity:,.2f} units** "
                f"(recommended order: **{result.recommended_order_quantity_integer:,} units**)."
            )
            lines.append(f"- **Critical Ratio (Critical Fractile):** **{result.critical_ratio:.4f}** (target in-stock probability: {result.in_stock_probability * 100:.2f}%).")
            lines.append(f"- **Expected Financial Profit:** **${result.expected_profit:,.2f}**.")
            lines.append(f"- **Expected Sales:** {result.expected_sales:,.2f} units | **Expected Leftover Inventory:** {result.expected_leftover_units:,.2f} units.")
            lines.append(f"- **Expected Lost Sales:** {result.expected_lost_sales:,.2f} units.")
            lines.append(f"- **Value of Perfect Information (EVPI):** ${result.evpi:,.2f}.")

        lines.append("")

        # 2. Extracted Parameters Table
        lines.append("#### 2. Input Parameters & Baseline Data")
        for k, v in inputs.items():
            if k == "discount_schedule" and isinstance(v, list):
                lines.append(f"- `discount_schedule`: {len(v)} pricing tiers parsed")
            elif v is not None:
                lines.append(f"- `{k}`: {v}")
        lines.append("")

        # 3. Mathematical Foundations & Calculation Breakdown
        lines.append("#### 3. Mathematical Derivations & Step-by-Step Breakdown")
        if model_id == "EOQ":
            lines.append("1. **EOQ Formula:**")
            lines.append(r"   $$Q^* = \sqrt{\frac{2 \cdot D \cdot S}{H}}$$")
            lines.append(f"   $$Q^* = \\sqrt{{\\frac{{2 \\cdot {result.annual_demand:,.0f} \\cdot {result.order_cost:,.2f}}}{{{result.holding_cost:,.2f}}}}} = {result.eoq:,.2f}\\text{{ units}}$$")
            lines.append("2. **Cost Breakdown:**")
            lines.append(f"   - Annual Setup / Ordering Cost: $(D / Q^*) \\times S = ({result.annual_demand:,.0f} / {result.eoq:,.2f}) \\times {result.order_cost:,.2f} = \\mathbf{{\\${result.annual_ordering_cost:,.2f}}}$")
            lines.append(f"   - Annual Carrying / Holding Cost: $(Q^* / 2) \\times H = ({result.eoq:,.2f} / 2) \\times {result.holding_cost:,.2f} = \\mathbf{{\\${result.annual_holding_cost:,.2f}}}$")

        elif model_id == "QUANTITY_DISCOUNT":
            lines.append("1. **Tier Evaluation Matrix:**")
            lines.append("   | Tier | Quantity Range | Unit Price | Unconstrained EOQ | Evaluated Q | Total Annual Cost | Status |")
            lines.append("   | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
            for t in result.tier_evaluations:
                rng = f"{t.min_qty:,.0f} - {t.max_qty:,.0f}" if t.max_qty is not None else f"{t.min_qty:,.0f}+"
                lines.append(
                    f"   | Tier {t.tier_index+1} | {rng} | ${t.unit_price:.2f} | {t.raw_eoq:,.1f} | {t.candidate_qty:,.1f} | ${t.total_annual_cost:,.2f} | {t.status} |"
                )
            lines.append(f"2. **Optimal Decision:** Tier {result.best_tier_index + 1} achieves the global minimum total logistics and acquisition cost.")

        elif model_id == "SAFETY_STOCK_ROP":
            lines.append("1. **Lead Time Demand & Uncertainty:**")
            lines.append(f"   - Formula: `{result.intermediate_calculations.get('formula_used')}`")
            lines.append(f"   - Expected Lead Time Demand (LTD): $\\mu_{{LTD}} = \\bar{{d}} \\times L = {result.lead_time_demand:,.2f}\\text{{ units}}$")
            lines.append(f"   - Std Dev during Lead Time ($\\sigma_{{LT}}$): ${result.lead_time_demand_std:,.2f}\\text{{ units}}$")
            lines.append("2. **Safety Stock & ROP Equations:**")
            lines.append(f"   - Safety Stock: $SS = z \\times \\sigma_{{LT}} = {result.z_score:.4f} \\times {result.lead_time_demand_std:,.2f} = \\mathbf{{{result.safety_stock:,.2f}\\text{{ units}}}}$")
            lines.append(f"   - Reorder Point: $ROP = \\mu_{{LTD}} + SS = {result.lead_time_demand:,.2f} + {result.safety_stock:,.2f} = \\mathbf{{{result.reorder_point:,.2f}\\text{{ units}}}}$")

        elif model_id == "FIXED_PERIOD_REVIEW":
            lines.append("1. **Protection Period & Demand:**")
            lines.append(f"   - Protection Period: $T_{{prot}} = P + L = {result.protection_period:.1f}\\text{{ {result.time_unit}}}$")
            lines.append(f"   - Expected Demand over Protection Period: $D_{{prot}} = \\bar{{d}} \\times (P + L) = {result.demand_during_protection_period:,.2f}\\text{{ units}}$")
            lines.append(f"   - Std Dev over Protection Period: $\\sigma_{{prot}} = \\sigma_d \\times \\sqrt{{P + L}} = {result.std_demand_protection_period:,.2f}\\text{{ units}}$")
            lines.append("2. **Target Inventory (M) & Net Order Quantity (q):**")
            lines.append(f"   - Target Base Stock (M): $D_{{prot}} + SS = \\mathbf{{{result.target_inventory:,.2f}\\text{{ units}}}}$")
            lines.append(f"   - Recommended Order (q): $\\max(0, M - IP) = \\mathbf{{{result.recommended_order_quantity:,.2f}\\text{{ units}}}}$")

        elif model_id == "NEWSVENDOR":
            lines.append("1. **Underage ($C_u$) and Overage ($C_o$) Costs:**")
            lines.append(f"   - Underage Cost $C_u = P - C + g = {result.selling_price:.2f} - {result.unit_cost:.2f} + {result.shortage_cost:.2f} = \\mathbf{{\\${result.underage_cost:.2f}}}$")
            lines.append(f"   - Overage Cost $C_o = C - V = {result.unit_cost:.2f} - {result.salvage_value:.2f} = \\mathbf{{\\${result.overage_cost:.2f}}}$")
            lines.append(f"   - Critical Ratio: $CR = C_u / (C_u + C_o) = {result.critical_ratio:.4f}$")
            lines.append(f"   - Optimal Order Quantity: $Q^* = \\mu + z^* \\cdot \\sigma = \\mathbf{{{result.optimal_order_quantity:,.2f}\\text{{ units}}}}$")

        lines.append("")

        # 4. Verification Report
        lines.append("#### 4. Model Verification & Invariant Checks")
        lines.append(f"**Verification Status:** `{verification['status']}`")
        for check in verification["checks"]:
            mark = "✅" if check["passed"] else "❌"
            lines.append(f"- {mark} **{check['check']}**: {check['detail']}")
        lines.append("")

        # 5. Stated Assumptions
        lines.append("#### 5. Stated Assumptions & Policy Guidelines")
        for a in result.assumptions:
            lines.append(f"- {a}")

        return "\n".join(lines)

    # ------------------------------------------------------------------------
    # Pipeline Orchestrator: End-to-End Solve
    # ------------------------------------------------------------------------
    def solve(self, query: str) -> AgentResponse:
        """Execute the full 6-step architecture on a natural language customer query."""
        # 1. Identify Model
        model_id, confidence, rationale = self.identify_model(query)
        model_name = self.MODELS[model_id]["name"]

        # 2. Extract Inputs & Detect Missing Parameters
        extracted_inputs, missing_params = self.extract_inputs(query, model_id)

        # If mandatory parameters are missing from query, return structured clarification brief
        if missing_params:
            explanation = self.explain(
                query=query,
                model_id=model_id,
                inputs=extracted_inputs,
                result=None,
                verification={"status": "INCOMPLETE", "checks": []},
                missing_params=missing_params,
            )
            return AgentResponse(
                query=query,
                model_id=model_id,
                model_name=model_name,
                confidence=confidence,
                identification_rationale=rationale,
                extracted_inputs=extracted_inputs,
                missing_parameters=missing_params,
                tool_name=self.MODELS[model_id]["tool"],
                raw_result=None,
                result_dict={"status": "MISSING_INPUTS", "missing_parameters": missing_params},
                verification={"status": "INCOMPLETE", "checks": []},
                explanation=explanation,
            )

        # 3. Select Python Tool
        tool_name, _ = self.select_tool(model_id)

        # 4. Calculate
        raw_result = self.calculate(model_id, extracted_inputs)
        result_dict = raw_result.to_dict()

        # 5. Verify
        verification = self.verify(model_id, raw_result)

        # 6. Explain
        explanation = self.explain(
            query=query,
            model_id=model_id,
            inputs=extracted_inputs,
            result=raw_result,
            verification=verification,
        )

        return AgentResponse(
            query=query,
            model_id=model_id,
            model_name=model_name,
            confidence=confidence,
            identification_rationale=rationale,
            extracted_inputs=extracted_inputs,
            missing_parameters=[],
            tool_name=tool_name,
            raw_result=raw_result,
            result_dict=result_dict,
            verification=verification,
            explanation=explanation,
        )
