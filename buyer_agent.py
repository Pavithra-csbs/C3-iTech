"""
Final Buyer Agent with Advanced Strategies and Concordia Compatibility
Filename: buyer_agent.py
"""

from __future__ import annotations
import random
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

# ---------------------------
# Concordia imports (safe fallback)
# ---------------------------
try:
    from concordia.components import entity_component
    ContextComponentBase = entity_component.ContextComponent  # type: ignore
except Exception:  # fallback for local runs
    class ContextComponentBase:
        def get_state(self) -> Dict[str, Any]:
            return {}
        def set_state(self, state: Dict[str, Any]) -> None:
            pass
        def make_pre_act_value(self) -> str:
            return ""

# ============================================
# PART 1: DATA STRUCTURES
# ============================================

@dataclass
class Product:
    """Product being negotiated"""
    name: str
    category: str
    quantity: int
    quality_grade: str  # 'A', 'B', or 'Export'
    origin: str
    base_market_price: int  # Reference price for this product
    attributes: Dict[str, Any]

@dataclass
class NegotiationContext:
    """Current negotiation state"""
    product: Product
    your_budget: int
    current_round: int
    seller_offers: List[int]
    your_offers: List[int]
    messages: List[Dict[str, str]]

class DealStatus(Enum):
    ONGOING = "ongoing"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TIMEOUT = "timeout"

# ============================================
# PART 2: BASE AGENT CLASS
# ============================================

class BaseBuyerAgent(ABC):
    """Base class for all buyer agents"""
    def __init__(self, name: str):
        self.name = name
        self.personality = self.define_personality()

    @abstractmethod
    def define_personality(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        pass

    @abstractmethod
    def respond_to_seller_offer(
        self, context: NegotiationContext, seller_price: int, seller_message: str
    ) -> Tuple[DealStatus, int, str]:
        pass

    @abstractmethod
    def get_personality_prompt(self) -> str:
        pass

# ============================================
# YOUR IMPLEMENTATION STARTS HERE
# ============================================

class YourBuyerAgent(BaseBuyerAgent):
    """
    Advanced analytical buyer agent:
    ✔ Predictive concessions
    ✔ Deadlock detection
    ✔ Market-aware strategy
    ✔ Personality consistency
    ✔ Concordia-ready state & pre-act context
    """

    CONFIG = {
        "max_rounds": 10,
        "desired_saving_pct": 0.12,
        "grade_adjustments": {"A": 0.93, "EXPORT": 0.96, "B": 0.86},
        "opening_jitter": (-0.05, 0.08),
        "base_concession": 0.05,
        "round_concession_step": 0.025,
        "max_concession": 0.33,
        "deadlock_boost": 0.03,
        "late_accept_premium": 0.05,
        "seller_min_projection_factor": 1.5,
        "small_gap_pct": 0.015,
        "min_step_pct": 0.02,
        "late_round_threshold": 8,
        "stalled_round_check": 3
    }

    def define_personality(self) -> Dict[str, Any]:
        return {
            "personality_type": "analytical-negotiator",
            "traits": ["data-driven", "calm", "patient", "decisive"],
            "negotiation_style": (
                "Uses market reference and logical arguments. Keeps messages concise, data-backed."
            ),
            "catchphrases": [
                "Let's keep this efficient.",
                "I can make this work if the numbers make sense.",
                "Numbers should justify the move.",
                "Fair value matters more than hype."
            ]
        }

    # ---------- State Management ----------
    def get_state(self) -> Dict[str, Any]:
        return {"name": self.name, "personality": self.personality}

    def set_state(self, state: Dict[str, Any]) -> None:
        if state:
            self.name = state.get("name", self.name)
            self.personality = state.get("personality", self.personality)

    def make_pre_act_value(self, context: NegotiationContext) -> str:
        last_s = context.seller_offers[-1] if context.seller_offers else None
        last_b = context.your_offers[-1] if context.your_offers else None
        return (
            f"[Persona: {self.personality['personality_type']} | Traits: {', '.join(self.personality['traits'])}] "
            f"[Round {context.current_round}/{self.CONFIG['max_rounds']}] "
            f"[Last buyer: {last_b}; Last seller: {last_s}] "
            f"[Market ref: ₹{context.product.base_market_price:,}; Budget: ₹{context.your_budget:,}]"
        )

    # ---------- Price Modeling ----------
    def calculate_fair_and_target(self, product: Product) -> Dict[str, int]:
        base = product.base_market_price
        grade = product.quality_grade.upper()
        fair_multiplier = self.CONFIG["grade_adjustments"].get(grade, 0.9)
        fair = int(base * fair_multiplier)
        aspiration = int(max(1, fair * (1 - self.CONFIG["desired_saving_pct"])))
        reservation = fair
        return {"fair": fair, "aspiration": aspiration, "reservation": reservation}

    def estimate_seller_reservation(self, seller_offers: List[int], base_market_price: int) -> int:
        if not seller_offers:
            return int(base_market_price * 0.85)
        first, last = seller_offers[0], seller_offers[-1]
        if len(seller_offers) >= 2:
            avg_drop = (first - last) / (len(seller_offers) - 1)
            projected = int(last - max(avg_drop, 0) * self.CONFIG["seller_min_projection_factor"])
        else:
            projected = int(last - 0.05 * base_market_price)
        return max(min(projected, last), 1)

    # ---------- Opening Offer ----------
    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        prices = self.calculate_fair_and_target(context.product)
        aspiration = prices["aspiration"]
        jitter = random.uniform(*self.CONFIG["opening_jitter"])
        opening = int(aspiration * (1 + jitter))
        if context.product.quality_grade.upper() == "B":
            opening = int(opening * 0.97)
        opening = max(1, min(opening, context.your_budget))
        msg = (
            f"{random.choice(self.personality['catchphrases'])} "
            f"Market ref ₹{context.product.base_market_price:,}. Opening at ₹{opening:,} "
            f"(target ≤ ₹{aspiration:,})."
        )
        return opening, msg

    # ---------- Progress Analytics ----------
    def analyze_negotiation_progress(self, context: NegotiationContext) -> Dict[str, Any]:
        s_hist = context.seller_offers
        last_seller = s_hist[-1] if s_hist else None
        last_buyer = context.your_offers[-1] if context.your_offers else None
        trend = s_hist[-1] - s_hist[-2] if len(s_hist) >= 2 else None
        stalled = (
            len(s_hist) >= self.CONFIG["stalled_round_check"]
            and len(set(s_hist[-self.CONFIG["stalled_round_check"]:])) == 1
        )
        return {"last_seller": last_seller, "last_buyer": last_buyer, "trend": trend, "stalled": stalled}

    # ---------- Core Decision Logic ----------
    def respond_to_seller_offer(
        self, context: NegotiationContext, seller_price: int, seller_message: str
    ) -> Tuple[DealStatus, int, str]:
        cfg = self.CONFIG
        product = context.product
        budget = context.your_budget
        prices = self.calculate_fair_and_target(product)
        fair, aspiration, reservation = prices["fair"], prices["aspiration"], min(prices["reservation"], budget)
        prog = self.analyze_negotiation_progress(context)
        last_buyer = prog["last_buyer"] or int(product.base_market_price * 0.6)
        est_min = self.estimate_seller_reservation(context.seller_offers, product.base_market_price)

        # Accept conditions
        if seller_price <= aspiration and seller_price <= budget:
            return DealStatus.ACCEPTED, seller_price, f"Deal at ₹{seller_price:,}. {random.choice(self.personality['catchphrases'])}"
        if context.current_round >= cfg["late_round_threshold"] and seller_price <= min(budget, int(aspiration * (1 + cfg["late_accept_premium"]))):
            return DealStatus.ACCEPTED, seller_price, f"I can accept ₹{seller_price:,} to close now."
        if context.current_round >= (cfg["max_rounds"] - 2) and seller_price <= min(budget, fair):
            return DealStatus.ACCEPTED, seller_price, f"Accepting ₹{seller_price:,} to wrap this up."

        # Over budget
        if seller_price > budget:
            if context.current_round >= (cfg["max_rounds"] - 1) and seller_price <= int(budget * 1.03):
                return DealStatus.ONGOING, budget, f"That's above my budget. I can stretch to ₹{budget:,} now."
            return DealStatus.ONGOING, budget, f"Above my ceiling—best I can do is ₹{budget:,}."

        # Concessions
        conc = cfg["base_concession"] + cfg["round_concession_step"] * (context.current_round - 1)
        conc = min(conc, cfg["max_concession"])
        if prog["trend"] is not None:
            conc *= 0.9 if prog["trend"] < 0 else 1.15
        if prog["stalled"]:
            conc += cfg["deadlock_boost"]
        conc = min(conc, cfg["max_concession"])

        target = int(last_buyer + (seller_price - last_buyer) * (0.35 + conc))
        counter = min(target, seller_price - 1, budget)
        min_step = max(1000, int(cfg["min_step_pct"] * product.base_market_price))
        if counter <= last_buyer:
            counter = min(last_buyer + min_step, seller_price - 1, budget)
        if est_min and context.current_round >= 6 and counter < est_min:
            counter = min(budget, max(counter, est_min - int(0.01 * product.base_market_price)), seller_price - 1)

        gap = max(1000, int(cfg["small_gap_pct"] * product.base_market_price))
        near_final = (seller_price - counter) <= gap
        explain = f"Based on fair price ₹{fair:,}. " if fair else ""

        if near_final:
            return DealStatus.ONGOING, counter, f"{explain}I can go to ₹{counter:,}. Near-final offer."
        return DealStatus.ONGOING, counter, f"{explain}I can do ₹{counter:,}. {random.choice(self.personality['catchphrases'])}"

    def get_personality_prompt(self) -> str:
        return "Analytical-negotiator: clear, data-backed, calm, professional. Reference market price and keep messages concise."

# ============================================
# OPTIONAL TEST HARNESS (if required)
# ============================================

if __name__ == "__main__":
    print("YourBuyerAgent ready. Use the test harness to evaluate.")
