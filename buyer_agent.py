# buyer_concordia_agent.py
import os
import json
import random
import time
import requests
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

# -------------------------
# Concordia base fallback
# -------------------------
try:
    from concordia.components import entity_component
    ContextComponentBase = entity_component.ContextComponent  # type: ignore
except Exception:
    # Minimal fallback so file can run without installing Concordia.
    class ContextComponentBase:
        def get_state(self) -> Dict[str, Any]:
            return {}
        def set_state(self, state: Dict[str, Any]) -> None:
            pass
        def make_pre_act_value(self) -> str:
            return ""

# -------------------------
# Domain dataclasses
# -------------------------
@dataclass
class Product:
    name: str
    category: str
    quantity: int
    quality_grade: str
    origin: str
    base_market_price: int
    attributes: Dict[str, Any]

@dataclass
class NegotiationContext:
    product: Product
    budget: int
    current_round: int
    seller_offers: List[int]
    buyer_offers: List[int]
    history: List[Dict[str, Any]]

class DealStatus(Enum):
    ONGOING = "ongoing"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TIMEOUT = "timeout"

# ====================================================
# Memory Component
# ====================================================
class MemoryComponent(ContextComponentBase):
    """Store and retrieve negotiation history (rounds, messages, offers)."""

    def __init__(self):
        self._history: List[Dict[str, Any]] = []

    def add_entry(self, entry: Dict[str, Any]) -> None:
        self._history.append(entry)

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def clear(self) -> None:
        self._history = []

    def get_state(self) -> Dict[str, Any]:
        return {"history": self._history}

    def set_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self._history = state.get("history", [])

    def make_pre_act_value(self) -> str:
        # Provide a short summary for the LLM: last 3 entries
        summary = self._history[-3:]
        if not summary:
            return "No negotiation history yet."
        s = "Recent negotiation history:\n"
        for e in summary:
            s += f"Round {e.get('round')} | Seller: ₹{e.get('seller_price')} | Buyer: {e.get('buyer_offer')} | Note: {e.get('note','')}\n"
        return s

# ====================================================
# Personality Component (Data Analyst)
# ====================================================
class BuyerPersonalityComponent(ContextComponentBase):
    """Your agent's personality definition (Data Analyst)."""

    def __init__(self):
        self.personality = {
            "archetype": "Data Analyst",
            "traits": ["data-driven", "calm", "concise", "evidence-first"],
            "negotiation_style": "Quotes market research, calculates fair prices, uses logical arguments, maintains professional tone."
        }
        self._notes: List[str] = []  # optional internal notes (e.g., justifications)

    def make_pre_act_value(self) -> str:
        # Return personality context that LLM can use to keep consistent role & tone
        traits = ", ".join(self.personality["traits"])
        return (
            f"Persona: {self.personality['archetype']} ({traits}). "
            f"Style: {self.personality['negotiation_style']}. "
            "Keep responses concise, reference market numbers and fairness, avoid threats or emotional language."
        )

    def add_note(self, note: str) -> None:
        self._notes.append(note)

    def get_state(self) -> Dict[str, Any]:
        return {"personality": self.personality, "notes": list(self._notes)}

    def set_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.personality = state.get("personality", self.personality)
        self._notes = state.get("notes", [])

# ====================================================
# Observation Component
# ====================================================
class ObservationComponent(ContextComponentBase):
    """
    Process seller messages and offers.
    Extract numeric price if present and classify message intent.
    """

    def __init__(self):
        pass

    def parse_seller_message(self, message: str) -> Dict[str, Any]:
        """
        Very simple parser: looks for '₹' or digits and returns the first price found.
        Returns a dict: {"price": int or None, "intent": str, "raw": message}
        """
        price = None
        tokens = message.replace(",", " ").split()
        for t in tokens:
            if t.startswith("₹"):
                try:
                    price = int(t.lstrip("₹").replace(",", ""))
                    break
                except:
                    continue
            else:
                # also accept plain digits
                cleaned = "".join(ch for ch in t if ch.isdigit())
                if cleaned and len(cleaned) >= 3:  # heuristic
                    try:
                        p = int(cleaned)
                        # sanity filter: price close to realistic range
                        price = p
                        break
                    except:
                        continue
        # intent heuristics
        lower = message.lower()
        if any(k in lower for k in ["final", "last", "take it", "take this", "firm"]):
            intent = "firm"
        elif any(k in lower for k in ["discount", "offer", "reduce", "negotiable", "quick deal"]):
            intent = "offer"
        else:
            intent = "inform"
        return {"price": price, "intent": intent, "raw": message}

    def make_pre_act_value(self) -> str:
        return "Observation component active: parses seller messages and extracts offered price and intent."

# ====================================================
# Decision Component
# ====================================================
class DecisionComponent(ContextComponentBase):
    """
    Implements negotiation strategy (Data Analyst).
    Numeric decisions come from this component; LLM is used only to phrase responses.
    """

    def __init__(self, personality: BuyerPersonalityComponent, memory: MemoryComponent,
                 backend: str = "ollama", ollama_model: str = "llama3",
                 ollama_url: str = "http://localhost:11434/api/generate"):
        self.personality = personality
        self.memory = memory
        self.backend = backend
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        # config
        self.max_rounds = 10
        self.desired_saving_pct = 0.12
        self.grade_adjustments = {"A": 0.95, "B": 0.88, "EXPORT": 0.97}
        self.base_concession = 0.04
        self.round_step = 0.03
        self.max_concession = 0.35

    def _price_model(self, product: Product) -> Dict[str, int]:
        base = product.base_market_price
        grade = product.quality_grade.upper()
        mult = self.grade_adjustments.get(grade, 0.9)
        fair = int(base * mult)
        aspiration = int(max(1, fair * (1 - self.desired_saving_pct)))
        reservation = fair
        return {"fair": fair, "aspiration": aspiration, "reservation": reservation}

    def decide_numeric(self, ctx: NegotiationContext, seller_price: int) -> Tuple[DealStatus, int, str]:
        product = ctx.product
        budget = ctx.budget
        prices = self._price_model(product)
        fair, aspiration, reservation = prices["fair"], prices["aspiration"], prices["reservation"]

        if seller_price <= aspiration and seller_price <= budget:
            note = f"Accepted because seller_price ({seller_price}) <= aspiration ({aspiration}) and within budget ({budget})."
            self.memory.add_entry({"round": ctx.current_round, "seller_price": seller_price, "buyer_offer": seller_price, "note": note})
            return DealStatus.ACCEPTED, seller_price, note

        if seller_price > budget:
            if ctx.current_round >= max(1, int(self.max_rounds * 0.8)):
                stretch = int(min(budget, seller_price * 0.99))
                note = f"Seller above budget. Stretching to {stretch} in late rounds."
                self.memory.add_entry({"round": ctx.current_round, "seller_price": seller_price, "buyer_offer": stretch, "note": note})
                return DealStatus.ONGOING, stretch, note
            note = f"Seller {seller_price} above budget {budget}. Holding at budget."
            self.memory.add_entry({"round": ctx.current_round, "seller_price": seller_price, "buyer_offer": budget, "note": note})
            return DealStatus.ONGOING, budget, note

        conc = self.base_concession + (ctx.current_round - 1) * self.round_step
        conc = min(conc, self.max_concession)

        if product.category.lower() in ("fruits", "vegetables", "produce"):
            conc *= 1.15

        last_buyer = ctx.buyer_offers[-1] if ctx.buyer_offers else int(product.base_market_price * 0.6)
        target = int(last_buyer + (seller_price - last_buyer) * (0.35 + conc))
        counter = min(target, seller_price - 1, budget)

        min_step = max(100, int(0.01 * product.base_market_price))
        if counter <= last_buyer:
            counter = min(last_buyer + min_step, seller_price - 1, budget)

        note = f"Countering at ₹{counter} (fair ₹{fair}, aspiration ₹{aspiration}, seller ₹{seller_price})"
        self.memory.add_entry({"round": ctx.current_round, "seller_price": seller_price, "buyer_offer": counter, "note": note})
        return DealStatus.ONGOING, counter, note

    def _build_prompt_for_llm(self, ctx: NegotiationContext, numeric_decision: Dict[str, Any], seller_msg: str) -> str:
        persona_block = self.personality.make_pre_act_value()
        history_block = self.memory.make_pre_act_value()
        prod = ctx.product
        return (
            f"Round {ctx.current_round}/{self.max_rounds}\n"
            f"Product: {prod.name} | Category: {prod.category} | Grade: {prod.quality_grade} | Market ref: ₹{prod.base_market_price}\n"
            f"Budget: ₹{ctx.budget}\n"
            f"Seller said: {seller_msg}\n"
            f"Numeric action: {numeric_decision['action']} at ₹{numeric_decision['price']}\n\n"
            f"Persona: {persona_block}\n"
            f"{history_block}\n\n"
            "Instruction: Write a concise, professional buyer reply consistent with the numeric action. "
            "Reference market numbers briefly and keep tone evidence-first."
        )

    def generate_llm_response(self, prompt: str) -> str:
        """Generate response using Ollama backend (or fallback)."""
        if self.backend == "ollama":
            try:
                payload = {"model": self.ollama_model, "prompt": prompt, "stream": False}
                response = requests.post(self.ollama_url, json=payload, timeout=150)

                raw = response.text.strip()
                if "\n" in raw:
                    raw = raw.split("\n")[0]

                try:
                    data = response.json()
                except Exception:
                    import json
                    data = json.loads(raw)

                return data.get("response", "").strip() if data else "[No response]"
            except Exception as e:
                return f"[LLM error: {e}]"

        return "[No LLM backend configured]"

    def make_decision(self, ctx: NegotiationContext, seller_msg: str) -> Dict[str, Any]:
        parsed_price = ctx.seller_offers[-1] if ctx.seller_offers else None
        if parsed_price is None:
            raise ValueError("No seller price available to decide on.")

        status, numeric_price, note = self.decide_numeric(ctx, parsed_price)
        numeric_decision = {"action": "accept" if status == DealStatus.ACCEPTED else "counter", "price": numeric_price, "note": note}

        prompt = self._build_prompt_for_llm(ctx, numeric_decision, seller_msg)
        llm_text = self.generate_llm_response(prompt)

        self.memory.add_entry({"round": ctx.current_round, "seller_price": parsed_price, "buyer_offer": numeric_price, "buyer_text": llm_text, "note": note})

        return {"status": status, "numeric_price": numeric_price, "text": llm_text, "note": note}

    def get_state(self) -> Dict[str, Any]:
        return {"backend": self.backend, "ollama_model": self.ollama_model}

    def set_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.backend = state.get("backend", self.backend)
        self.ollama_model = state.get("ollama_model", self.ollama_model)

    def make_pre_act_value(self) -> str:
        return f"Decision component: data-analyst strategy. Max rounds: {self.max_rounds}."

# ====================================================
# Orchestrator: run negotiation
# ====================================================
def seller_simulator(prev_price: Optional[int], base_price: int) -> Tuple[int, str]:
    """Seller starts above market and slowly reduces. Also sends short message."""
    if prev_price is None:
        p = int(base_price * 1.18)  # start at +18%
    else:
        # drop 0-4% of base per round
        drop = random.uniform(0.0, 0.04) * base_price
        p = max(int(prev_price - drop), int(base_price * 0.85))
    msg = f"Seller offers ₹{p:,}. Quick deal — negotiable."
    return p, msg

def score_outcome(ctx: NegotiationContext, closed_price: Optional[int]) -> Dict[str, Any]:
    score = {"deal_success": 0, "savings_pct": 0.0, "character_consistency": 0.0}
    if closed_price is not None:
        score["deal_success"] = 1
        savings = max(0, (ctx.budget - closed_price) / ctx.budget)
        score["savings_pct"] = savings
        # character_consistency heuristic: check if most buyer messages include marketplace numbers and calm tone
        history = ctx.history
        buyer_texts = [h.get("buyer_text", "") for h in history if h.get("buyer_text")]
        references = sum(1 for t in buyer_texts if any(s in t for s in ["₹", "market", "fair", "based"]))
        consistency = (references / max(1, len(buyer_texts))) if buyer_texts else 0.0
        score["character_consistency"] = consistency
    return score

def run_negotiation(
    product: Product,
    budget: int,
    backend: str = "ollama",
    ollama_model: str = "llama3",
    ollama_url: str = "http://localhost:11434/api/generate",
    max_rounds: int = 10
) -> Dict[str, Any]:
    # prepare components
    memory = MemoryComponent()
    personality = BuyerPersonalityComponent()
    decision = DecisionComponent(personality, memory, backend=backend, ollama_model=ollama_model, ollama_url=ollama_url)

    ctx = NegotiationContext(product=product, budget=budget, current_round=0, seller_offers=[], buyer_offers=[], history=[])
    closed_price = None

    seller_price = None
    seller_msg = "Initial offer."

    print(f"Starting negotiation for {product.name}. Budget: ₹{budget:,}. Market ref: ₹{product.base_market_price:,}.\nPersona: {personality.personality['archetype']}\n")

    for r in range(1, max_rounds + 1):
        ctx.current_round = r
        seller_price, seller_msg = seller_simulator(seller_price, product.base_market_price)
        ctx.seller_offers.append(seller_price)

        # record seller message into memory
        memory.add_entry({"round": r, "seller_price": seller_price, "seller_msg": seller_msg})
        # Decision component acts
        result = decision.make_decision(ctx, seller_msg)

        # attach buyer outputs to context.history
        entry = {
            "round": r,
            "seller_price": seller_price,
            "seller_msg": seller_msg,
            "buyer_offer": result["numeric_price"],
            "buyer_text": result["text"],
            "status": result["status"].value,
            "note": result["note"]
        }
        ctx.buyer_offers.append(result["numeric_price"])
        ctx.history.append(entry)

        # Print round summary
        print(f"[Round {r}] Seller: ₹{seller_price:,}")
        print(f"  Buyer numeric decision: ₹{result['numeric_price']} ({result['status'].name})")
        print(f"  Buyer message: {result['text']}\n")

        if result["status"] == DealStatus.ACCEPTED:
            closed_price = result["numeric_price"]
            print(f"Deal closed at ₹{closed_price:,} in round {r}.\n")
            break

    # scoring
    scores = score_outcome(ctx, closed_price)
    summary = {
        "closed_price": closed_price,
        "rounds_used": ctx.current_round,
        "scores": scores,
        "history": ctx.history
    }
    return summary

# ====================================================
# If run as main, run Mango negotiation
# ====================================================
if __name__ == "__main__":
    # product: Mango (1 ton)
    mango = Product(name="Mango (1 ton)", category="Fruits", quantity=1, quality_grade="A", origin="India", base_market_price=120000, attributes={})
    budget = 100000  # buyer's max budget

    # environment override for backend or model
    backend = os.getenv("LLM_BACKEND", "ollama")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

    result = run_negotiation(mango, budget, backend=backend, ollama_model=ollama_model, ollama_url=ollama_url, max_rounds=10)

    print("=== Negotiation Summary ===")
    print(f"Closed price: {result['closed_price']}")
    print(f"Rounds used: {result['rounds_used']}")
    print("Scores:", json.dumps(result["scores"], indent=2))
    print("\nHistory (last 10 entries):")
    for h in result["history"]:
        print(h)
