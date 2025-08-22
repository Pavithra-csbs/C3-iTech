# mock_seller.py
from dataclasses import dataclass

@dataclass
class MockSellerAgent:
    name: str
    initial_price: int      # Seller's starting ask (hidden to buyer at first)
    min_price: int          # Seller's hidden reservation price
    firmness: float = 0.35  # 0..1: higher = concedes slower

    def reset(self):
        self.current_price = self.initial_price
        self.round = 0

    def __post_init__(self):
        self.reset()

    def respond(self, buyer_offer: int) -> tuple[int, str]:
        """
        Return (seller_price, seller_message).
        - Accept if buyer >= min_price (or very close late).
        - Otherwise decrement toward min_price with a speed based on 'firmness'.
        """
        self.round += 1

        # Accept logic
        if buyer_offer >= self.min_price:
            ask = max(self.min_price, min(buyer_offer, self.current_price))
            self.current_price = ask
            return ask, f"Alright, we can close at ₹{ask:,}."

        # Late close if very close to min (sweetener)
        close_gap = max(1000, int(0.01 * self.initial_price))
        if self.round >= 8 and buyer_offer >= self.min_price - close_gap:
            self.current_price = self.min_price
            return self.min_price, f"Last call: ₹{self.min_price:,} and we have a deal."

        # Otherwise, counter downwards toward min
        gap = self.current_price - self.min_price
        step = max(1000, int(gap * (0.25 * (1 - self.firmness) + 0.1)))
        new_price = max(self.min_price, self.current_price - step)
        self.current_price = new_price
        return new_price, f"Quality won't come cheaper. I can do ₹{new_price:,}."
