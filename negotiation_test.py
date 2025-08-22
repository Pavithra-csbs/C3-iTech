# negotiation_test.py
from buyer_agent import YourBuyerAgent, Product, NegotiationContext, DealStatus
from seller_agent import MockSellerAgent

def run_single_scenario(title: str, product: Product, budget: int, seller_start: int, seller_min: int):
    print("\n" + "="*70)
    print(f"{title}")
    print("="*70)

    # Build agents + initial context
    buyer = YourBuyerAgent("Buyer")
    seller = MockSellerAgent("Seller", initial_price=seller_start, min_price=seller_min)

    ctx = NegotiationContext(
        product=product,
        your_budget=budget,
        current_round=1,
        seller_offers=[],
        your_offers=[],
        messages=[]
    )

    # Opening from buyer
    buyer_price, buyer_msg = buyer.generate_opening_offer(ctx)
    ctx.your_offers.append(buyer_price)
    ctx.messages.append({"role": "buyer", "text": buyer_msg})
    print(f"[Round {ctx.current_round}] Buyer → ₹{buyer_price:,} | {buyer_msg}")

    # Seller's first counter
    seller_price, seller_msg = seller.respond(buyer_price)
    ctx.seller_offers.append(seller_price)
    ctx.messages.append({"role": "seller", "text": seller_msg})
    print(f"[Round {ctx.current_round}] Seller → ₹{seller_price:,} | {seller_msg}")

    # Continue up to 10 rounds total
    MAX_ROUNDS = 10
    while ctx.current_round < MAX_ROUNDS:
        ctx.current_round += 1

        status, counter_price, out_msg = buyer.respond_to_seller_offer(ctx, seller_price, seller_msg)
        if status == DealStatus.ACCEPTED:
            print(f"[Round {ctx.current_round}] Buyer ACCEPTS @ ₹{counter_price:,} | {out_msg}")
            print("✅ DEAL CLOSED")
            return True, counter_price
        elif status in (DealStatus.REJECTED, DealStatus.TIMEOUT):
            print(f"[Round {ctx.current_round}] Buyer WALKS | {out_msg}")
            print("⛔ NO DEAL")
            return False, None

        # Otherwise, continue with counter
        ctx.your_offers.append(counter_price)
        ctx.messages.append({"role": "buyer", "text": out_msg})
        print(f"[Round {ctx.current_round}] Buyer → ₹{counter_price:,} | {out_msg}")

        # Seller replies
        seller_price, seller_msg = seller.respond(counter_price)
        ctx.seller_offers.append(seller_price)
        ctx.messages.append({"role": "seller", "text": seller_msg})
        print(f"[Round {ctx.current_round}] Seller → ₹{seller_price:,} | {seller_msg}")

        # If seller accepts implicitly (offers <= buyer counter), close
        if seller_price <= counter_price:
            print(f"[Round {ctx.current_round}] Seller ACCEPTS @ ₹{seller_price:,}")
            print("✅ DEAL CLOSED")
            return True, seller_price

    print("⏱️ Timeout reached. No deal.")
    return False, None


def run_negotiation_test():
    # Scenario 1: Easy Market
    s1_product = Product(
        name="Alphonso Mangoes", category="Fruit", quantity=100, quality_grade="A",
        origin="Ratnagiri", base_market_price=180_000, attributes={"type":"box"}
    )
    run_single_scenario(
        "Scenario 1: Easy Market (Grade-A, Budget 200k, Market 180k, SellerMin~150k)",
        product=s1_product, budget=200_000, seller_start=210_000, seller_min=150_000
    )

    # Scenario 2: Tight Budget
    s2_product = Product(
        name="Kesar Mangoes", category="Fruit", quantity=150, quality_grade="B",
        origin="Junagadh", base_market_price=150_000, attributes={"type":"box"}
    )
    run_single_scenario(
        "Scenario 2: Tight Budget (Grade-B, Budget 140k, Market 150k, SellerMin~125k)",
        product=s2_product, budget=140_000, seller_start=170_000, seller_min=125_000
    )

    # Scenario 3: Premium Product
    s3_product = Product(
        name="Export-Grade Mangoes", category="Fruit", quantity=50, quality_grade="Export",
        origin="Devgad", base_market_price=200_000, attributes={"type":"box"}
    )
    run_single_scenario(
        "Scenario 3: Premium Product (Export, Budget 190k, Market 200k, SellerMin~175k)",
        product=s3_product, budget=190_000, seller_start=230_000, seller_min=175_000
    )

if __name__ == "__main__":
    run_negotiation_test()
