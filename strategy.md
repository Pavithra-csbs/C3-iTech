# Buyer Agent Strategy (YourBuyerAgent)
- Applies a slight downward adjustment for lower-grade products (e.g., grade B).
- Ensures the opening offer never exceeds buyer budget and is at least 1.


### Responding to seller offers


Decision order when a seller offer arrives:


1. **Immediate acceptance** if `seller_price <= aspiration` and `seller_price <= budget`.
2. **Late-round acceptance**: if the negotiation is past `late_round_threshold` and price is within `aspiration * (1 + late_accept_premium)`.
3. **Accept-to-close** in final rounds if price ≤ `fair` and within budget.
4. **Over-budget handling**: politely refuse or stretch (very small stretch near the end).
5. **Concession calculation**:
- Base concession = `base_concession` + `round_concession_step * (current_round - 1)`.
- Adjust concession by market trend (if seller offers are moving downwards, concede less).
- Add `deadlock_boost` if negotiation appears stalled.
- Cap concession at `max_concession`.


6. **Counteroffer computation**:
- `target = last_buyer + (seller_price - last_buyer) * (0.35 + concession)`
- `counter = min(target, seller_price - 1, budget)`
- Enforce `min_step` (absolute step size) and `small_gap` thresholds to avoid tiny incremental moves.
- If seller's estimated reservation (`est_min`) is known and the negotiation is late, ensure `counter` approaches `est_min` if that yields a better closing chance.


7. **Near-final handling**:
- If `seller_price - counter <= small_gap` treat as near-final and send a near-final message.


### Seller reservation estimation


- If there are multiple seller offers, compute average drop per offer and project a lower bound by multiplying the drop by `seller_min_projection_factor`.
- Fallback: estimate as `base_market_price * 0.85` when no seller data exists.


---


## Important Parameters (tunable)


- `max_rounds`: Max negotiation rounds before forced decisions (default 10).
- `desired_saving_pct`: Target savings vs fair price (default 0.12).
- `grade_adjustments`: Multipliers for A / EXPORT / B.
- `opening_jitter`: Randomization range for opening.
- `base_concession`, `round_concession_step`, `max_concession`: Controls concession pace.
- `deadlock_boost`: Extra concession applied when offers stall.
- `late_accept_premium`: How much over aspiration the agent may accept late.
- `seller_min_projection_factor`: Aggressiveness of projecting seller reservation.
- `small_gap_pct`, `min_step_pct`: Controls smallest meaningful price changes.


Tweak these values to make the agent more aggressive (higher concessions, higher late_accept_premium) or more conservative (lower concessions, higher min_step_pct).


---


## Concordia Integration Notes


- The agent exposes `get_state()` and `set_state(state)` to persist `name` and `personality` for Concordia's `ContextComponent` use.
- `make_pre_act_value(context)` formats a short summary string that Concordia can use to present pre-action context.
- The fallback `ContextComponentBase` in the code ensures local runs without Concordia are still possible.


---


## Practical examples


1. **Example product**


```text
base_market_price = ₹100,000
quality_grade = 'A' (multiplier 0.93)
=> fair = 93,000
=> aspiration = int(93,000 * (1 - 0.12)) = 81,840 -> 81,840