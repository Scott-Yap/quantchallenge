"""
Quant Challenge 2025

Algorithmic strategy template
"""

import math
from enum import Enum
from typing import Optional, Dict, Tuple, List

class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    # TEAM_A (home team)
    TEAM_A = 0

def place_market_order(side: Side, ticker: Ticker, quantity: float) -> None:
    """Place a market order.
    
    Parameters
    ----------
    side
        Side of order to place
    ticker
        Ticker of order to place
    quantity
        Quantity of order to place
    """
    return

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    """Place a limit order.
    
    Parameters
    ----------
    side
        Side of order to place
    ticker
        Ticker of order to place
    quantity
        Quantity of order to place
    price
        Price of order to place
    ioc
        Immediate or cancel flag (FOK)

    Returns
    -------
    order_id
        Order ID of order placed
    """
    return 0

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    """Cancel an order.
    
    Parameters
    ----------
    ticker
        Ticker of order to cancel
    order_id
        Order ID of order to cancel

    Returns
    -------
    success
        True if order was cancelled, False otherwise
    """
    return 0

class Strategy:
    # =========================
    # Lifecycle / State
    # =========================
    def __init__(self) -> None:
        self.reset_state()

    def reset_state(self) -> None:
        # --- Core State ---
        self.pos: float = 0.0
        self.cash: float = 0.0
        self.mid: Optional[float] = None
        self.best_bid: Optional[float] = None
        self.best_ask: Optional[float] = None
        self.best_bid_qty: Optional[float] = None
        self.best_ask_qty: Optional[float] = None
        self.tick: int = 0
        self.last_time_s: Optional[float] = None

        # --- Rate Limiting ---
        self.tokens: float = 30.0
        self.token_cap: float = 30.0
        self.token_rate_per_s: float = 30.0 / 60.0  # 0.5 token/sec

        # --- Order Management ---
        self.cur_bid_id: Optional[int] = None
        self.cur_ask_id: Optional[int] = None
        self.cur_bid_px: Optional[float] = None
        self.cur_ask_px: Optional[float] = None
        self.cur_qty: float = 0.0
        self.post_pending: List[Tuple["Side", float, float]] = []
        self.cancel_pending: List[int] = []

        # --- Quoting Logic Parameters (tuned) ---
        self.min_life_ticks: int = 6
        self.last_quote_tick: int = -10**9
        self.quote_every_ticks: int = 3
        self.quote_move_thr: float = 0.15

        # Inventory / risk (looser spreads, gentler skew)
        self.size: float = 10.0
        self.gamma: float = 0.0025          # ↓ from 0.005 (less risk aversion, narrower A–S spread)
        self.inv_cap: float = 150.0         # ↑ a bit if allowed

        # Volatility (EWMA of log returns)
        self.vol_window: int = 100
        self.vol_alpha: float = 2.0 / (self.vol_window + 1)
        self.last_mid_price: Optional[float] = None
        self.ew_volatility_sq: float = 1e-8

        # A–S order book density param (higher -> tighter)
        self.kappa: float = 2.5             # ↑ from 1.0 helps reduce half-spread term

        # Momentum (fast/slow EWMA crossover)
        self.fast_ewma: Optional[float] = None
        self.slow_ewma: Optional[float] = None
        self.alpha_fast: float = 2.0 / (10 + 1)   # ~10-tick half-life
        self.alpha_slow: float = 2.0 / (50 + 1)   # ~50-tick half-life
        self.momentum_bias: float = 0.0           # -1, 0, +1
        self.mom_coef: float = 0.15               # price skew (in price units) per bias

        # Sparse-book quoting (fallbacks when sizes are tiny/None)
        self.min_top_qty: float = 10.0            # target top-of-book depth per side
        self.sparse_allow: bool = True
        self.sparse_qty: float = 3.0              # tiny feeler quotes when book is empty
        self.sparse_spread_mult: float = 1.6      # widen spreads in sparse mode

        # End-of-Game flattening
        self.hard_flat_time: float = 45.0
        self.soft_flat_time: float = 120.0
        self.spread_widen_late_mult: float = 1.4

        # Trailing take-profit / stop-loss
        self.anchor_mid: Optional[float] = None   # reset on new/increased position in a direction
        self.best_favorable_mid: Optional[float] = None
        self.last_pos_sign: int = 0               # -1 short, +1 long, 0 flat

        self.tp_retrace: float = 0.8              # retrace (price units) to lock profit
        self.tp_qty: float = 5.0                  # how much to realize on trigger
        self.sl_retrace: float = 1.6              # max adverse from entry anchor
        self.sl_qty: float = 5.0                  # how much to reduce on stop

        # Debug cadence
        self.last_print_tick: int = -10**9

    # =========================
    # Helpers
    # =========================
    def _refill_tokens(self, time_seconds: Optional[float]) -> None:
        # time_seconds is "time left" (decreases), so dt = last - current
        if time_seconds is None:
            return
        if self.last_time_s is None:
            self.last_time_s = time_seconds
            return
        dt = self.last_time_s - time_seconds
        if dt > 0:
            self.tokens = min(self.token_cap, self.tokens + dt * self.token_rate_per_s)
        self.last_time_s = time_seconds

    def _spend(self, n: float) -> bool:
        if self.tokens >= n:
            self.tokens -= n
            return True
        print("algo_print: [RATE] skip; budget hit")
        return False

    def _update_volatility(self) -> None:
        if self.mid is None:
            return
        if self.last_mid_price is None:
            self.last_mid_price = self.mid
            return
        if self.last_mid_price > 0 and self.mid > 0 and self.mid != self.last_mid_price:
            log_ret = math.log(self.mid / self.last_mid_price)
            self.ew_volatility_sq = (
                self.vol_alpha * (log_ret ** 2) +
                (1.0 - self.vol_alpha) * self.ew_volatility_sq
            )
            self.ew_volatility_sq = max(self.ew_volatility_sq, 1e-8)
        self.last_mid_price = self.mid

    def _update_momentum(self) -> None:
        if self.mid is None:
            return
        if self.fast_ewma is None or self.slow_ewma is None:
            self.fast_ewma = self.mid
            self.slow_ewma = self.mid
            self.momentum_bias = 0.0
            return
        self.fast_ewma = self.alpha_fast * self.mid + (1 - self.alpha_fast) * self.fast_ewma
        self.slow_ewma = self.alpha_slow * self.mid + (1 - self.alpha_slow) * self.slow_ewma
        if self.fast_ewma > self.slow_ewma:
            self.momentum_bias = 1.0
        elif self.fast_ewma < self.slow_ewma:
            self.momentum_bias = -1.0
        else:
            self.momentum_bias = 0.0

    def _mtm_mid(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return 0.5 * (self.best_bid + self.best_ask)
        return self.mid

    def _fair(self) -> Optional[float]:
        return self._mtm_mid()

    def _late_mult(self, t_left: Optional[float]) -> float:
        if t_left is None:
            return 1.0
        if t_left <= self.hard_flat_time:
            return 2.0
        if t_left <= self.soft_flat_time:
            return self.spread_widen_late_mult
        return 1.0

    def _need_refresh(self, new_bid: float, new_ask: float) -> bool:
        if self.cur_bid_px is None or self.cur_ask_px is None:
            return True
        return (abs(new_bid - self.cur_bid_px) >= self.quote_move_thr or
                abs(new_ask - self.cur_ask_px) >= self.quote_move_thr)

    # ---------- Liquidity gating (relaxed + fallback) ----------
    def _sufficient_liquidity(self, min_qty: float) -> Tuple[bool, bool, bool]:
        """
        Returns (bid_ok, ask_ok, sparse_mode).
        - If either side meets half the threshold -> ok for that side.
        - If both sides are tiny/None and sparse_allow -> enter sparse_mode (quote tiny & wider).
        """
        bq = self.best_bid_qty if self.best_bid_qty is not None else 0.0
        aq = self.best_ask_qty if self.best_ask_qty is not None else 0.0
        bid_ok = bq >= 0.5 * min_qty
        ask_ok = aq >= 0.5 * min_qty
        sparse_mode = False
        if not bid_ok and not ask_ok and self.sparse_allow:
            # Only enter sparse mode if we actually *have* prices to lean on
            if self.best_bid is not None and self.best_ask is not None:
                sparse_mode = True
        print(
            f"algo_print: [LIQUIDITY] best_bid_qty={bq:.2f} best_ask_qty={aq:.2f} "
            f"threshold={min_qty:.1f} -> bid_ok={bid_ok} ask_ok={ask_ok} sparse={sparse_mode}"
        )
        return bid_ok, ask_ok, sparse_mode

    # ---------- A–S target w/ momentum skew ----------
    def _target_quotes(self, t_left: Optional[float], sparse: bool) -> Optional[Tuple[float, float, float]]:
        fair = self._fair()
        if fair is None or t_left is None:
            return None

        # Reservation price: inventory skew + momentum skew
        inv_skew = self.pos * self.gamma * self.ew_volatility_sq * t_left
        mom_skew = -self.mom_coef * self.momentum_bias   # +bias -> push fair *up*; we subtract on bid, add on ask via res_price
        res_price = fair - inv_skew + mom_skew

        # Optimal half-spread (A–S)
        half = 0.5 * self.gamma * self.ew_volatility_sq * t_left \
               + (1.0 / self.gamma) * math.log(1.0 + self.gamma / max(self.kappa, 1e-8))
        half *= self._late_mult(t_left)

        if sparse:
            half *= self.sparse_spread_mult
            qty = self.sparse_qty
        else:
            qty = self.size

        bid = res_price - half
        ask = res_price + half
        return (bid, ask, qty)

    # ---------- Posting & draining ----------
    def _stage_cancel_current(self) -> None:
        if self.cur_bid_id is not None:
            self.cancel_pending.append(self.cur_bid_id)
            self.cur_bid_id = None
        if self.cur_ask_id is not None:
            self.cancel_pending.append(self.cur_ask_id)
            self.cur_ask_id = None

    def _stage_post_quotes(self, bid: Optional[float], ask: Optional[float], qty: float, bid_on: bool, ask_on: bool) -> None:
        # Post only the permitted sides
        if bid_on and bid is not None:
            self.post_pending.append((Side.BUY, bid, qty))
            self.cur_bid_px = bid
        else:
            self.cur_bid_px = None
            self.cur_bid_id = None

        if ask_on and ask is not None:
            self.post_pending.append((Side.SELL, ask, qty))
            self.cur_ask_px = ask
        else:
            self.cur_ask_px = None
            self.cur_ask_id = None

        if bid_on or ask_on:
            self.cur_qty = qty

    def _drain_buffers(self) -> None:
        # cancels first (cheap safety), then posts; each op spends a token
        while self.cancel_pending and self._spend(1.0):
            oid = self.cancel_pending.pop(0)
            cancel_order(Ticker.TEAM_A, oid)
        while self.post_pending and self._spend(1.0):
            side, px, qty = self.post_pending.pop(0)
            oid = place_limit_order(side, Ticker.TEAM_A, qty, px, ioc=False)
            if side == Side.BUY:
                self.cur_bid_id = oid
            else:
                self.cur_ask_id = oid

    # ---------- Risk actions ----------
    def _flatten_if_needed(self, t_left: Optional[float]) -> None:
        if t_left is None:
            return
        if t_left <= self.hard_flat_time and abs(self.pos) > 0:
            side = Side.SELL if self.pos > 0 else Side.BUY
            qty = abs(self.pos)
            if self._spend(1.0):
                place_market_order(side, Ticker.TEAM_A, qty)
                print(f"algo_print: [FLAT] t={self.tick} tl={t_left:.1f} mkt {side.name} qty={qty:.2f}")

    def _update_trailing_anchors(self) -> None:
        # Reset anchors when position sign changes or increases in that direction
        curr_sign = 0 if self.pos == 0 else (1 if self.pos > 0 else -1)
        if curr_sign != self.last_pos_sign:
            self.anchor_mid = self.mid
            self.best_favorable_mid = self.mid
            self.last_pos_sign = curr_sign
            return
        # Extend favorable extreme
        if self.mid is None or curr_sign == 0 or self.best_favorable_mid is None:
            return
        if curr_sign > 0:
            # long: favorable if price goes up
            if self.mid > self.best_favorable_mid:
                self.best_favorable_mid = self.mid
        else:
            # short: favorable if price goes down
            if self.mid < self.best_favorable_mid:
                self.best_favorable_mid = self.mid

    def _take_profit_if_needed(self) -> None:
        if self.mid is None or self.pos == 0 or self.best_favorable_mid is None:
            return
        if self.pos > 0:
            # long: lock in if retrace down from best_favorable
            if self.mid <= self.best_favorable_mid - self.tp_retrace:
                qty = min(self.tp_qty, self.pos)
                if self._spend(1.0):
                    place_market_order(Side.SELL, Ticker.TEAM_A, qty)
                    print(f"algo_print: [TP] Long retrace -> SELL {qty:.2f} @ mkt")
        else:
            # short: lock in if retrace up from best_favorable
            if self.mid >= self.best_favorable_mid + self.tp_retrace:
                qty = min(self.tp_qty, abs(self.pos))
                if self._spend(1.0):
                    place_market_order(Side.BUY, Ticker.TEAM_A, qty)
                    print(f"algo_print: [TP] Short retrace -> BUY {qty:.2f} @ mkt")

    def _stop_out_if_needed(self) -> None:
        if self.mid is None or self.pos == 0 or self.anchor_mid is None:
            return
        if self.pos > 0:
            # long: adverse if price below anchor by sl_retrace
            if self.mid <= self.anchor_mid - self.sl_retrace:
                qty = min(self.sl_qty, self.pos)
                if self._spend(1.0):
                    place_market_order(Side.SELL, Ticker.TEAM_A, qty)
                    print(f"algo_print: [SL] Long stop -> SELL {qty:.2f} @ mkt")
                self.anchor_mid = self.mid  # reset anchor after stop
                self.best_favorable_mid = self.mid
        else:
            # short: adverse if price above anchor by sl_retrace
            if self.mid >= self.anchor_mid + self.sl_retrace:
                qty = min(self.sl_qty, abs(self.pos))
                if self._spend(1.0):
                    place_market_order(Side.BUY, Ticker.TEAM_A, qty)
                    print(f"algo_print: [SL] Short stop -> BUY {qty:.2f} @ mkt")
                self.anchor_mid = self.mid
                self.best_favorable_mid = self.mid

    # ---------- Quoting driver ----------
    def _maybe_quote(self, t_left: Optional[float]) -> None:
        # cadence
        if (self.tick - self.last_quote_tick) < self.quote_every_ticks:
            return

        bid_ok, ask_ok, sparse = self._sufficient_liquidity(self.min_top_qty)

        quotes = self._target_quotes(t_left, sparse)
        if quotes is None:
            return
        bid, ask, qty = quotes

        # If neither side has depth and we don't allow sparse, skip
        if not sparse and not bid_ok and not ask_ok:
            print("algo_print: [QUOTE] Skipped due to low book depth on BOTH sides")
            return

        # Allow one-sided quoting:
        want_bid = bid_ok or sparse
        want_ask = ask_ok or sparse

        # Only refresh if price moved OR quotes lived long enough
        refresh_needed = self._need_refresh(bid, ask)
        if not refresh_needed and (self.tick - self.last_quote_tick) < self.min_life_ticks:
            return

        self._stage_cancel_current()
        self._stage_post_quotes(bid, ask, qty, want_bid, want_ask)
        if want_bid or want_ask:
            self.last_quote_tick = self.tick

    # ---------- Logging ----------
    def _print_state(self, t_left: Optional[float]) -> None:
        if self.mid is None:
            return
        if self.tick - self.last_print_tick < 5:
            return
        mtm = self.cash + (self.mid * self.pos if self.mid is not None else 0.0)
        tl = t_left if t_left is not None else -1.0
        print(
            f"algo_print: [STATE] t={self.tick} tl={tl:.1f} mid={self.mid:.2f} "
            f"pos={self.pos:.2f} cash={self.cash:.2f} mtm={mtm:.2f} tok={self.tokens:.1f}"
        )
        self.last_print_tick = self.tick

    # =========================
    # Event Handlers
    # =========================
    def on_trade_update(self, ticker: "Ticker", side: "Side", quantity: float, price: float) -> None:
        self.mid = price

    def on_orderbook_update(self, ticker: "Ticker", side: "Side", quantity: float, price: float) -> None:
        if side == Side.BUY:
            if self.best_bid is None or price > self.best_bid:
                self.best_bid = price
                self.best_bid_qty = quantity
            elif self.best_bid is not None and price == self.best_bid:
                self.best_bid_qty = quantity
        else:
            if self.best_ask is None or price < self.best_ask:
                self.best_ask = price
                self.best_ask_qty = quantity
            elif self.best_ask is not None and price == self.best_ask:
                self.best_ask_qty = quantity

        if self.best_bid is not None and self.best_ask is not None and self.best_bid <= self.best_ask:
            self.mid = 0.5 * (self.best_bid + self.best_ask)
            print(f"algo_print: [BOOK] New mid from book: {self.mid:.2f} (B/A: {self.best_bid:.2f}/{self.best_ask:.2f})")

    def on_account_update(self, ticker: "Ticker", side: "Side", price: float, quantity: float, capital_remaining: float) -> None:
        if side == Side.BUY:
            self.pos += quantity
            self.cash -= price * quantity
        else:
            self.pos -= quantity
            self.cash += price * quantity

        # cap inventory
        self.pos = max(-self.inv_cap, min(self.pos, self.inv_cap))
        print(f"algo_print: [FILL] {side.name} {quantity:.2f} @ {price:.2f} -> pos={self.pos:.2f} cash={self.cash:.2f}")

        # update trailing anchors on fills
        self._update_trailing_anchors()

    def on_game_event_update(self,
                             event_type: str,
                             home_away: str,
                             home_score: int,
                             away_score: int,
                             player_name: Optional[str],
                             substituted_player_name: Optional[str],
                             shot_type: Optional[str],
                             assist_player: Optional[str],
                             rebound_type: Optional[str],
                             coordinate_x: Optional[float],
                             coordinate_y: Optional[float],
                             time_seconds: Optional[float]) -> None:
        self.tick += 1
        self._refill_tokens(time_seconds)
        self._update_volatility()
        self._update_momentum()
        self._update_trailing_anchors()

        t_left = time_seconds
        self._flatten_if_needed(t_left)     # hard late-game flatten
        self._take_profit_if_needed()       # realize MTM into cash
        self._stop_out_if_needed()          # cut losers
        self._maybe_quote(t_left)           # (re)quote
        self._drain_buffers()
        self._print_state(t_left)

        if event_type == "END_GAME":
            self._stage_cancel_current()
            self._drain_buffers()
            self.reset_state()

    def on_orderbook_snapshot(self, ticker: "Ticker", bids: list, asks: list) -> None:
        # bids/asks like [(price, qty), ...]
        if bids:
            self.best_bid = bids[0][0]
            self.best_bid_qty = bids[0][1] if len(bids[0]) > 1 else None
        else:
            self.best_bid = None
            self.best_bid_qty = None
        if asks:
            self.best_ask = asks[0][0]
            self.best_ask_qty = asks[0][1] if len(asks[0]) > 1 else None
        else:
            self.best_ask = None
            self.best_ask_qty = None

        if self.best_bid is not None and self.best_ask is not None and self.best_bid <= self.best_ask:
            self.mid = 0.5 * (self.best_bid + self.best_ask)

        if self.mid is not None and self.best_bid is not None and self.best_ask is not None:
            print(f"algo_print: [BOOK] Snapshot mid={self.mid:.2f} bestB/A={self.best_bid:.2f}/{self.best_ask:.2f}")