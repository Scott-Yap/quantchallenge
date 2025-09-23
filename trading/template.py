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
    def reset_state(self) -> None:
        # --- Core State ---
        self.pos = 0.0
        self.cash = 0.0
        self.mid = None
        self.best_bid = None
        self.best_ask = None
        self.tick = 0
        self.last_time_s = None

        # --- Rate Limiting ---
        self.tokens = 30.0
        self.token_cap = 30.0
        self.token_rate_per_s = 30.0 / 60.0

        # --- Order Management ---
        self.cur_bid_id = None
        self.cur_ask_id = None
        self.cur_bid_px = None
        self.cur_ask_px = None
        self.cur_qty = 0.0
        self.post_pending: List[Tuple[Side,float,float]] = []
        self.cancel_pending: List[int] = []

        # --- Quoting Logic Parameters ---
        self.min_life_ticks = 6
        self.last_quote_tick = -999999
        self.quote_every_ticks = 3
        self.quote_move_thr = 0.15
        self.size = 10 # size of orders placed
        self.gamma = 0.005 # risk aversion parameter
        self.inv_cap = 120 # max position size

        # --- Volatility Calculation ---
        self.vol_window = 100 # volatility based on last 100 ticks
        self.vol_alpha = 2 / (self.vol_window + 1) # smoothing factor for ewma
        self.last_mid_price = None
        self.ew_volatility_sq = 0.0

        # --- Spread Calculation ---
        self.base_half = 0.9 # base half-spread
        # TODO: parameter 'k' for order book density needs to be estimated
        self.kappa = 1

        # --- End-of-Game Flattening Logic ---
        self.hard_flat_time = 45.0
        self.soft_flat_time = 120.0
        self.spread_widen_late_mult = 1.4
        
        # --- Debugging ---
        self.last_print_tick = -9999

    def __init__(self) -> None:
        self.reset_state()

    def _update_volatility(self) -> None:
        if self.mid is None:
            return
        
        if self.last_mid_price is None:
            self.last_mid_price = self.mid
            return
        
        # avoid division by zero and stale prices (last known volatility stays during quiet moments)
        if self.last_mid_price <= 0 or self.mid == self.last_mid_price:
            return
        
        log_return = math.log(self.mid / self.last_mid_price)
        self.ew_volatility_sq = self.vol_alpha * (log_return ** 2) + (1 - self.vol_alpha) * self.ew_volatility_sq # Update EWMA of variance
        self.last_mid_price = self.mid

    def _refill_tokens(self, time_seconds: Optional[float]) -> None:
        if time_seconds is None:
            return
        if self.last_time_s is None:
            self.last_time_s = time_seconds
            return
        # time flows forward, so delta is current - last
        dt = time_seconds - self.last_time_s
        if dt > 0:
            self.tokens = min(self.token_cap, self.tokens + dt * self.token_rate_per_s)
        self.last_time_s = time_seconds

    def _spend(self, n: float) -> bool:
        if self.tokens >= n:
            self.tokens -= n
            return True
        print("algo_print: [RATE] skip; budget hit")
        return False

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

    def _target_quotes(self, t_left: Optional[float]) -> Optional[Tuple[float,float,float]]:
        if self.mid is None or t_left is None:
            return None
        
        res_price = self._fair() - self.pos * self.gamma * self.ew_volatility_sq * t_left

        half_spread = (self.gamma * self.ew_volatility_sq * t_left) / 2.0 + (1.0 / self.gamma) * math.log(1 + self.gamma / self.kappa)

        half_spread *= self._late_mult(t_left)

        bid = res_price - half_spread
        ask = res_price + half_spread
        return (bid, ask, self.size)

    def _need_refresh(self, new_bid: float, new_ask: float) -> bool:
        if self.cur_bid_px is None or self.cur_ask_px is None:
            return True
        return abs(new_bid - self.cur_bid_px) >= self.quote_move_thr or abs(new_ask - self.cur_ask_px) >= self.quote_move_thr

    def _stage_cancel_current(self) -> None:
        if self.cur_bid_id is not None:
            self.cancel_pending.append(self.cur_bid_id)
            self.cur_bid_id = None
        if self.cur_ask_id is not None:
            self.cancel_pending.append(self.cur_ask_id)
            self.cur_ask_id = None

    def _stage_post_quotes(self, bid: float, ask: float, qty: float) -> None:
        self.post_pending.append((Side.BUY, bid, qty))
        self.post_pending.append((Side.SELL, ask, qty))
        self.cur_bid_px = bid
        self.cur_ask_px = ask
        self.cur_qty = qty

    def _drain_buffers(self) -> None:
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

    def _flatten_if_needed(self, t_left: Optional[float]) -> None:
        if t_left is None:
            return
        if t_left <= self.hard_flat_time and abs(self.pos) > 0:
            side = Side.SELL if self.pos > 0 else Side.BUY
            qty = abs(self.pos)
            if self._spend(1.0):
                place_market_order(side, Ticker.TEAM_A, qty)
                print(f"algo_print: [FLAT] t={self.tick} tl={t_left:.1f} mkt {side.name} qty={qty:.2f}")

    def _maybe_quote(self, t_left: Optional[float]) -> None:
        quotes = self._target_quotes(t_left)
        if quotes is None:
            return
        bid, ask, qty = quotes
        if (self.tick - self.last_quote_tick) < self.quote_every_ticks:
            return
        if not self._need_refresh(bid, ask) and (self.tick - self.last_quote_tick) < self.min_life_ticks:
            return
        self._stage_cancel_current()
        self._stage_post_quotes(bid, ask, qty)
        self.last_quote_tick = self.tick

    def _print_state(self, t_left: Optional[float]) -> None:
        if self.mid is None:
            return
        if self.tick - self.last_print_tick < 5:
            return
        mtm = self.cash + (self.mid * self.pos if self.mid is not None else 0.0)
        print(f"algo_print: [STATE] t={self.tick} tl={t_left if t_left is not None else -1:.1f} mid={self.mid:.2f} pos={self.pos:.2f} cash={self.cash:.2f} mtm={mtm:.2f} tok={self.tokens:.1f}")
        self.last_print_tick = self.tick

    def on_trade_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        self.mid = price

    def on_orderbook_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        if side == Side.BUY:
            if self.best_bid is None or price > self.best_bid:
                self.best_bid = price
        else:
            if self.best_ask is None or price < self.best_ask:
                self.best_ask = price
        if self.best_bid is not None and self.best_ask is not None and self.best_bid <= self.best_ask:
            self.mid = 0.5 * (self.best_bid + self.best_ask)
            print(f"algo_print: [BOOK] New mid from book: {self.mid:.2f} (B/A: {self.best_bid:.2f}/{self.best_ask:.2f})")

    def on_account_update(self, ticker: Ticker, side: Side, price: float, quantity: float, capital_remaining: float) -> None:
        if side == Side.BUY:
            self.pos += quantity
            self.cash -= price * quantity
        else:
            self.pos -= quantity
            self.cash += price * quantity

        print(f"algo_print: [FILL] Filled our {side.name} order for {quantity} @ {price:.2f}. New pos: {self.pos:.2f}, cash: {self.cash:.2f}")
        self.pos = max(-self.inv_cap, min(self.pos, self.inv_cap))

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
        t_left = time_seconds
        self._flatten_if_needed(t_left)
        self._maybe_quote(t_left)
        self._drain_buffers()
        self._print_state(t_left)
        if event_type == "END_GAME":
            self._stage_cancel_current()
            self._drain_buffers()
            self.reset_state()

    def on_orderbook_snapshot(self, ticker: Ticker, bids: list, asks: list) -> None:
        self.best_bid = bids[0][0] if bids else None
        self.best_ask = asks[0][0] if asks else None
        if self.best_bid is not None and self.best_ask is not None and self.best_bid <= self.best_ask:
            self.mid = 0.5 * (self.best_bid + self.best_ask)
        print(f"algo_print: [BOOK] Received full snapshot. Mid: {self.mid:.2f} Best B/A: {self.best_bid:.2f}/{self.best_ask:.2f}")