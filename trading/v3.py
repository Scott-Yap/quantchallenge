"""
Quant Challenge 2025

Algorithmic strategy template
"""

from enum import Enum
from typing import Optional, Dict, Tuple, List
import math

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
        self.post_pending: List[Tuple[Side, float, float]] = []
        self.cancel_pending: List[int] = []

        # --- Quoting Logic Parameters ---
        self.min_life_ticks: int = 6
        self.last_quote_tick: int = -10**9
        self.quote_every_ticks: int = 3
        self.quote_move_thr: float = 0.15

        # Avellaneda–Stoikov style params
        self.size: float = 10.0
        self.gamma: float = 0.003            # a bit looser than before
        self.inv_cap: float = 150.0          # allow a touch more inventory

        # --- Volatility (EWMA of log returns) ---
        self.vol_window: int = 100
        self.vol_alpha: float = 2.0 / (self.vol_window + 1)
        self.last_mid_price: Optional[float] = None
        self.ew_volatility_sq: float = 1e-8  # small positive floor

        # --- Order book density param (tunable) ---
        self.kappa: float = 1.5              # slightly higher (tighter spreads)

        # --- End-of-Game Flattening Logic ---
        self.hard_flat_time: float = 45.0
        self.soft_flat_time: float = 120.0
        self.spread_widen_late_mult: float = 1.4

        # --- Logging helpers ---
        self.last_print_tick: int = -10**9
        self.last_home_score: Optional[int] = None
        self.last_away_score: Optional[int] = None
        self.last_mid_for_deriv: Optional[float] = None
        self.last_time_for_deriv: Optional[float] = None

    def __init__(self) -> None:
        self.reset_state()

    # ---------- Helpers ----------
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

    def _target_quotes(self, t_left: Optional[float]) -> Optional[Tuple[float, float, float]]:
        if t_left is None:
            return None
        fair = self._fair()
        if fair is None:
            return None

        # Reservation price with inventory skew (A–S)
        res_price = fair - self.pos * self.gamma * self.ew_volatility_sq * t_left

        # Optimal half-spread (A–S)
        half_spread = (
            0.5 * self.gamma * self.ew_volatility_sq * t_left
            + (1.0 / self.gamma) * math.log(1.0 + self.gamma / max(self.kappa, 1e-8))
        )
        half_spread *= self._late_mult(t_left)

        bid = res_price - half_spread
        ask = res_price + half_spread
        return (bid, ask, self.size)

    def _need_refresh(self, new_bid: float, new_ask: float) -> bool:
        if self.cur_bid_px is None or self.cur_ask_px is None:
            return True
        return (
            abs(new_bid - self.cur_bid_px) >= self.quote_move_thr or
            abs(new_ask - self.cur_ask_px) >= self.quote_move_thr
        )

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
        # Cancels first, then posts (each costs 1 token)
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

    # -------- Liquidity check (safe + one-sided) --------
    def _sufficient_liquidity(self, min_qty: float = 10.0) -> Tuple[bool, bool]:
        bid_sz = self.best_bid_qty if self.best_bid_qty is not None else 0.0
        ask_sz = self.best_ask_qty if self.best_ask_qty is not None else 0.0
        bid_ok = bid_sz >= 0.5 * min_qty
        ask_ok = ask_sz >= 0.5 * min_qty
        print(
            f"algo_print: [LIQUIDITY] best_bid_qty={bid_sz:.2f} best_ask_qty={ask_sz:.2f} "
            f"threshold={min_qty:.1f} -> bid_ok={bid_ok} ask_ok={ask_ok}"
        )
        return bid_ok, ask_ok

    def _maybe_quote(self, t_left: Optional[float]) -> None:
        quotes = self._target_quotes(t_left)
        if quotes is None:
            return
        bid, ask, qty = quotes

        # Rate limit quoting cadence
        if (self.tick - self.last_quote_tick) < self.quote_every_ticks:
            return

        bid_ok, ask_ok = self._sufficient_liquidity(min_qty=8.0)  # a bit looser
        if not bid_ok and not ask_ok:
            print("algo_print: [QUOTE] Skipped due to low book depth on BOTH sides")
            return

        # Only refresh if moved meaningfully, unless quotes are old enough
        if not self._need_refresh(bid, ask) and (self.tick - self.last_quote_tick) < self.min_life_ticks:
            return

        # Cancel existing then (re)post only sides that pass the liquidity check
        self._stage_cancel_current()

        posted_any = False
        if bid_ok:
            self.post_pending.append((Side.BUY, bid, qty))
            self.cur_bid_px = bid
            posted_any = True
        else:
            self.cur_bid_px = None
            self.cur_bid_id = None

        if ask_ok:
            self.post_pending.append((Side.SELL, ask, qty))
            self.cur_ask_px = ask
            posted_any = True
        else:
            self.cur_ask_px = None
            self.cur_ask_id = None

        if posted_any:
            self.cur_qty = qty
            self.last_quote_tick = self.tick

    # -------- Data dump helpers --------
    def _compute_dmid_dt(self, time_seconds: Optional[float]) -> float:
        if self.mid is None or time_seconds is None:
            return 0.0
        if self.last_mid_for_deriv is None or self.last_time_for_deriv is None:
            self.last_mid_for_deriv = self.mid
            self.last_time_for_deriv = time_seconds
            return 0.0
        # time_seconds is "time left", so dt = prev - current
        dt = self.last_time_for_deriv - time_seconds
        dmid = self.mid - self.last_mid_for_deriv
        self.last_mid_for_deriv = self.mid
        self.last_time_for_deriv = time_seconds
        if dt is None or dt <= 0:
            return 0.0
        return dmid / dt

    def _dump_row(self,
                  tag: str,
                  event_type: str,
                  home_score: Optional[int],
                  away_score: Optional[int],
                  t_left: Optional[float]) -> None:
        pd = None
        if home_score is not None and away_score is not None:
            pd = home_score - away_score

        best_bid = self.best_bid if self.best_bid is not None else float('nan')
        best_ask = self.best_ask if self.best_ask is not None else float('nan')
        best_bid_qty = self.best_bid_qty if self.best_bid_qty is not None else 0.0
        best_ask_qty = self.best_ask_qty if self.best_ask_qty is not None else 0.0
        mid = self._mtm_mid()
        spread = float('nan')
        if self.best_bid is not None and self.best_ask is not None:
            spread = self.best_ask - self.best_bid

        mtm = 0.0
        if mid is not None:
            mtm = self.cash + mid * self.pos

        vol_ew = (self.ew_volatility_sq ** 0.5) if self.ew_volatility_sq > 0 else 0.0
        dmid_dt = self._compute_dmid_dt(t_left)

        # Format safe values
        mid_val = mid if mid is not None else float('nan')
        tl_val = t_left if t_left is not None else float('nan')
        h = home_score if home_score is not None else -1
        a = away_score if away_score is not None else -1
        pd_val = pd if pd is not None else float('nan')

        print(
            "algo_print: [DATA-{tag}] "
            f"t={self.tick} tl={tl_val:.6f} event={event_type} "
            f"home={h} away={a} pd={pd_val} "
            f"mid={mid_val:.6f} spread={spread:.6f} "
            f"best_bid={best_bid:.6f} best_ask={best_ask:.6f} "
            f"best_bid_qty={best_bid_qty:.6f} best_ask_qty={best_ask_qty:.6f} "
            f"pos={self.pos:.6f} cash={self.cash:.6f} mtm={mtm:.6f} "
            f"vol_ew={vol_ew:.6f} dmid_dt={dmid_dt:.6f}".replace("{tag}", tag)
        )

    def _print_state(self, t_left: Optional[float]) -> None:
        if self.mid is None:
            return
        if self.tick - self.last_print_tick < 5:
            return
        mtm = self.cash + (self.mid * self.pos if self.mid is not None else 0.0)
        tl_disp = t_left if t_left is not None else -1.0
        print(
            f"algo_print: [STATE] t={self.tick} tl={tl_disp:.1f} mid={self.mid:.2f} "
            f"pos={self.pos:.2f} cash={self.cash:.2f} mtm={mtm:.2f} tok={self.tokens:.1f}"
        )
        self.last_print_tick = self.tick

    # ---------- Event handlers ----------
    def on_trade_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        self.mid = price

    def on_orderbook_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
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

    def on_account_update(self, ticker: Ticker, side: Side, price: float, quantity: float, capital_remaining: float) -> None:
        if side == Side.BUY:
            self.pos += quantity
            self.cash -= price * quantity
        else:
            self.pos -= quantity
            self.cash += price * quantity

        # hard cap inventory
        self.pos = max(-self.inv_cap, min(self.pos, self.inv_cap))
        print(f"algo_print: [FILL] {side.name} {quantity:.2f} @ {price:.2f} -> pos={self.pos:.2f} cash={self.cash:.2f}")

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

        # --- Always dump one row per tick ---
        self._dump_row(tag="TICK",
                       event_type=event_type if event_type else "NOTHING",
                       home_score=home_score,
                       away_score=away_score,
                       t_left=time_seconds)

        # --- Extra dump on score change ---
        if (self.last_home_score is not None and self.last_away_score is not None
            and (home_score != self.last_home_score or away_score != self.last_away_score)):
            self._dump_row(tag="SCORE",
                           event_type=event_type if event_type else "SCORE_CHANGE",
                           home_score=home_score,
                           away_score=away_score,
                           t_left=time_seconds)

        # Update score memory
        self.last_home_score = home_score
        self.last_away_score = away_score

        # Core trading loop
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
        # Expecting [(price, qty), ...]; guard for empties
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