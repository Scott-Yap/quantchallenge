#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <utility>
#include <mutex>
#include <algorithm> // max_element, min_element
#include <cmath>     // std::fabs
#include <cstdlib>   // std::abs (integral)

enum class Side
{
  buy = 0,
  sell = 1
};

enum class Ticker : std::uint8_t
{
  TEAM_A = 0
}; // NOLINT

bool place_market_order(Side side, Ticker ticker, float quantity);
void println(const std::string &text);

class Strategy
{
  float bankroll = 100000.0f;
  double last_trade_time = -1.0;

  // We only keep top-of-book for reference pricing.
  std::vector<std::pair<float, float>> local_bids; // best bid at [0]
  std::vector<std::pair<float, float>> local_asks; // best ask at [0]
  std::vector<std::pair<float, float>> full_bids;
  std::vector<std::pair<float, float>> full_asks;
  std::mutex mu_;

public:
  void reset_state()
  {
    bankroll = 100000.0f;
    last_trade_time = -1.0;
    std::lock_guard<std::mutex> g(mu_);
    local_bids.clear();
    local_asks.clear();
  }

  Strategy() { reset_state(); }

  void on_trade_update(Ticker ticker, Side side, float quantity, float price) {}

  // Always reflect the latest tick as top-of-book for that side.
  void on_orderbook_update(Ticker ticker, Side side, float quantity, float price)
  {
    std::lock_guard<std::mutex> g(mu_);
    auto &book = (side == Side::buy) ? local_bids : local_asks;

    // If quantity <= 0, treat as a basic delete of current top if price matches.
    if (quantity <= 0.0f)
    {
      if (!book.empty() && book.front().first == price)
        book.clear();
      return;
    }

    if (book.empty())
    {
      book.clear();
      book.emplace_back(price, quantity);
    }
    else
    {
      // Replace/update the current top regardless of "improvement".
      book.front() = {price, quantity};
      if (book.size() > 1)
        book.resize(1);
    }
  }

  void on_account_update(Ticker ticker, Side side, float price, float quantity, float capital_remaining) {}

  // Normalize snapshots to true best bid/ask (store only best-of-book).
  virtual void on_orderbook_snapshot(
      Ticker ticker,
      const std::vector<std::pair<float, float>> &bids,
      const std::vector<std::pair<float, float>> &asks)
  {
    std::lock_guard<std::mutex> g(mu_);

    if (!bids.empty())
    {
      auto it = std::max_element(
          bids.begin(), bids.end(),
          [](const auto &a, const auto &b)
          { return a.first < b.first; });
      local_bids.clear();
      local_bids.push_back(*it);
    }
    else
    {
      local_bids.clear();
    }

    if (!asks.empty())
    {
      auto it = std::min_element(
          asks.begin(), asks.end(),
          [](const auto &a, const auto &b)
          { return a.first < b.first; });
      local_asks.clear();
      local_asks.push_back(*it);
    }
    else
    {
      local_asks.clear();
    }
  }

  virtual void on_game_event_update(
      const std::string &event_type,
      const std::string &home_away,
      int home_score,
      int away_score,
      const std::optional<std::string> &player_name,
      const std::optional<std::string> &substituted_player_name,
      const std::optional<std::string> &shot_type,
      const std::optional<std::string> &assist_player,
      const std::optional<std::string> &rebound_type,
      const std::optional<double> &coordinate_x,
      const std::optional<double> &coordinate_y,
      const std::optional<double> &time_seconds)
  {
    (void)home_away;
    (void)player_name;
    (void)substituted_player_name;
    (void)shot_type;
    (void)assist_player;
    (void)rebound_type;
    (void)coordinate_x;
    (void)coordinate_y;

    if (event_type == "END_GAME")
    {
      reset_state();
      return;
    }
    if (!time_seconds.has_value() || bankroll <= 0.0f)
      return;

    float t = static_cast<float>(*time_seconds);

    // Avoid duplicate actions at (nearly) the same timestamp.
    if (std::fabs(static_cast<double>(t) - last_trade_time) < 1e-9)
      return;
    last_trade_time = static_cast<double>(t);

    int diff_signed = home_score - away_score;
    if (diff_signed == 0)
      return;

    Side side = (diff_signed > 0) ? Side::buy : Side::sell;

    // Use absolute gap for sizing decisions, keep sign in side.
    int diff = std::abs(diff_signed);

    float qty_notional = 0.0f;

    if (t <= 10.0f && t > 5.0f)
    {
      if (diff >= 6)
        qty_notional = bankroll * 0.70f;
      else if (diff >= 5)
        qty_notional = bankroll * 0.50f;
      else if (diff >= 3)
        qty_notional = bankroll * 0.25f;
    }
    else if (t <= 5.0f && t > 3.0f)
    {
      if (diff == 2)
        qty_notional = bankroll * 0.50f;
      else if (diff > 2)
        qty_notional = bankroll; // all-in
    }
    else if (t <= 3.0f && t > 2.0f)
    {
      if (diff == 2)
        qty_notional = bankroll * 0.50f;
      else if (diff > 2)
        qty_notional = bankroll; // all-in
    }
    else if (t <= 2.0f)
    {
      if (diff > 0)
        qty_notional = bankroll; // all-in with a lead
    }

    if (qty_notional <= 0.0f)
      return;

    if (qty_notional > bankroll)
      qty_notional = bankroll;

    float ref_price = 50.0f;
    {
      std::lock_guard<std::mutex> g(mu_);
      if (side == Side::buy)
      {
        if (!local_asks.empty())
          ref_price = local_asks.front().first;
      }
      else // Side::sell
      {
        if (!local_bids.empty())
          ref_price = local_bids.front().first;
      }
    }

    // Avoid div-by-zero or nonsense prices.
    if (!(ref_price > 0.0f && std::isfinite(ref_price)))
      ref_price = 50.0f;

    float qty_units = qty_notional / ref_price;

    println("Trade: t=" + std::to_string(t) + " diff=" + std::to_string(diff) + " qty=" + std::to_string(qty_units) + " price=" + std::to_string(ref_price));

    place_market_order(side, Ticker::TEAM_A, qty_units);
    bankroll -= qty_notional;
  }
};