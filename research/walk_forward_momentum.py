from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np




@dataclass(frozen=True)
class WalkForwardConfig:
    # Walk-forward windows
    train_years: int = 3
    test_months: int = 6
    step_months: int = 6

    # Portfolio construction
    positions: int = 12
    universe_top_n: int = 800  # top liquid stocks by $ volume
    rebalance_weekday: int = 0  # 0=Mon ... 4=Fri (weekly rebalance anchor)
    rebalance_interval_weeks: int = 2  # 1 = weekly (default), 2 = biweekly, etc.
    min_rebalance_weight_change: float = 0.0
    starting_cash: float = 100_000.0
    stability_lookback_periods: int = 1  # 1 = no filter; N = stock must appear in top positions*2 for last N rebalances

    # Liquidity / lookbacks (trading days)
    liq_lookback: int = 60

    # Momentum horizons (trading days)
    mom_3m: int = 63
    mom_6m: int = 126
    mom_12m: int = 252

    # Momentum weights: recent winners lead, persistence confirms
    w_3m: float = 0.60
    w_6m: float = 0.30
    w_12m: float = 0.10
    use_strength_filter: bool = False  # if True, only allow momentum_score > 0
    percentile_filter_enabled: bool = False  # if True, filter to top momentum percentile
    percentile_threshold: float = 0.80  # keep scores >= this cross-sectional percentile

    # Long-term persistence veto (per-stock)
    veto_if_12m_return_below: float = 0.0  # don’t buy if 12m return < 0

    # Market-level risk-off: SPY (or your market proxy) trend filter
    market_symbol: str = "SPY"
    market_sma_days: int = 200
    risk_on_buffer: float = 0.0  # require market_close > SMA*(1+buffer)

    # Trading frictions (optional)
    cost_bps: float = 10.0      # broker commission / market impact, applied on traded notional
    slippage_bps: float = 5.0  # execution slippage: buys priced higher, sells priced lower

    # Risk-off partial exposure: 0.0 = go fully to cash (default), 1.0 = stay fully invested
    risk_off_exposure: float = 0.3  # fraction of portfolio to keep in selected stocks when risk-off

    # Dynamic exposure scaling: exposure ramps smoothly with market distance above SMA
    min_exposure: float = 0.0    # exposure when risk-off (floor); 0.0 = go fully to cash
    max_exposure: float = 1.0    # maximum exposure; reached when risk-on and slope=0 (step function)
    exposure_slope: float = 0.0  # >0 ramps from min to max as market moves above SMA; 0 = step function

    # Volatility-adjusted momentum
    use_vol_adjusted_momentum: bool = False  # divide momentum score by trailing daily vol
    vol_lookback: int = 63  # trading days used to estimate trailing volatility

    # SMA slope filter: require the SMA to be rising (not merely price above SMA)
    require_positive_sma_slope: bool = False
    sma_slope_lookback: int = 20  # trading days over which to measure SMA direction

    # Market anti-whipsaw gate
    market_filter_mode: str = "none"  # "none" | "skip_choppy_rebalance" | "choppy_filter_reduce_exposure" | "momentum_effectiveness_skip"
    choppy_vol_lookback: int = 126    # rolling window for vol-regime median baseline
    choppy_reduce_exposure: float = 0.70  # exposure fraction during choppy regime (used by choppy_filter_reduce_exposure mode)
    momentum_effectiveness_lookback: int = 63  # bars used to evaluate whether momentum has recently worked
    momentum_effectiveness_skip_threshold: Optional[float] = 0.0  # skip when effectiveness < threshold; None disables effectiveness skip




def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lower().strip() for c in out.columns]
    return out


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out


def _daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().fillna(0.0)


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _month_delta(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    # Safe month shift using pandas offsets
    return (ts + pd.DateOffset(months=months)).normalize()


def _year_delta(ts: pd.Timestamp, years: int) -> pd.Timestamp:
    return (ts + pd.DateOffset(years=years)).normalize()


def _week_rebalance_dates(dates: pd.DatetimeIndex, weekday: int) -> pd.DatetimeIndex:
    # Use trading dates that fall on the chosen weekday; if no exact weekday that week,
    # we’ll rebalance on the first trading day AFTER the weekday by resampling.
    # Simpler: choose all dates where date.weekday == weekday.
    sel = [d for d in dates if d.weekday() == weekday]
    return pd.DatetimeIndex(sel)


def _clip_to_index(
    df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    return df.loc[(df.index >= start) & (df.index <= end)].copy()


def _get_close(df: pd.DataFrame) -> pd.Series:
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")
    return df["close"]


def _get_volume(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        raise ValueError("DataFrame must contain 'volume' column")
    return df["volume"]




def market_risk_on(
    market_df: pd.DataFrame, asof: pd.Timestamp, cfg: WalkForwardConfig
) -> bool:
    """
    Market-level risk-off: risk-on if market close > SMA(market_sma_days)*(1+buffer).
    Optionally also requires the SMA to be rising over sma_slope_lookback days.
    Evaluated using data up to 'asof' only.
    """
    m = _clip_to_index(market_df, market_df.index.min(), asof)
    close = _get_close(m)
    sma = _sma(close, cfg.market_sma_days)
    if len(sma) == 0 or pd.isna(sma.iloc[-1]):
        return False  # not enough history -> conservative

    risk_on_price = close.iloc[-1] > sma.iloc[-1] * (1.0 + cfg.risk_on_buffer)

    if not cfg.require_positive_sma_slope:
        return risk_on_price

    sma_valid = sma.dropna()
    if len(sma_valid) <= cfg.sma_slope_lookback:
        return False  # not enough SMA history for slope check -> conservative

    current_sma = sma_valid.iloc[-1]
    prior_sma = sma_valid.iloc[-1 - cfg.sma_slope_lookback]
    risk_on_slope = current_sma > prior_sma

    return risk_on_price and risk_on_slope


def compute_market_exposure(
    market_df: pd.DataFrame, asof: pd.Timestamp, cfg: WalkForwardConfig, risk_on: bool
) -> float:
    """
    Compute portfolio exposure fraction given the current regime.

    When risk-off (risk_on=False): returns cfg.min_exposure (the floor).
    When risk-on and exposure_slope == 0: returns cfg.max_exposure (binary step function).
    When risk-on and exposure_slope > 0: ramps linearly from min to max based on
        distance = (close / sma) - 1.0, clamped to [min_exposure, max_exposure].
    When min_exposure == max_exposure: always returns that constant regardless of regime.
    """
    if cfg.min_exposure == cfg.max_exposure:
        return cfg.min_exposure
    if not risk_on:
        return cfg.min_exposure
    # Risk-on path
    if cfg.exposure_slope == 0.0:
        # Step function: jump straight to max_exposure when risk-on
        return cfg.max_exposure
    m = _clip_to_index(market_df, market_df.index.min(), asof)
    close = _get_close(m)
    sma = _sma(close, cfg.market_sma_days)
    if len(sma) == 0 or pd.isna(sma.iloc[-1]) or sma.iloc[-1] == 0:
        return cfg.min_exposure  # not enough history -> conservative
    distance = float(close.iloc[-1]) / float(sma.iloc[-1]) - 1.0
    raw_exposure = cfg.min_exposure + cfg.exposure_slope * distance
    return float(min(cfg.max_exposure, max(cfg.min_exposure, raw_exposure)))


def _is_choppy_market(
    market_df: pd.DataFrame, asof: pd.Timestamp, cfg: WalkForwardConfig
) -> bool:
    """
    Returns True when the market is choppy / low-quality on the given date.
    Uses only data up to asof — no forward-looking leakage.

    All three conditions must hold:
    1. SPY price is within 2% of its 20d SMA  (price hugging trend, little directional conviction).
    2. SMA20 / SMA50 are within 1.5% of each other  (short and medium trends not separated).
    3. Current 20d realised vol is above its rolling median over the last choppy_vol_lookback days.
    """
    m = _clip_to_index(market_df, market_df.index.min(), asof)
    close = _get_close(m)

    # Need 50 bars for SMA50 plus enough history for the vol baseline.
    if len(close) < 50 + cfg.choppy_vol_lookback + 20:
        return False  # conservative: don't skip when history is short

    sma20 = close.rolling(20, min_periods=20).mean()
    sma50 = close.rolling(50, min_periods=50).mean()

    if pd.isna(sma20.iloc[-1]) or pd.isna(sma50.iloc[-1]):
        return False

    c   = float(close.iloc[-1])
    s20 = float(sma20.iloc[-1])
    s50 = float(sma50.iloc[-1])

    cond1 = abs(c / s20 - 1.0) < 0.02     # price within 2% of SMA20
    cond2 = abs(s20 / s50 - 1.0) < 0.015  # SMA20 within 1.5% of SMA50

    daily_rets   = close.pct_change().dropna()
    rolling_vol  = daily_rets.rolling(20, min_periods=20).std() * np.sqrt(252)
    rv_clean     = rolling_vol.dropna()

    if len(rv_clean) < cfg.choppy_vol_lookback:
        return cond1 and cond2  # vol baseline not yet available; use 2-condition fallback

    current_vol = float(rv_clean.iloc[-1])
    vol_median  = float(rv_clean.iloc[-cfg.choppy_vol_lookback :].median())
    cond3 = current_vol > vol_median

    return cond1 and cond2 and cond3


def dollar_volume_rank(
    symbol_dfs: Dict[str, pd.DataFrame], asof: pd.Timestamp, cfg: WalkForwardConfig
) -> pd.Series:
    """
    Compute trailing avg dollar volume = mean(close*volume) over liq_lookback days ending at asof.
    Returns a Series: symbol -> dollar_volume
    """
    vals = {}
    for sym, df in symbol_dfs.items():
        d = _clip_to_index(df, df.index.min(), asof)
        if len(d) < cfg.liq_lookback:
            continue
        d = d.iloc[-cfg.liq_lookback :]
        dv = (d["close"] * d["volume"]).mean()
        if np.isfinite(dv):
            vals[sym] = dv
    return pd.Series(vals).sort_values(ascending=False)


def momentum_score(
    df: pd.DataFrame, asof: pd.Timestamp, cfg: WalkForwardConfig
) -> Optional[float]:
    """
    Weighted momentum: 3m,6m,12m returns.
    Veto: if 12m return < cfg.veto_if_12m_return_below -> None
    If use_vol_adjusted_momentum: divide raw score by trailing daily return std.
    """
    d = _clip_to_index(df, df.index.min(), asof)
    close = _get_close(d)

    need = max(cfg.mom_12m, cfg.mom_6m, cfg.mom_3m) + 1
    if len(close) < need:
        return None

    c0 = close.iloc[-1]
    r3 = (c0 / close.iloc[-1 - cfg.mom_3m]) - 1.0
    r6 = (c0 / close.iloc[-1 - cfg.mom_6m]) - 1.0
    r12 = (c0 / close.iloc[-1 - cfg.mom_12m]) - 1.0

    if r12 < cfg.veto_if_12m_return_below:
        return None

    raw_score = cfg.w_3m * r3 + cfg.w_6m * r6 + cfg.w_12m * r12

    if not cfg.use_vol_adjusted_momentum:
        return raw_score

    # Volatility adjustment: require enough history for vol lookback
    if len(close) < cfg.vol_lookback + 1:
        return None
    daily_rets = close.iloc[-cfg.vol_lookback - 1 :].pct_change().dropna()
    vol = float(daily_rets.std())
    if vol <= 0 or not np.isfinite(vol):
        return None
    return raw_score / vol


def compute_rebalance_target(
    symbol_dfs: Dict[str, pd.DataFrame],
    market_df: pd.DataFrame,
    asof: pd.Timestamp,
    cfg: WalkForwardConfig,
    ranking_history: Optional[List[set]] = None,
) -> Dict[str, object]:
    """
    Shared target-construction logic for a single rebalance date.
    This mirrors the baseline selection/ranking/sizing path used in run_weekly_portfolio.

    Returns keys:
      - risk_on: bool
      - target_exposure: float
      - eligible_count: int
      - selected_symbols: List[str]
      - target_weights: Dict[str, float]
      - ranking_history: List[set] (updated in-place + returned for convenience)
    """
    if ranking_history is None:
        ranking_history = []

    # Market regime and exposure sizing
    risk_on = market_risk_on(market_df, asof, cfg)
    target_exposure = compute_market_exposure(market_df, asof, cfg, risk_on)

    # Dynamic liquid universe at asof
    dv = dollar_volume_rank(symbol_dfs, asof, cfg)
    universe = list(dv.head(cfg.universe_top_n).index)

    # Momentum scoring
    scored: List[Tuple[str, float]] = []
    for sym in universe:
        sc = momentum_score(symbol_dfs[sym], asof, cfg)
        if sc is not None and np.isfinite(sc):
            scored.append((sym, float(sc)))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Optional filters
    if cfg.use_strength_filter:
        scored = [(s, sc) for s, sc in scored if sc > 0.0]

    if cfg.percentile_filter_enabled and scored:
        threshold = float(np.quantile([sc for _, sc in scored], cfg.percentile_threshold))
        scored = [(s, sc) for s, sc in scored if sc >= threshold]

    eligible_count = len(scored)

    # Stability filter path (same semantics as backtest engine)
    top_pool_size = cfg.positions * 2
    top_pool = {s for s, _ in scored[:top_pool_size]}
    ranking_history.append(top_pool)
    if len(ranking_history) > cfg.stability_lookback_periods:
        ranking_history.pop(0)

    if cfg.stability_lookback_periods > 1 and len(ranking_history) >= cfg.stability_lookback_periods:
        stable = [(s, sc) for s, sc in scored if all(s in h for h in ranking_history)]
    else:
        stable = scored

    # Final picks with fallback fill
    picks = [s for s, _ in stable[: cfg.positions]]
    if len(picks) < cfg.positions:
        picks_set = set(picks)
        for s, _ in scored:
            if len(picks) >= cfg.positions:
                break
            if s not in picks_set:
                picks.append(s)
                picks_set.add(s)

    target_weights = {s: 1.0 / len(picks) for s in picks} if picks else {}

    return {
        "risk_on": bool(risk_on),
        "target_exposure": float(target_exposure),
        "eligible_count": int(eligible_count),
        "selected_symbols": picks,
        "target_weights": target_weights,
        "ranking_history": ranking_history,
    }




@dataclass
class WindowResult:
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    equity_curve: pd.Series
    trades: int
    turnover: float
    turnover_per_rebalance: List[float]  # per-rebalance fractions for distribution analysis
    metrics: Dict[str, float]
    rebalance_records: List[Dict[str, object]] = field(default_factory=list)
    state_snapshots: List[Dict[str, object]] = field(default_factory=list)
    ending_cash: float = 0.0
    ending_holdings: Dict[str, float] = field(default_factory=dict)
    non_zero_exposure_days: int = 0


def run_weekly_portfolio(
    symbol_dfs: Dict[str, pd.DataFrame],
    market_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: WalkForwardConfig,
    initial_cash: Optional[float] = None,
    initial_holdings: Optional[Dict[str, float]] = None,
    snapshot_dates: Optional[set[pd.Timestamp]] = None,
) -> WindowResult:
    """
    Long-only, equal-weighted, rebalance weekly.
    Dynamic universe: top cfg.universe_top_n by trailing $ volume at each rebalance.
    Risk-off: if market is not risk-on, hold cash (0 exposure).
    Transaction costs: cost_bps applied on traded notional (broker fees / market impact).
    Slippage: slippage_bps applied on traded notional (execution at worse prices).
    """
    # Build a unified trading calendar from market (or any symbol) dates
    cal = market_df.index[(market_df.index >= start) & (market_df.index <= end)]
    cal = pd.DatetimeIndex(cal).sort_values()
    if len(cal) == 0:
        raise ValueError("No trading dates in window")

    rebals = _week_rebalance_dates(cal, cfg.rebalance_weekday)
    if cfg.rebalance_interval_weeks > 1:
        rebals = rebals[:: cfg.rebalance_interval_weeks]
    if len(rebals) == 0:
        # fallback: if no exact weekday in range, just use first date
        rebals = pd.DatetimeIndex([cal[0]])

    cash = cfg.starting_cash if initial_cash is None else float(initial_cash)
    holdings: Dict[str, float] = (
        {} if initial_holdings is None else {s: float(sh) for s, sh in initial_holdings.items()}
    )
    equity_curve = []
    eq_index = []

    trades = 0
    total_turnover = 0.0
    turnover_per_rebalance: List[float] = []  # one entry per rebalance event
    total_costs_dollar: float = 0.0           # cumulative dollar cost (commission + slippage)
    rebalance_records: List[Dict[str, object]] = []
    state_snapshots: List[Dict[str, object]] = []
    non_zero_exposure_days = 0

    # Precompute close series access
    closes: Dict[str, pd.Series] = {s: _get_close(df) for s, df in symbol_dfs.items()}
    mclose = _get_close(market_df)

    def portfolio_value(on_date: pd.Timestamp) -> float:
        v = cash
        for sym, sh in holdings.items():
            cs = closes.get(sym)
            if cs is None:
                continue
            px = (
                cs.loc[:on_date].iloc[-1]
                if (on_date in cs.index or len(cs.loc[:on_date]) > 0)
                else np.nan
            )
            if np.isfinite(px):
                v += sh * float(px)
        return float(v)

    def liquid_universe(asof: pd.Timestamp) -> List[str]:
        dv = dollar_volume_rank(symbol_dfs, asof, cfg)
        return list(dv.head(cfg.universe_top_n).index)

    def portfolio_state(on_date: pd.Timestamp) -> Tuple[float, float, Dict[str, float]]:
        invested = 0.0
        values: Dict[str, float] = {}
        for sym, sh in holdings.items():
            cs = closes.get(sym)
            if cs is None:
                continue
            hist = cs.loc[:on_date]
            if len(hist) == 0:
                continue
            px = float(hist.iloc[-1])
            if np.isfinite(px):
                val = float(sh) * px
                values[sym] = val
                invested += val
        equity = float(cash + invested)
        return equity, float(invested), values

    def top3_holdings_str(on_date: pd.Timestamp) -> str:
        vals = []
        for sym, sh in holdings.items():
            cs = closes.get(sym)
            if cs is None:
                continue
            hist = cs.loc[:on_date]
            if len(hist) == 0:
                continue
            px = float(hist.iloc[-1])
            if np.isfinite(px):
                vals.append((sym, abs(float(sh) * px), float(sh)))
        vals.sort(key=lambda x: x[1], reverse=True)
        return "|".join([f"{sym}:{sh:.6f}" for sym, _, sh in vals[:3]])

    def holdings_signature_str() -> str:
        if not holdings:
            return ""
        items = sorted((sym, float(sh)) for sym, sh in holdings.items())
        return "|".join(f"{sym}:{sh:.8f}" for sym, sh in items)

    def momentum_effectiveness_signal(asof: pd.Timestamp) -> Optional[float]:
        """
        Estimate recent momentum effectiveness using a trailing lookback window.
        At rebalance date t:
        - Form top-N by momentum at signal_date = t - lookback bars
        - Measure realized return from signal_date to t
        - Effectiveness = avg(top-N returns) - avg(universe returns)
        """
        if asof not in cal:
            return None

        lookback = int(cfg.momentum_effectiveness_lookback)
        if lookback <= 0:
            return None

        pos = int(cal.get_loc(asof))
        if pos < lookback:
            return None

        signal_date = cal[pos - lookback]

        uni = liquid_universe(signal_date)
        if not uni:
            return None

        scored = []
        for sym in uni:
            sc = momentum_score(symbol_dfs[sym], signal_date, cfg)
            if sc is not None and np.isfinite(sc):
                scored.append((sym, sc))

        if not scored:
            return None

        scored.sort(key=lambda x: x[1], reverse=True)
        top_syms = [s for s, _ in scored[: cfg.positions]]

        def _realized_return(sym: str) -> Optional[float]:
            cs = closes.get(sym)
            if cs is None:
                return None
            hs = cs.loc[:signal_date]
            he = cs.loc[:asof]
            if len(hs) == 0 or len(he) == 0:
                return None
            p0 = float(hs.iloc[-1])
            p1 = float(he.iloc[-1])
            if not np.isfinite(p0) or not np.isfinite(p1) or p0 <= 0:
                return None
            return p1 / p0 - 1.0

        top_rets = []
        uni_rets = []
        for sym, _ in scored:
            r = _realized_return(sym)
            if r is None:
                continue
            uni_rets.append(r)
            if sym in top_syms:
                top_rets.append(r)

        if not top_rets or not uni_rets:
            return None

        return float(np.mean(top_rets) - np.mean(uni_rets))

    # Rolling history of top-ranked symbol sets for stability filter
    # Each entry is a set of symbols that were in the top positions*2 at that rebalance
    ranking_history: List[set] = []
    stability_window = cfg.stability_lookback_periods  # alias for clarity

    for i, dt in enumerate(cal):
        # Choppy-market skip gate — evaluated before any rebalance execution.
        _is_rebalance = dt in rebals
        # _is_choppy_on_this_date: computed once per rebalance date; shared by both skip and
        # reduce-exposure modes to avoid calling _is_choppy_market twice.
        _is_choppy_on_this_date = (
            _is_rebalance
            and cfg.market_filter_mode in ("skip_choppy_rebalance", "choppy_filter_reduce_exposure")
            and _is_choppy_market(market_df, dt, cfg)
        )
        _momentum_effectiveness = None
        _is_mom_effectiveness_skip = (
            _is_rebalance and cfg.market_filter_mode == "momentum_effectiveness_skip"
        )
        if _is_mom_effectiveness_skip:
            _momentum_effectiveness = momentum_effectiveness_signal(dt)

        _me_threshold = cfg.momentum_effectiveness_skip_threshold
        _skip_by_momentum_effectiveness = (
            _is_mom_effectiveness_skip
            and _me_threshold is not None
            and _momentum_effectiveness is not None
            and np.isfinite(_momentum_effectiveness)
            and float(_momentum_effectiveness) < float(_me_threshold)
        )

        _skip_this = (
            (_is_choppy_on_this_date and cfg.market_filter_mode == "skip_choppy_rebalance")
            or _skip_by_momentum_effectiveness
        )

        _skip_reason = ""
        if _skip_this:
            if cfg.market_filter_mode == "skip_choppy_rebalance" and _is_choppy_on_this_date:
                _skip_reason = "choppy_market"
            elif cfg.market_filter_mode == "momentum_effectiveness_skip":
                _skip_reason = "momentum_effectiveness"

        if _skip_this:
            _pv_s = portfolio_value(dt)
            _holdings_sig_before = holdings_signature_str()
            _holdings_count_before = int(len(holdings))
            _cash_before = float(cash)
            rebalance_records.append(
                {
                    "rebalance_date": dt,
                    "skipped": True,
                    "choppy_override": False,
                    "skip_reason": _skip_reason,
                    "momentum_effectiveness": (
                        float(_momentum_effectiveness)
                        if _momentum_effectiveness is not None else np.nan
                    ),
                    "risk_on": bool(market_risk_on(market_df, dt, cfg)),
                    "target_exposure": 0.0,
                    "eligible_count": 0,
                    "selected_symbols": "",
                    "holdings_count_before": _holdings_count_before,
                    "holdings_count_after": _holdings_count_before,
                    "holdings_signature_before": _holdings_sig_before,
                    "holdings_signature_after": _holdings_sig_before,
                    "cash_before": _cash_before,
                    "cash_after": _cash_before,
                    "turnover": 0.0,
                    "estimated_cost": 0.0,
                    "estimated_slippage": 0.0,
                    "equity_before_rebalance": float(_pv_s),
                    "equity_after_rebalance": float(_pv_s),
                }
            )
        # Execute rebalance if it is scheduled AND not blocked by the market-quality gate.
        if _is_rebalance and not _skip_this:
            # Mark-to-market before rebalance
            pv = portfolio_value(dt)
            risk_on = market_risk_on(market_df, dt, cfg)
            target_portfolio_fraction = compute_market_exposure(market_df, dt, cfg, risk_on)
            # Choppy-regime exposure scaling: override target fraction downward when choppy.
            _choppy_override = (
                cfg.market_filter_mode == "choppy_filter_reduce_exposure"
                and _is_choppy_on_this_date
            )
            if _choppy_override:
                target_portfolio_fraction = min(target_portfolio_fraction, cfg.choppy_reduce_exposure)
            picks: List[str] = []
            eligible_count = 0
            traded_notional = 0.0
            cost = 0.0
            slippage_cost = 0.0

            if not risk_on and cfg.min_exposure == 0.0:
                # Original behavior: go fully to cash
                if holdings:
                    held_value = pv - cash  # dollar value of all open positions
                    traded_notional = abs(held_value)
                    if pv > 0:
                        to = abs(held_value) / pv
                        total_turnover += to
                        turnover_per_rebalance.append(to)
                    # Slippage on forced liquidation: selling receives less
                    slip_cost = abs(held_value) * (cfg.slippage_bps / 10_000.0)
                    slippage_cost = slip_cost
                    total_costs_dollar += slip_cost
                    trades += len(holdings)
                    cash = pv - slip_cost
                else:
                    # Already in cash — no trades, record 0-turnover rebalance
                    turnover_per_rebalance.append(0.0)
                    cash = pv
                holdings = {}
            else:
                # Partial-exposure risk-off OR fully risk-on path.
                # Dynamic exposure: step function (slope=0) or smooth ramp (slope>0) above SMA.
                # Risk-off with min_exposure>0 uses min_exposure as the floor.
                # Build tradable set
                uni = liquid_universe(dt)

                # Score momentum for universe
                scored = []
                for sym in uni:
                    sc = momentum_score(symbol_dfs[sym], dt, cfg)
                    if sc is not None and np.isfinite(sc):
                        scored.append((sym, sc))

                scored.sort(key=lambda x: x[1], reverse=True)

                # Optional strength filter: keep only names with genuinely positive momentum.
                if cfg.use_strength_filter:
                    scored = [(s, sc) for s, sc in scored if sc > 0.0]

                # Optional percentile filter: keep scores in the top cross-sectional percentile.
                if cfg.percentile_filter_enabled and scored:
                    threshold = float(np.quantile([sc for _, sc in scored], cfg.percentile_threshold))
                    scored = [(s, sc) for s, sc in scored if sc >= threshold]

                eligible_count = len(scored)

                # Build top-2x candidate set and record for stability history
                top_pool_size = cfg.positions * 2
                top_pool = {s for s, _ in scored[:top_pool_size]}
                ranking_history.append(top_pool)
                if len(ranking_history) > stability_window:
                    ranking_history.pop(0)

                # Stability filter: keep only stocks that appeared in ALL recent top pools
                if stability_window > 1 and len(ranking_history) >= stability_window:
                    stable = [
                        (s, sc) for s, sc in scored
                        if all(s in h for h in ranking_history)
                    ]
                else:
                    stable = scored

                # Select top positions from stable candidates; fall back to full ranking if needed
                picks = [s for s, _ in stable[: cfg.positions]]
                if len(picks) < cfg.positions:
                    # Fallback: fill remaining slots from full ranked list (excluding already picked)
                    picks_set = set(picks)
                    for s, _ in scored:
                        if len(picks) >= cfg.positions:
                            break
                        if s not in picks_set:
                            picks.append(s)
                            picks_set.add(s)

                # Target equal weight among picks
                target_weights = {s: 1.0 / len(picks) for s in picks} if picks else {}

                # Compute current weights
                pv = portfolio_value(dt)
                current_values = {}
                for sym, sh in holdings.items():
                    px = closes[sym].loc[:dt].iloc[-1]
                    current_values[sym] = sh * float(px)

                # Determine trades to reach targets
                # We’ll do "full rebalance": sell names not in picks, resize those in picks
                target_values = {s: pv * target_portfolio_fraction * w for s, w in target_weights.items()}

                # Sell removed
                for sym in list(holdings.keys()):
                    if sym not in target_values:
                        traded_notional += abs(current_values.get(sym, 0.0))
                        trades += 1
                        del holdings[sym]

                # Resize/add picks
                for sym, tval in target_values.items():
                    # price as of dt
                    cs = closes.get(sym)
                    if cs is None:
                        continue
                    px = cs.loc[:dt].iloc[-1]
                    if not np.isfinite(px) or px <= 0:
                        continue

                    cur_val = current_values.get(sym, 0.0)
                    diff = tval - cur_val

                    # Apply threshold only to resize trades for symbols already held.
                    # New entries should always be allowed; full exits are handled above.
                    is_resize_trade = sym in current_values
                    if is_resize_trade:
                        current_weight = (cur_val / pv) if pv > 0 else 0.0
                        target_weight = target_weights.get(sym, 0.0)
                        weight_diff = abs(target_weight - current_weight)
                        if weight_diff < cfg.min_rebalance_weight_change:
                            continue

                    if abs(diff) / max(pv, 1e-9) < 1e-4:
                        continue

                    # trade shares
                    dsh = diff / float(px)
                    holdings[sym] = holdings.get(sym, 0.0) + dsh
                    traded_notional += abs(diff)
                    trades += 1

                # Transaction cost: broker commission / market impact
                cost = traded_notional * (cfg.cost_bps / 10_000.0)
                # Slippage: execution at worse prices for both buys and sells
                slippage_cost = traded_notional * (cfg.slippage_bps / 10_000.0)
                total_costs_dollar += cost + slippage_cost

                cash = (
                    pv
                    - sum(
                        holdings[s] * float(closes[s].loc[:dt].iloc[-1])
                        for s in holdings
                    )
                    - cost
                    - slippage_cost
                )

                # Per-rebalance turnover = traded notional / portfolio value
                rebal_to = traded_notional / pv if pv > 0 else 0.0
                turnover_per_rebalance.append(rebal_to)
                total_turnover += rebal_to

            # Record rebalance-level diagnostics for reconciliation/debugging.
            pv_after = portfolio_value(dt)
            rebal_to = traded_notional / pv if pv > 0 else 0.0
            rebalance_records.append(
                {
                    "rebalance_date": dt,
                    "choppy_override": bool(_choppy_override),
                    "skip_reason": "",
                    "momentum_effectiveness": (
                        float(_momentum_effectiveness)
                        if _momentum_effectiveness is not None else np.nan
                    ),
                    "risk_on": bool(risk_on),
                    "target_exposure": float(target_portfolio_fraction),
                    "eligible_count": int(eligible_count),
                    "selected_symbols": "|".join(picks),
                    "turnover": float(rebal_to),
                    "estimated_cost": float(cost),
                    "estimated_slippage": float(slippage_cost),
                    "equity_before_rebalance": float(pv),
                    "equity_after_rebalance": float(pv_after),
                }
            )

        # Record daily equity
        eq_s, invested_s, _ = portfolio_state(dt)
        equity_curve.append(eq_s)
        eq_index.append(dt)

        if eq_s > 0 and invested_s / eq_s > 1e-9:
            non_zero_exposure_days += 1

        if snapshot_dates is not None and dt in snapshot_dates:
            eq_s, invested_s, _ = portfolio_state(dt)
            state_snapshots.append(
                {
                    "date": dt,
                    "equity": float(eq_s),
                    "cash": float(cash),
                    "invested_value": float(invested_s),
                    "holdings_count": int(len(holdings)),
                    "top3_holdings": top3_holdings_str(dt),
                }
            )

    eq = pd.Series(equity_curve, index=pd.DatetimeIndex(eq_index), name="equity")

    # Metrics
    rets = eq.pct_change().dropna()
    ann_factor = 252.0
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (ann_factor / max(len(eq) - 1, 1)) - 1.0
    vol = rets.std() * np.sqrt(ann_factor) if len(rets) > 1 else 0.0
    sharpe = (
        (rets.mean() * ann_factor) / (rets.std() * np.sqrt(ann_factor))
        if rets.std() > 0
        else 0.0
    )
    dd = (eq / eq.cummax() - 1.0).min()

    # Turnover distribution across all rebalance events
    to_arr = np.array(turnover_per_rebalance) if turnover_per_rebalance else np.array([0.0])

    metrics = {
        "final_equity": float(eq.iloc[-1]),
        "cagr": float(cagr),
        "vol": float(vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd),
        "avg_turnover": float(to_arr.mean()),        # average turnover per rebalance
        "median_turnover": float(np.median(to_arr)), # median (less sensitive to outliers)
        "max_turnover": float(to_arr.max()),         # worst single rebalance
        "total_costs": total_costs_dollar,           # combined cost drag in dollars
    }

    return WindowResult(
        window_start=start,
        window_end=end,
        equity_curve=eq,
        trades=trades,
        turnover=float(total_turnover),
        turnover_per_rebalance=turnover_per_rebalance,
        metrics=metrics,
        rebalance_records=rebalance_records,
        state_snapshots=state_snapshots,
        ending_cash=float(cash),
        ending_holdings={s: float(sh) for s, sh in holdings.items()},
        non_zero_exposure_days=int(non_zero_exposure_days),
    )




def walk_forward_validate(
    symbol_dfs: Dict[str, pd.DataFrame],
    market_df: pd.DataFrame,
    cfg: WalkForwardConfig,
    return_debug: bool = False,
) -> Tuple[pd.DataFrame, pd.Series] | Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    """
    Returns:
      - results_df: per-window metrics table
      - combined_equity: stitched out-of-sample equity curve (test windows concatenated)
    """
    # Determine global date range intersection
    market_df = _ensure_datetime_index(_normalize_cols(market_df))
    market_dates = market_df.index

    # Require symbols normalized
    symbol_dfs = {
        s: _ensure_datetime_index(_normalize_cols(df)) for s, df in symbol_dfs.items()
    }

    # Start after enough history for liquidity + momentum + market SMA
    min_need = max(cfg.liq_lookback, cfg.mom_12m + 1, cfg.market_sma_days + 1)
    start_candidates = [
        df.index[min_need] for df in symbol_dfs.values() if len(df.index) > min_need
    ]
    if not start_candidates:
        raise ValueError("Not enough history across symbols for configured lookbacks.")
    global_start = max(max(start_candidates), market_dates[min_need])
    global_end = market_dates.max()

    # Build walk-forward windows
    windows = []
    cursor = global_start

    # Cursor should start at a point where a full train window exists before first test window
    first_test_start = _year_delta(cursor, cfg.train_years)
    if first_test_start > global_end:
        raise ValueError("Not enough history for first test window.")
    cursor = first_test_start

    while True:
        test_start = cursor
        test_end = _month_delta(test_start, cfg.test_months) - pd.Timedelta(days=1)
        if test_start >= global_end:
            break
        test_end = min(test_end, global_end)
        windows.append((test_start, test_end))
        cursor = _month_delta(cursor, cfg.step_months)
        if cursor > global_end:
            break

    # Run each test window as pure OOS (we don't optimize anything in train)
    # Starting equity resets each window? For strict walk-forward, you usually *carry* equity forward.
    # We'll CARRY forward by feeding ending equity into next window.
    results = []
    combined = []
    combined_idx = []
    all_rebalance_dates: List[pd.Timestamp] = []
    total_selected_symbols_across_rebalances = 0
    total_non_zero_exposure_days = 0

    cash = cfg.starting_cash
    holdings: Dict[str, float] = {}
    cfg_local = cfg

    for ws, we in windows:
        # Carry full portfolio state across windows (cash + holdings),
        # not just ending equity as all-cash.
        cfg_local = WalkForwardConfig(**{**cfg.__dict__, "starting_cash": cash})
        res = run_weekly_portfolio(
            symbol_dfs,
            market_df,
            ws,
            we,
            cfg_local,
            initial_cash=cash,
            initial_holdings=holdings,
        )

        row = {
            "window_start": ws,
            "window_end": we,
            "trades": res.trades,
            "turnover": res.turnover,
            **res.metrics,
        }
        results.append(row)

        all_rebalance_dates.extend(
            [pd.Timestamp(r["rebalance_date"]) for r in res.rebalance_records]
        )
        total_selected_symbols_across_rebalances += sum(
            len(str(r.get("selected_symbols", "")).split("|"))
            for r in res.rebalance_records
            if str(r.get("selected_symbols", "")).strip() != ""
        )
        total_non_zero_exposure_days += int(res.non_zero_exposure_days)

        # stitch equity curve
        eq = res.equity_curve
        if combined:
            # avoid duplicate date at boundary
            eq = eq[eq.index > combined_idx[-1]]
        combined.extend(eq.values.tolist())
        combined_idx.extend(eq.index.tolist())

        cash = float(res.ending_cash)
        holdings = {s: float(sh) for s, sh in res.ending_holdings.items()}

    results_df = pd.DataFrame(results)
    combined_equity = pd.Series(
        combined, index=pd.DatetimeIndex(combined_idx), name="equity_oos"
    )

    if not return_debug:
        return results_df, combined_equity

    unique_rebals = sorted(set(all_rebalance_dates))
    debug_info: Dict[str, object] = {
        "first_rebalance_date": unique_rebals[0] if unique_rebals else None,
        "last_rebalance_date": unique_rebals[-1] if unique_rebals else None,
        "num_rebalances": len(unique_rebals),
        "total_trades": int(results_df["trades"].sum()) if "trades" in results_df.columns else 0,
        "total_selected_symbols_across_rebalances": int(total_selected_symbols_across_rebalances),
        "non_zero_exposure_days": int(total_non_zero_exposure_days),
    }

    return results_df, combined_equity, debug_info




def run_sensitivity(
    symbol_dfs: Dict[str, pd.DataFrame],
    market_df: pd.DataFrame,
    base_cfg: WalkForwardConfig,
    cost_bps_list: Optional[List[float]] = None,
    slippage_bps_list: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Sweep cost and slippage assumptions to answer: does performance collapse
    under realistic friction? Runs full walk-forward for each combination.

    Returns a DataFrame sorted by CAGR descending so degradation is visible at a glance.
    """
    if cost_bps_list is None:
        cost_bps_list = [0.0, 5.0, 10.0]
    if slippage_bps_list is None:
        slippage_bps_list = [0.0, 2.0, 5.0]

    rows = []
    for cost in cost_bps_list:
        for slip in slippage_bps_list:
            cfg = WalkForwardConfig(
                **{**base_cfg.__dict__, "cost_bps": cost, "slippage_bps": slip}
            )
            results_df, combined_equity = walk_forward_validate(symbol_dfs, market_df, cfg)

            rets = combined_equity.pct_change().dropna()
            years = len(combined_equity) / 252.0
            cagr = (
                (combined_equity.iloc[-1] / combined_equity.iloc[0])
                ** (1.0 / max(years, 1e-9))
                - 1.0
            )
            sharpe = (
                rets.mean() / rets.std() * np.sqrt(252.0) if rets.std() > 0 else 0.0
            )
            dd = (combined_equity / combined_equity.cummax() - 1.0).min()

            total_costs = (
                results_df["total_costs"].sum()
                if "total_costs" in results_df.columns
                else float("nan")
            )
            avg_turnover = (
                results_df["avg_turnover"].mean()
                if "avg_turnover" in results_df.columns
                else float("nan")
            )

            rows.append(
                {
                    "cost_bps": cost,
                    "slippage_bps": slip,
                    "total_friction_bps": cost + slip,
                    "cagr_pct": round(cagr * 100, 2),
                    "sharpe": round(sharpe, 3),
                    "max_drawdown_pct": round(dd * 100, 2),
                    "avg_turnover_pct": (
                        round(avg_turnover * 100, 2)
                        if not np.isnan(avg_turnover)
                        else float("nan")
                    ),
                    "total_costs": (
                        round(total_costs, 2) if not np.isnan(total_costs) else float("nan")
                    ),
                    "final_equity": round(float(combined_equity.iloc[-1]), 2),
                }
            )

    return pd.DataFrame(rows).sort_values("cagr_pct", ascending=False).reset_index(drop=True)




def print_summary(results_df: pd.DataFrame, combined_equity: pd.Series) -> None:
    """
    Print a clean OOS performance summary. Answers:
      1. How much do costs reduce returns?  (cost_drag)
      2. Is turnover too high?              (avg_turnover per rebalance)
      3. Does performance collapse under realistic assumptions?  (per-window table)
    """
    rets = combined_equity.pct_change().dropna()
    years = len(combined_equity) / 252.0
    cagr = (
        (combined_equity.iloc[-1] / combined_equity.iloc[0]) ** (1.0 / max(years, 1e-9))
        - 1.0
    )
    sharpe = rets.mean() / rets.std() * np.sqrt(252.0) if rets.std() > 0 else 0.0
    dd = (combined_equity / combined_equity.cummax() - 1.0).min()

    initial = combined_equity.iloc[0]
    final = combined_equity.iloc[-1]
    gross_gain = final - initial

    total_costs = (
        results_df["total_costs"].sum() if "total_costs" in results_df.columns else None
    )
    avg_turnover = (
        results_df["avg_turnover"].mean() if "avg_turnover" in results_df.columns else None
    )

    print("=== WITH COSTS ===")
    print(f"  final:        ${final:>12,.2f}")
    print(f"  cagr:         {cagr * 100:>8.2f}%")
    print(f"  sharpe:       {sharpe:>8.2f}")
    print(f"  max_dd:       {dd * 100:>8.2f}%")
    if avg_turnover is not None:
        print(f"  avg_turnover: {avg_turnover * 100:>8.2f}%  per rebalance")
    if total_costs is not None:
        print(f"  total_costs:  ${total_costs:>12,.2f}")
        if gross_gain > 0:
            # Cost drag = fraction of gross profit consumed by friction
            print(f"  cost_drag:    {total_costs / gross_gain * 100:>7.1f}%  of gross gain")
    print()
    print("=== PER-WINDOW BREAKDOWN ===")
    cols = [
        "window_start", "window_end", "cagr", "sharpe",
        "max_drawdown", "avg_turnover", "total_costs",
    ]
    show = [c for c in cols if c in results_df.columns]
    df = results_df[show].copy()
    fmt = {
        "cagr": lambda x: f"{x * 100:.1f}%",
        "sharpe": lambda x: f"{x:.2f}",
        "max_drawdown": lambda x: f"{x * 100:.1f}%",
        "avg_turnover": lambda x: f"{x * 100:.1f}%",
        "total_costs": lambda x: f"${x:,.0f}",
    }
    for col, fn in fmt.items():
        if col in df.columns:
            df[col] = df[col].map(fn)
    print(df.to_string(index=False))
