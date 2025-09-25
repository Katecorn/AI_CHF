from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.covariance import LedoitWolf

# for dayly dinamic
TRADING_DAYS = 252

# ---------- base utils ----------

def _sanitize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    
    px = prices.sort_index()
    if px.index.has_duplicates:
        px = px.groupby(level=0).last()
    px = px.replace([np.inf, -np.inf], np.nan).ffill()
    return px

def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    px = _sanitize_prices(prices)
    return px.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

# ---------- target weights ----------

def equal_weight_targets(prices: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    
    px = _sanitize_prices(prices)
    sched = px.resample(freq).last().index
    cols = px.columns
    tw_sched = pd.DataFrame(1.0/len(cols), index=sched, columns=cols)
    tw = tw_sched.reindex(px.index).ffill()
    return tw

def mv_targets(prices: pd.DataFrame,
               lookback: int = 60,
               max_w: float = 0.4,
               freq: str = "M",
               allow_short: bool = False) -> pd.DataFrame:

    px = _sanitize_prices(prices)
    rets = px.pct_change().dropna(how="all")
    sched = px.resample(freq).last().index
    weights_list = []
    for t in sched:
        end = rets.index.searchsorted(t, side="right")
        start = max(0, end - lookback)
        win = rets.iloc[start:end]
        if len(win) < max(10, lookback // 3):
            
            w = pd.Series(1.0/len(px.columns), index=px.columns)
        else:
            mu = win.mean().replace([np.inf, -np.inf], np.nan).fillna(0.0)  
            cov = win.cov().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            try:
                inv_cov = pd.DataFrame(np.linalg.pinv(cov.values), index=cov.index, columns=cov.columns)
            except Exception:
                inv_cov = pd.DataFrame(np.eye(len(cov)), index=cov.index, columns=cov.columns)
            raw = inv_cov.dot(mu)
            
            w = raw.clip(lower=(-1 if allow_short else 0))
            if not allow_short:
                w = w.clip(0, max_w)
            if w.abs().sum() == 0:
                w = pd.Series(1.0/len(px.columns), index=px.columns)
            else:
                w = w / w.abs().sum() if allow_short else w / w.sum()
        weights_list.append(w.reindex(px.columns).fillna(0.0).rename(t))
    tw_sched = pd.DataFrame(weights_list)
    tw = tw_sched.reindex(px.index).ffill()
    return tw

# ---------- Adaptive lines for rebalancing  ----------

def bands_from_vol(prices: pd.DataFrame,
                   span: int = 20,
                   base_bps: float = 40.0,
                   k_vol: float = 150.0,
                   floor_bps: float = 10.0,
                   cap_bps: float = 200.0) -> pd.Series:
    """
    band = base + k * EWMA(avg asset vol). С разумными границами.
    """
    px = _sanitize_prices(prices)
    r  = px.pct_change()
    vol = r.ewm(span=span, adjust=False).std().mean(axis=1)
    band_bps = base_bps + k_vol * vol.fillna(0.0)
    band_bps = band_bps.clip(lower=floor_bps, upper=cap_bps)
    return band_bps.rename("band_bps")


# ---------- Main part for dynamic rebalancing ----------

def portfolio_equity_dynamic_v3(
    prices: pd.DataFrame,
    target_weights_ts: pd.DataFrame,
    band_bps_ts: pd.Series | float = 0.0,
    fee_bps: float = 5.0,
    max_turnover: float | None = None,
    rebalance_calendar: str | None = None,
    rebalance_before_return: bool = True,

    min_trade_bps: float = 0.0,                 
    per_asset_band_bps: float | pd.Series | None = None,
    return_details: bool = False                
) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
    px  = _sanitize_prices(prices)
    idx = px.index
    cols = px.columns

    tw = (target_weights_ts.reindex(idx).ffill()
          .reindex(columns=cols).fillna(0.0))


    band_ts = pd.Series(float(band_bps_ts), index=idx) if isinstance(band_bps_ts, (int, float)) \
              else pd.Series(band_bps_ts).reindex(idx).ffill().fillna(0.0)


    if per_asset_band_bps is None:
        pab = pd.Series(0.0, index=cols)
    elif isinstance(per_asset_band_bps, (int, float)):
        pab = pd.Series(float(per_asset_band_bps), index=cols)
    else:
        pab = pd.Series(per_asset_band_bps).reindex(cols).fillna(0.0)

  
    if rebalance_calendar is None:
        allow = pd.Series(True, index=idx, dtype=bool)
    else:
        sched = px.resample(rebalance_calendar).last().index
        allow = pd.Series(False, index=idx, dtype=bool)
        allow.loc[idx.intersection(sched)] = True


    rets = daily_returns(px)

    eq = pd.Series(index=idx, dtype=float); eq.iloc[0] = 1.0
    w_act = tw.iloc[0].clip(lower=0.0); s0 = float(w_act.sum())
    w_act = w_act / s0 if s0 > 0 else pd.Series(1.0/len(cols), index=cols)


    logs = {
        "eq": [], "port_ret": [], "cost": [], "turnover": [], "dev_L1": [],
        "band_L1": [], "rebalanced": [], "w_sum": []
    }

    def _apply_rebalance(at_eq_idx: int, w_tar: pd.Series, band_now: float):
        nonlocal w_act, eq
        raw_delta = (w_tar - w_act)
        mask_small = raw_delta.abs() <= (pab.values / 1e4)
        raw_delta[mask_small] = 0.0

        desired_turnover = float(np.abs(raw_delta).sum())
        if desired_turnover <= 0:
            return 0.0, 0.0

        
        if (max_turnover is not None) and (desired_turnover > max_turnover):
            scale = max_turnover / desired_turnover
            raw_delta *= scale
            desired_turnover = max_turnover

        
        if min_trade_bps > 0:
        
            if desired_turnover < (min_trade_bps / 1e4):
                return 0.0, 0.0

        
        cost = desired_turnover * (fee_bps / 1e4)
        cost = float(np.clip(cost, 0.0, 0.5))
        eq.iloc[at_eq_idx] *= (1 - cost)

        
        w_act = (w_act + raw_delta).clip(lower=0.0)
        s2 = float(w_act.sum())
        if s2 > 0:
            w_act /= s2
        return desired_turnover, cost

    for t in range(1, len(idx)):
        r = rets.iloc[t]
        w_tar = tw.iloc[t].clip(lower=0.0); s = float(w_tar.sum())
        w_tar = w_tar / s if s > 0 else pd.Series(1.0/len(cols), index=cols)

        band = float(band_ts.iloc[t]) / 1e4
        dev_L1 = float(np.abs(w_act - w_tar).sum())
        did_reb, turn_t, cost_t = False, 0.0, 0.0

        if rebalance_before_return:
            if bool(allow.iloc[t]) and (dev_L1 > band):
                turn_t, cost_t = _apply_rebalance(at_eq_idx=t-1, w_tar=w_tar, band_now=band)
                did_reb = turn_t > 0

            port_ret = float((w_act * r).sum())
            eq.iloc[t] = eq.iloc[t-1] * (1 + port_ret)

            growth = (1 + r) * w_act; s3 = float(growth.sum())
            w_act = growth / s3 if s3 > 0 else w_act
        else:
            port_ret = float((w_act * r).sum())
            eq.iloc[t] = eq.iloc[t-1] * (1 + port_ret)
            growth = (1 + r) * w_act; s3 = float(growth.sum())
            w_act = growth / s3 if s3 > 0 else w_act

            if bool(allow.iloc[t]) and (dev_L1 > band):
                turn_t, cost_t = _apply_rebalance(at_eq_idx=t, w_tar=w_tar, band_now=band)
                did_reb = turn_t > 0

        logs["eq"].append(eq.iloc[t])
        logs["port_ret"].append(port_ret)
        logs["cost"].append(cost_t)
        logs["turnover"].append(turn_t)
        logs["dev_L1"].append(dev_L1)
        logs["band_L1"].append(band)
        logs["rebalanced"].append(1 if did_reb else 0)
        logs["w_sum"].append(float(w_act.sum()))

    eq = eq.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    if not return_details:
        return eq

    details = pd.DataFrame({
        "eq": pd.Series(logs["eq"], index=idx[1:]),
        "port_ret": pd.Series(logs["port_ret"], index=idx[1:]),
        "cost": pd.Series(logs["cost"], index=idx[1:]),
        "turnover": pd.Series(logs["turnover"], index=idx[1:]),
        "dev_L1": pd.Series(logs["dev_L1"], index=idx[1:]),
        "band_L1": pd.Series(logs["band_L1"], index=idx[1:]),
        "rebalanced": pd.Series(logs["rebalanced"], index=idx[1:]),
        "w_sum": pd.Series(logs["w_sum"], index=idx[1:])
    })
    return eq, details


def portfolio_monitor(details: pd.DataFrame) -> pd.Series:
    r = details["eq"].pct_change().dropna()
    ann = TRADING_DAYS
    vol = r.std(ddof=1) * np.sqrt(ann) if len(r) > 1 else np.nan
    sharpe = (r.mean() * ann / vol) if (vol and vol > 0) else np.nan
    mdd = float((details["eq"]/details["eq"].cummax() - 1).min())
    years = max(1e-9, len(details)/ann)
    cagr = float(details["eq"].iloc[-1]**(1/years) - 1) if len(details) else np.nan
    turns = details["turnover"].sum()
    avg_turn = details["turnover"].mean()
    n_reb = int(details["rebalanced"].sum())
    avg_dev = details["dev_L1"].mean()
    avg_band = details["band_L1"].mean()
    cost_total = details["cost"].sum()
    return pd.Series({
        "CAGR": cagr, "Sharpe": sharpe, "MaxDD": mdd,
        "TotalTurnover": turns, "AvgDailyTurnover": avg_turn,
        "RebalanceCount": n_reb, "AvgDevL1": avg_dev, "AvgBandL1": avg_band,
        "TotalCost": cost_total
    }).round(4)


# ---------- Reort ----------

def brief_report(eq: pd.Series) -> pd.Series:
    import numpy as np
    import pandas as pd

    if eq is None or len(eq) < 2:
        return pd.Series({"CAGR": np.nan, "Sharpe": np.nan, "MaxDD": np.nan, "FinalEq": np.nan})

    r = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    
    ANN = 24 * 365  

    
    vol = r.std(ddof=1) * np.sqrt(ANN) if len(r) > 1 else np.nan
    mu  = r.mean() * ANN
    sharpe = float(mu / vol) if (isinstance(vol, (int, float)) and vol > 0) else np.nan

    mdd = float((eq / eq.cummax() - 1.0).min())
    
    try:
        span_years = (eq.index[-1] - eq.index[0]).total_seconds() / (365 * 24 * 3600)
    except Exception:
        span_years = max(1e-12, len(eq) / ANN)
    span_years = max(1e-12, float(span_years))

    cagr = float(eq.iloc[-1] ** (1.0 / span_years) - 1.0)
    final_eq = float(eq.iloc[-1])

    return pd.Series({"CAGR": cagr, "Sharpe": sharpe, "MaxDD": mdd, "FinalEq": final_eq})


def run_dynamic_demo(prices: pd.DataFrame,
                     lookback: int = 60,
                     max_w: float = 0.4,
                     fee_bps: float = 5.0) -> tuple[Dict[str, pd.Series], pd.DataFrame]:
  
    px = _sanitize_prices(prices)

    
    tw_eqM = equal_weight_targets(px, freq='M')
    tw_mvM = mv_targets(px, lookback=lookback, max_w=max_w, freq='M')

    
    band_ts = bands_from_vol(px, span=20, base_bps=40, k_vol=150)

    
    eq_eq_bands = portfolio_equity_dynamic_v3(px, tw_eqM, band_bps_ts=band_ts, fee_bps=fee_bps, max_turnover=0.25)
    eq_mv_bands = portfolio_equity_dynamic_v3(px, tw_mvM, band_bps_ts=band_ts, fee_bps=fee_bps, max_turnover=0.25)

    
    eq_eq_monthly = portfolio_equity_dynamic_v3(px, tw_eqM, band_bps_ts=0.0,    fee_bps=fee_bps, rebalance_calendar='M')
    eq_eq_50bps   = portfolio_equity_dynamic_v3(px, tw_eqM, band_bps_ts=50.0,   fee_bps=fee_bps, rebalance_calendar='M')

    curves = {
        "EQ_M_monthly": eq_eq_monthly,
        "EQ_M_50bps":   eq_eq_50bps,
        "EQ_M_dynBands": eq_eq_bands,
        "MV_M_dynBands": eq_mv_bands,
    }
    summary = pd.concat({k: brief_report(v) for k, v in curves.items()}, axis=1).T.sort_values("Sharpe", ascending=False).round(4)
    return curves, summary



def risk_parity_targets(prices: pd.DataFrame,
                        lookback: int = 60,
                        freq: str = "M",
                        max_w: float = 0.4,
                        allow_short: bool = False) -> pd.DataFrame:
    
    px = _sanitize_prices(prices)
    rets = px.pct_change().dropna(how="all")
    sched = px.resample(freq).last().index
    cols = px.columns
    out = []
    for t in sched:
        end = rets.index.searchsorted(t, side="right")
        start = max(0, end - lookback)
        win = rets.iloc[start:end]
        if len(win) < max(10, lookback//3):
            w = pd.Series(1.0/len(cols), index=cols)
        else:
            vol = win.std().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            inv = 1.0 / vol.replace(0.0, np.nan)
            w = inv.fillna(0.0)
            if not allow_short:
                w = w.clip(0.0, max_w)
            w = (w / (w.abs().sum() if allow_short else w.sum())) if w.abs().sum() > 0 else pd.Series(1.0/len(cols), index=cols)
        out.append(w.rename(t))
    tw_sched = pd.DataFrame(out)
    return tw_sched.reindex(px.index).ffill()





# NEW PART WITH FITERING COINS

def _align_signals_to_prices(prices: pd.DataFrame, signals: dict[str, pd.Series]) -> pd.DataFrame:
    px = _sanitize_prices(prices)
    cols = list(px.columns)
    S = pd.DataFrame(index=px.index, columns=cols, dtype=float)
    for a in cols:
        s = signals.get(a)
        if s is None: 
            S[a] = 0.0
        else:
            S[a] = s.reindex(px.index).astype(float).fillna(0.0)
    return S.fillna(0.0)


def signal_mv_targets(prices: pd.DataFrame,
                      signals: dict[str, pd.Series],
                      lookback: int = 60,
                      freq: str = "M",
                      gate_abs: float = 0.20,       
                      long_only: bool = True,
                      max_w: float = 0.25,          
                      net_neutral: bool = False,    
                      use_zscore: bool = True,      
                      cov_shrink: bool = True) -> pd.DataFrame:



    px   = _sanitize_prices(prices)
    rets = px.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    S    = _align_signals_to_prices(px, signals)  # S[t, asset] ∈ [-1..1]
    sched = px.resample(freq).last().index
    cols  = px.columns
    TW_rows = []

    for t in sched:
        
        end = rets.index.searchsorted(t, side="right")
        start = max(0, end - lookback)
        win = rets.iloc[start:end]
        s_t = S.loc[:t].iloc[-1].copy() if len(S.loc[:t]) else pd.Series(0.0, index=cols)

        if len(win) < max(10, lookback//3):
            w = pd.Series(1.0/len(cols), index=cols)  
            TW_rows.append(w.rename(t))
            continue

        # гейт по силе сигнала
        s_t[(s_t.abs() < gate_abs)] = 0.0

        
        if not long_only and net_neutral:
            s_t = s_t - s_t.mean()

        
        mu = s_t.copy()
        if use_zscore:
            mu_std = mu.replace(0, np.nan).std()
            if pd.notna(mu_std) and mu_std > 0:
                mu = (mu - mu.mean()) / mu_std
            mu = mu.fillna(0.0)

        
        try:
            if cov_shrink:
                lw = LedoitWolf().fit(win.fillna(0.0).values)
                cov = pd.DataFrame(lw.covariance_, index=cols, columns=cols)
            else:
                cov = win.cov().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        except Exception:
            cov = win.cov().replace([np.inf, -np.inf], np.nan).fillna(0.0)

        
        try:
            inv = pd.DataFrame(np.linalg.pinv(cov.values), index=cols, columns=cols)
        except Exception:
            inv = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)

        raw = inv.dot(mu)

        if long_only:
            w = raw.clip(lower=0.0)
            if max_w is not None:
                w = w.clip(0.0, max_w)
            s = w.sum()
            w = (w / s) if s > 0 else pd.Series(1.0/len(cols), index=cols)
        else:
            
            if raw.abs().sum() == 0:
                w = pd.Series(0.0, index=cols)
            else:
                w = raw / raw.abs().sum()
            if max_w is not None:
                w = w.clip(-max_w, max_w)
            
            g = w.abs().sum()
            if g > 1e-9:
                w = w / max(1.0, g)

        TW_rows.append(w.rename(t))

    tw_sched = pd.DataFrame(TW_rows)
    tw = tw_sched.reindex(px.index).ffill().reindex(columns=cols).fillna(0.0)
    return tw
