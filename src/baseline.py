
#------------------------------------------------------------------------------------------
# FIRST part of the project
#------------------------------------------------------------------------------------------

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


TRADING_HOURS_YEAR = 24 * 365   

@dataclass
class SMAConfig:
    fast: int = 20
    slow: int = 50
    fee_bps: float = 5.0       
    slippage_bps: float = 0.0  
    resample: Optional[str] = None 
    symbol: Optional[str] = None   
    annualizer: int = TRADING_HOURS_YEAR  

def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    need = {"ts", "close"}
    have = set(c.lower() for c in df.columns)

  
    out = df.copy()
    if "ts" not in out.columns:
        
        for cand in ["timestamp", "time", "date", "datetime"]:
            if cand in out.columns:
                out = out.rename(columns={cand: "ts"})
                break
    out["ts"] = pd.to_datetime(out["ts"], utc=False)
    out = out.sort_values("ts")
    out = out.set_index("ts")
    return out

def _apply_resample(df: pd.DataFrame, rule: Optional[str]) -> pd.DataFrame:
    if not rule:
        return df
    agg = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "open":
            agg[c] = "first"
        elif lc == "high":
            agg[c] = "max"
        elif lc == "low":
            agg[c] = "min"
        elif lc in ("close",):
            agg[c] = "last"
        elif lc in ("volume", "vol"):
            agg[c] = "sum"
        else:
    
            agg[c] = "last"
    return df.resample(rule).agg(agg).dropna(how="all")

def _cost_per_trade(fee_bps: float, slippage_bps: float) -> float:
    
    return (fee_bps + 2 * slippage_bps) / 1e4

def compute_metrics(equity: pd.Series, ann: int) -> Dict[str, float]:
    eq = equity.dropna()
    r = eq.pct_change().dropna()
    if len(r) == 0:
        return {"ROI": np.nan, "Sharpe": np.nan, "MaxDD": np.nan, "AnnVol": np.nan, "CAGR": np.nan, "FinalEq": float(eq.iloc[-1]) if len(eq) else np.nan}
    roi = float(eq.iloc[-1] - 1.0)
    vol = float(r.std(ddof=1) * np.sqrt(ann))
    sharpe = float((r.mean() * ann) / vol) if vol > 0 else np.nan
    cummax = eq.cummax()
    maxdd = float((eq / cummax - 1.0).min())
    years = len(eq) / ann
    cagr = float(eq.iloc[-1] ** (1/years) - 1) if years > 0 else np.nan
    return {"ROI": roi, "Sharpe": sharpe, "MaxDD": maxdd, "AnnVol": vol, "CAGR": cagr, "FinalEq": float(eq.iloc[-1])}

def run_sma(
    df: pd.DataFrame,
    fast: int = 20,
    slow: int = 50,
    fee_bps: float = 5.0,
    slippage_bps: float = 0.0,
    resample: Optional[str] = None,
    symbol: Optional[str] = None,
    annualizer: Optional[int] = None,
) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
   
    ann = annualizer if annualizer is not None else TRADING_HOURS_YEAR

    data = df.copy()
    
    if "symbol" in data.columns and symbol is not None:
        data = data[data["symbol"] == symbol].copy()

    data = _ensure_ohlcv(data)
    data = _apply_resample(data, resample)
    data = data[["close"]].copy().dropna()

    
    data["sma_fast"] = data["close"].rolling(int(fast)).mean()
    data["sma_slow"] = data["close"].rolling(int(slow)).mean()

    
    sig = pd.Series(0.0, index=data.index)
    sig[data["sma_fast"] > data["sma_slow"]] = 1.0

    
    sig = sig.shift(1).fillna(0.0)

    
    pos_change = sig.diff().abs().fillna(0.0)
    
    ret = data["close"].pct_change().fillna(0.0)

    gross = sig * ret
    cost = pos_change * _cost_per_trade(fee_bps=fee_bps, slippage_bps=slippage_bps)
    strat_ret = gross - cost

    equity = (1.0 + strat_ret).cumprod()
    equity.iloc[0] = 1.0

    metrics = compute_metrics(equity, ann=ann)

    
    signal = sig.astype(float)
    return signal, equity, metrics

