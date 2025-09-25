# trade_compare.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict


def _sanitize_series(s: pd.Series) -> pd.Series:
    s = s.sort_index()
    if s.index.has_duplicates:
        s = s.groupby(level=0).last()
    return s

# -----------------------
# Build trade-log
# -----------------------

def make_trades(price: pd.Series,
                signal: pd.Series,
                side: str = "long_only",
                *,
                band: float = 0.0,            
                cost_per_change: float = 0.0, 
                close_last: bool = False      
                ) -> pd.DataFrame:
    
    px  = _sanitize_series(price)
    raw = _sanitize_series(signal).reindex(px.index).fillna(0.0)

    
    if side == "long_only":
        state = (raw > band).astype(float)  # {0,1}
    else:
        s = raw.copy().astype(float)
        state = pd.Series(0.0, index=s.index)
        state[s >  band] =  1.0
        state[s < -band] = -1.0             # {-1,0,1}
    
    st = state.shift(1).fillna(0.0)

    prev = st.shift(1).fillna(0.0)
    ch   = st - prev
    change_idx = st.index[ch != 0]

    trades = []
    pos = 0.0
    entry_i = None
    entry_px = None

    for i, t in enumerate(st.index):
        if ch.iloc[i] == 0:
            continue

        new_pos  = float(st.iloc[i])
        old_pos  = float(prev.iloc[i])
        
        if pos != 0 and entry_i is not None:
            exit_i = i
            exit_px = float(px.iloc[i])
            ret = (exit_px/entry_px - 1.0) if pos > 0 else -(exit_px/entry_px - 1.0)
            
            close_cost = cost_per_change if old_pos != 0 else 0.0
            
            trades.append({
                "entry_time": px.index[entry_i],
                "exit_time":  px.index[exit_i],
                "entry_px":   entry_px,
                "exit_px":    exit_px,
                "ret":        float(ret),
                "ret_net":    float(ret - close_cost),  
                "hold_bars":  int(exit_i - entry_i),
            })
            pos = 0.0
            entry_i = None
            entry_px = None

        if new_pos != 0.0:
            pos = new_pos
            entry_i = i
            entry_px = float(px.iloc[i])
            
            open_has_cost = True
        else:
            open_has_cost = False


        if entry_i is not None:
            
            pass


    idx = st.index
    change_pos = np.flatnonzero(ch.values != 0)
    if len(change_pos) == 0:
        return pd.DataFrame(columns=["entry_time","exit_time","entry_px","exit_px","ret","ret_net","hold_bars"])

    # сегменты
    seg_starts = change_pos
    seg_ends   = np.r_[change_pos[1:], len(idx)-1]  
    if not close_last:
        if seg_ends[-1] == len(idx)-1:
            seg_starts = seg_starts[:-1]
            seg_ends   = seg_ends[:-1]

    rows = []
    for s_i, e_i in zip(seg_starts, seg_ends):
        new_pos = float(st.iloc[s_i])
        old_pos = float(prev.iloc[s_i])
        if new_pos == 0.0:
            
            continue
        entry_px = float(px.iloc[s_i])
        exit_px  = float(px.iloc[e_i])
        ret = (exit_px/entry_px - 1.0) if new_pos > 0 else -(exit_px/entry_px - 1.0)
        
        open_cost  = cost_per_change if new_pos != 0.0 else 0.0                          
        close_cost = cost_per_change if (e_i in change_pos) else (cost_per_change if close_last else 0.0)
        
        rows.append({
            "entry_time": idx[s_i],
            "exit_time":  idx[e_i],
            "entry_px":   entry_px,
            "exit_px":    exit_px,
            "ret":        float(ret),
            "ret_net":    float(ret - (open_cost + close_cost)),
            "hold_bars":  int(e_i - s_i),
        })

    tdf = pd.DataFrame(rows)
    if len(tdf):
        tdf = tdf.sort_values("entry_time").reset_index(drop=True)
    return tdf


# ---------------------------------
# Equity and metrics
# ---------------------------------

def equity_from_trades(trades_df: pd.DataFrame, start_eq: float = 1.0) -> pd.Series:
    
    if trades_df is None or trades_df.empty:
        return pd.Series(dtype=float)
    eq = start_eq * (1.0 + trades_df["ret_net"]).cumprod()
    eq.index = pd.to_datetime(trades_df["exit_time"])
    return eq

def trade_metrics(trades_df: pd.DataFrame) -> pd.Series:
    
    if trades_df is None or trades_df.empty:
        return pd.Series({
            "N": 0, "WinRate": np.nan, "ExpRet": np.nan, "MedianRet": np.nan,
            "AvgWin": np.nan, "AvgLoss": np.nan, "Payoff": np.nan, "PF": np.nan,
            "AvgHoldBars": np.nan, "Best": np.nan, "Worst": np.nan, "CumRet": np.nan
        })

    r = trades_df["ret_net"].astype(float)
    wins = r[r > 0]; losses = r[r <= 0]
    win_rate = len(wins) / len(r) if len(r) else np.nan
    avg_win = wins.mean() if len(wins) else np.nan
    avg_loss = losses.mean() if len(losses) else np.nan  # отрицательное
    payoff = (avg_win / abs(avg_loss)) if pd.notna(avg_loss) and avg_loss < 0 else np.nan
    pf = (wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else np.nan
    cum_ret = (1.0 + r).prod() - 1.0

    return pd.Series({
        "N": int(len(r)),
        "WinRate": float(win_rate),
        "ExpRet": float(r.mean()),
        "MedianRet": float(r.median()),
        "AvgWin": float(avg_win) if pd.notna(avg_win) else np.nan,
        "AvgLoss": float(avg_loss) if pd.notna(avg_loss) else np.nan,
        "Payoff": float(payoff) if pd.notna(payoff) else np.nan,
        "PF": float(pf) if pd.notna(pf) else np.nan,
        "AvgHoldBars": float(trades_df["hold_bars"].mean()),
        "Best": float(r.max()),
        "Worst": float(r.min()),
        "CumRet": float(cum_ret),
    })


# -----------------------
# Visual
# -----------------------
def plot_price_with_trades(price: pd.Series, signal: pd.Series, title="Price with trades", long_short=False):
    px  = _sanitize_series(price)
    sig = _sanitize_series(signal).reindex(px.index).fillna(0.0)
    sig_used = sig.shift(1).fillna(0.0)
    ch = sig_used.diff().fillna(sig_used)

    entries = ch[ch > 0].index
    exits   = ch[ch < 0].index

    plt.figure(figsize=(15,3))
    px.plot(label="price")
    plt.scatter(entries, px.reindex(entries), marker="^", color="green", s=60, label="entry")
    plt.scatter(exits,   px.reindex(exits),   marker="v", color="red",   s=60, label="exit/flip")
    plt.title(title); plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

def plot_trade_hist(trades_dict: Dict[str, pd.DataFrame], bins=40, title="Trade returns distribution"):
    plt.figure(figsize=(7,3))
    for name, tdf in trades_dict.items():
        if tdf is None or tdf.empty:
            continue
        plt.hist(tdf["ret_net"], bins=bins, alpha=0.4, label=name)
    plt.title(title); plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

def plot_trade_equities(eq_by_trades: Dict[str, pd.Series], title="Equity by trades (compounded)"):
    plt.figure(figsize=(15,3))
    for name, eq in eq_by_trades.items():
        if eq is None or len(eq) == 0:
            continue
        eq.plot(label=name)
    plt.title(title); plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

# ---------------------------------
# Compare wrapper
# ---------------------------------
def compare_by_trades(price: pd.Series,
                      signals: Dict[str, pd.Series],
                      side: str = "long_only"):
    
    px = _sanitize_series(price)
    sigs = {k: _sanitize_series(v).reindex(px.index).ffill().fillna(0.0) for k, v in signals.items()}

    trades_logs = {name: make_trades(px, sig, side=side) for name, sig in sigs.items()}
    summary = pd.DataFrame({name: trade_metrics(tdf) for name, tdf in trades_logs.items()}).T
    eq_trades = {name: equity_from_trades(tdf, start_eq=1.0) for name, tdf in trades_logs.items()}

    plot_trade_hist(trades_logs, bins=40, title="Trade returns distribution")
    plot_trade_equities(eq_trades, title="Equity by trades (compounded at exits)")

    return summary.round(4), trades_logs, eq_trades

