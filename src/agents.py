#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
# SECOND part of the project
#------------------------------------------------------------------------------------------


from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Iterable

from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# For base ML models
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

        

# CPD mathods
import ruptures as rpt

# for TS split and backtest models
from sklearn.model_selection import TimeSeriesSplit



TRADING_HOURS_YEAR = 24 * 365     


@dataclass
class SuiteConfig:
    fee_bps: float = 5.0
    slippage_bps: float = 0.0
    annualizer: int = TRADING_HOURS_YEAR



@dataclass 
class _CPDConfig:
    """Config for CPD"""
    method: str = "pelt"
    model: str = "rbf" 
    min_size: int = 10
    jump: int = 5
    penalty: Optional[float] = None
    metric: str = "bic"
    target_changes_per_year: Optional[float] = None



class AITradingSuite:
    """
    Union class with models: SMA, ARIMA, GARCH, Logistic Regression,
    """

    def __init__(self, df: pd.DataFrame, cfg: SuiteConfig = SuiteConfig()):
        self.raw = df.copy()
        self.cfg = cfg
        self.df = self._normalize_df(self.raw)

    # ---------- utils ----------
    @staticmethod
    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # Время → индекс
        if "ts" not in out.columns:
            for cand in ("timestamp", "time", "date", "datetime"):
                if cand in out.columns:
                    out = out.rename(columns={cand: "ts"})
                    break
        assert "ts" in out.columns, "Нужна колонка времени 'ts'"
        assert "close" in out.columns, "Нужна колонка 'close'"

        out["ts"] = pd.to_datetime(out["ts"])
        out = out.sort_values("ts").set_index("ts")

        # ret1 пригодится для ARIMA/GARCH/ML
        if "ret1" not in out.columns:
            out["ret1"] = out["close"].pct_change()
        out["ret1"] = out["ret1"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return out

    @staticmethod
    def _cost_per_trade(fee_bps: float, slippage_bps: float) -> float:
        # комиссия на оборот веса + проскальзывание в обе стороны
        return (fee_bps + 2 * slippage_bps) / 1e4

    @staticmethod
    def _align(price: pd.Series | pd.DataFrame, sig: pd.Series) -> tuple[pd.Series, pd.Series]:
        # выравниваем по общему индексу
        idx = price.index.intersection(sig.index)
        return price.reindex(idx), sig.reindex(idx).fillna(0.0)

    def backtest(self, price: pd.Series, signal: pd.Series) -> pd.Series:
        """Простой бэктест: long/flat/short сигнал {-1,0,1}. Комиссия на изменение позиции."""
        p, s = self._align(price, signal.shift(1).fillna(0.0))  # вход со следующей свечи
        ret = p.pct_change().fillna(0.0)
        pos_change = s.diff().abs().fillna(0.0)
        gross = s * ret
        cost = pos_change * self._cost_per_trade(self.cfg.fee_bps, self.cfg.slippage_bps)
        strat_ret = gross - cost
        eq = (1.0 + strat_ret).cumprod()
        eq.iloc[0] = 1.0
        return eq

    def metrics(self, equity: pd.Series) -> pd.Series:
        r = equity.pct_change().dropna()
        ann = self.cfg.annualizer
        vol = r.std(ddof=1) * np.sqrt(ann) if len(r) > 1 else np.nan
        sharpe = (r.mean() * ann / vol) if (vol is not None and vol > 0) else np.nan
        cummax = equity.cummax()
        mdd = float((equity / cummax - 1).min())
        years = max(1e-9, len(equity) / ann)
        cagr = float(equity.iloc[-1] ** (1 / years) - 1)
        return pd.Series({
            "FinalEq": float(equity.iloc[-1]),
            "ROI": float(equity.iloc[-1] - 1),
            "Sharpe": float(sharpe) if sharpe == sharpe else np.nan,
            "AnnVol": float(vol) if vol == vol else np.nan,
            "MaxDD": mdd,
            "CAGR": cagr
        })

    # ---------- models ----------
    def sma(self, fast: int = 20, slow: int = 50) -> pd.Series:
        df = self.df.copy()
        df["sma_fast"] = df["close"].rolling(int(fast)).mean()
        df["sma_slow"] = df["close"].rolling(int(slow)).mean()
        sig = pd.Series(0.0, index=df.index)
        sig[df["sma_fast"] > df["sma_slow"]] = 1.0
        return sig.rename("SMA")

    def arima(self,
              window: int = 400,
              tune: bool = False,
              param_grid: Optional[Dict[str, Iterable[int]]] = None,
              order: tuple[int, int, int] = (1, 0, 1)) -> pd.Series:
        """ARIMA по скользящему окну на ret1. tune=True — перебор по сетке p,d,q с tqdm."""
        r = self.df["ret1"]
        sig = pd.Series(0.0, index=self.df.index)

        if tune and param_grid:
            best_aic, best_params = np.inf, order
    
            for p in tqdm(param_grid.get("p", [order[0]]), desc="ARIMA p"):
                for d in param_grid.get("d", [order[1]]):
                    for q in param_grid.get("q", [order[2]]):
                        for t in range(window, len(r) - 1):
                            end_idx = r.index[t]
                            win = r.iloc[t - window:t]
                            if win.std(ddof=1) == 0 or win.isna().any():
                                continue
                            try:
                                fit = ARIMA(win, order=(p, d, q)).fit(method_kwargs={"warn_convergence": False})
                                if fit.aic < best_aic:
                                    best_aic, best_params = fit.aic, (p, d, q)
                                fc = float(fit.forecast(steps=1).iloc[0])
                                sig.loc[end_idx] = np.sign(fc)
                            except Exception:
                                continue
            print(f"[ARIMA] best params={best_params}, AIC={best_aic:.2f}")
        else:
            p, d, q = order
            for t in range(window, len(r) - 1):
                end_idx = r.index[t]
                win = r.iloc[t - window:t]
                if win.std(ddof=1) == 0 or win.isna().any():
                    continue
                try:
                    fit = ARIMA(win, order=(p, d, q)).fit(method_kwargs={"warn_convergence": False})
                    fc = float(fit.forecast(steps=1).iloc[0])
                    sig.loc[end_idx] = np.sign(fc)
                except Exception:
                    continue

        return sig.fillna(0.0).rename("ARIMA")

    def garch(self,
              window: int = 400,
              tune: bool = False,
              param_grid: Optional[Dict[str, Iterable[int]]] = None,
              order: tuple[int, int] = (1, 1)) -> pd.Series:
        """GARCH( p, q ) по скользящему окну на ret1. Сигнал — знак изменения волатильности"""
        r = self.df["ret1"]
        sig = pd.Series(0.0, index=self.df.index)

        if tune and param_grid:
            best_aic, best_params = np.inf, order
            for p in tqdm(param_grid.get("p", [order[0]]), desc="GARCH p"):
                for q in param_grid.get("q", [order[1]]):
                    for t in range(window, len(r) - 1):
                        end_idx = r.index[t]
                        win = r.iloc[t - window:t]
                        if win.std(ddof=1) == 0 or win.isna().any():
                            continue
                        try:
                            fit = arch_model(win, vol="Garch", p=p, q=q).fit(disp="off")
                            if fit.aic < best_aic:
                                best_aic, best_params = fit.aic, (p, q)
                            vol = fit.conditional_volatility
                            sig.loc[end_idx] = np.sign(vol.iloc[-2] - vol.iloc[-1]) if len(vol) > 1 else 0.0
                        except Exception:
                            continue
            print(f"[GARCH] best params={best_params}, AIC={best_aic:.2f}")
        else:
            p, q = order
            for t in range(window, len(r) - 1):
                end_idx = r.index[t]
                win = r.iloc[t - window:t]
                if win.std(ddof=1) == 0 or win.isna().any():
                    continue
                try:
                    fit = arch_model(win, vol="Garch", p=p, q=q).fit(disp="off")
                    vol = fit.conditional_volatility
                    sig.loc[end_idx] = np.sign(vol.iloc[-2] - vol.iloc[-1]) if len(vol) > 1 else 0.0
                except Exception:
                    continue

        return sig.fillna(0.0).rename("GARCH")



    # --------------------------------------------------------------------------------
    # ---------- FEATIRES FOE LOGREG ----------
    # --------------------------------------------------------------------------------
    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0.0)
        down = (-delta).clip(lower=0.0)
        roll_up = up.ewm(alpha=1/period, adjust=False).mean()
        roll_down = down.ewm(alpha=1/period, adjust=False).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        out = 100 - (100 / (1 + rs))
        return out.fillna(50.0)

    @staticmethod
    def _parkinson_vol(high: pd.Series, low: pd.Series, window: int = 24) -> pd.Series:
        if high is None or low is None or high.isnull().all() or low.isnull().all():
            
            idx = high.index if high is not None else (low.index if low is not None else None)
            return pd.Series(index=idx, dtype=float)
        hl = (high / low).replace([np.inf, -np.inf], np.nan)
        x = np.log(hl)
        return (x**2).rolling(window).mean() / (4 * np.log(2))

    @staticmethod
    def _make_features(df: pd.DataFrame,
                       mom_windows=(24, 72),
                       vol_windows=(24, 72),
                       rsi_period=14,
                       parkinson_window=24,
                       drop_na_tail=True) -> pd.DataFrame:
        
        out = df.copy()

        if "ts" in out.columns:
            out["ts"] = pd.to_datetime(out["ts"])
            out = out.sort_values("ts").set_index("ts")

        out["ret1"] = out["close"].pct_change().replace([np.inf, -np.inf], np.nan)

        
        for w in mom_windows:
            out[f"mom_{w}h"] = out["close"].pct_change(w)

        r = out["ret1"]
        for w in vol_windows:
            out[f"vol_{w}h"] = r.rolling(w).std()

        if {"high", "low"}.issubset(out.columns):
            out[f"parkinson_{parkinson_window}h"] = AITradingSuite._parkinson_vol(
                out["high"], out["low"], window=parkinson_window
            )
        else:
            out[f"parkinson_{parkinson_window}h"] = np.nan

        # RSI
        out["rsi14"] = AITradingSuite._rsi(out["close"], period=rsi_period)

        out["lag1"] = r.shift(1)
        out["lag2"] = r.shift(2)
        out["lag24"] = r.shift(24)

        out["target_up"] = (out["close"].pct_change().shift(-1) > 0).astype(int)

        out = out.replace([np.inf, -np.inf], np.nan)
        feat_cols = [c for c in out.columns if c != "target_up"]
        out[feat_cols] = out[feat_cols].ffill().bfill()

        if drop_na_tail:
            
            out = out.dropna(subset=["ret1", "target_up"])

        return out


# TARGET BUILDING 
# ----------------------------------------------------------------------------------------------------------------
# Different ways for target settings
# ----------------------------------------------------------------------------------------------------------------
    def set_target(self, horizon: int = 6, thr: float = 0.002, mode: str = "cls", column_price: str = "close"):
        """
        Формирование таргета без утечки.
        - mode='cls'     : бинарный таргет без нейтральной зоны. Внутри ±thr метка берётся по знаку будущей доходности.
                        thr может быть float или (thr_long, thr_short).
        - mode='cls_vol' : бинарный таргет с порогами k * rolling_std (k = thr). Внутри ±k·σ метка по знаку будущей доходности.
        """

        
        p = self.df[column_price].astype(float)
        fwd_ret = p.shift(-horizon) / p - 1.0

        
        def _trim_to_fwd():
            last_valid = fwd_ret.dropna().index.max()
            if last_valid is not None:
                self.df[:]  
                self.df = self.df.loc[:last_valid]
            return last_valid

        if mode == "reg":
            self.df["target_reg"] = fwd_ret.astype(float)
            _trim_to_fwd()
            return

        if mode == "cls":
            
            if isinstance(thr, (tuple, list)) and len(thr) == 2:
                thr_long, thr_short = float(thr[0]), float(thr[1])
            else:
                thr_long = thr_short = float(thr)

            y = pd.Series(np.nan, index=self.df.index, dtype="float64")
            up = fwd_ret > +thr_long
            dn = fwd_ret < -thr_short
            mid = ~(up | dn) & fwd_ret.notna()

            y[up] = 1.0
            y[dn] = 0.0
            
            y[mid] = (fwd_ret.loc[mid] >= 0).astype(float)

            self.df["target_up"] = y
            _trim_to_fwd()
            
            self.df["target_up"] = self.df["target_up"].astype(float)
            return

        if mode == "cls_vol":
            
            k = float(thr)
            r = self.df["ret1"].astype(float)
            vol = r.rolling(horizon, min_periods=1).std() * np.sqrt(max(1, horizon))
            vol = vol.replace([np.inf, -np.inf], np.nan).fillna(0.0)  

            y = pd.Series(np.nan, index=self.df.index, dtype="float64")
            up = fwd_ret >  k * vol
            dn = fwd_ret < -k * vol
            mid = ~(up | dn) & fwd_ret.notna()

            y[up] = 1.0
            y[dn] = 0.0
            
            y[mid] = (fwd_ret.loc[mid] >= 0).astype(float)

            self.df["target_up"] = y
            _trim_to_fwd()
            self.df["target_up"] = self.df["target_up"].astype(float)
            return

        if mode == "triple":
            tp = float(thr.get("tp", 0.004))
            sl = float(thr.get("sl", 0.004))
            timeout_policy = thr.get("timeout", "final")  # по умолчанию без NaN

            y = pd.Series(np.nan, index=self.df.index, dtype="float64")
            vals = p.values
            n = len(p)


            last_valid = _trim_to_fwd()
            if last_valid is not None:
                n = self.df.index.get_loc(last_valid) + 1

            for i in range(n - horizon):
                p0 = vals[i]
                win = vals[i+1:i+1+horizon]
                if np.isnan(p0) or np.isnan(win).any():
                    continue
                path_ret = win / p0 - 1.0
                if np.max(path_ret) >= tp:
                    y.iloc[i] = 1.0
                elif np.min(path_ret) <= -sl:
                    y.iloc[i] = 0.0
                else:
                    if timeout_policy == "final":
                        y.iloc[i] = 1.0 if (vals[i+horizon] / p0 - 1.0) > 0 else 0.0
                    else:  
                        y.iloc[i] = np.nan

            self.df["target_up"] = y.loc[self.df.index]
           
            self.df["target_up"] = self.df["target_up"].astype(float)
            return

        raise ValueError("mode must be one of: 'cls' | 'cls_vol' | 'reg' | 'triple'")


    def build_features(
        self,
        mom_windows=(24, 72),
        vol_windows=(24, 72),
        rsi_period=14,
        parkinson_window=24,
        overwrite=False,
        target_horizon: int = 6,
        target_thr: float = 0.002,
        target_mode: str = "cls",
    ) -> None:
        """
        BUILD FEATURES + target
        """
        tmp = self.df.reset_index().rename(columns={"index": "ts"}) if "ts" not in self.df.columns else self.df
        if "ts" not in tmp.columns:
            tmp = self.df.reset_index()
            tmp = tmp.rename(columns={tmp.columns[0]: "ts"})

        enriched = AITradingSuite._make_features(
            tmp,
            mom_windows=mom_windows,
            vol_windows=vol_windows,
            rsi_period=rsi_period,
            parkinson_window=parkinson_window,
            drop_na_tail=False,
        )
        enriched = enriched.reindex(self.df.index)

        for c in enriched.columns:
            if c not in self.df.columns or overwrite:
                self.df[c] = enriched[c]

        
        self.df = self.df.replace([np.inf, -np.inf], np.nan).ffill()

        
        self.set_target(horizon=target_horizon, thr=target_thr, mode=target_mode)




    def logreg(self, features: list[str] | None = None, C=1.0, penalty="l2") -> pd.Series:

        if features is None:
            self.build_features()
            default_feats = [
                "mom_24h","mom_72h","vol_24h","vol_72h",
                "parkinson_24h","rsi14","ret1","lag1","lag2","lag24"
            ]
            features = [f for f in default_feats if f in self.df.columns]

        assert "target_up" in self.df.columns, "После build_features в self.df должна быть колонка target_up."

        X = self.df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = self.df["target_up"].astype(int)

        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=1000, solver="liblinear", C=C, penalty=penalty))
        ])
        pipe.fit(X, y)
        proba = pd.Series(pipe.predict_proba(X)[:, 1], index=self.df.index)
        sig = np.sign(proba - 0.5)
        return sig.rename("ML")

    
    def boost_simple(
        self,
        features: list[str] | None = None,
        threshold: float = 0.5,
        params: dict | None = None,
        scale: bool = True,
        return_proba: bool = False,
    ) -> (pd.Series | tuple[pd.Series, pd.Series]):
        """
        Gradient Boosting (без HPO). Возвращает сигналы {-1,0,1} по порогу.
        """
        # 1) фичи/таргет
        if features is None:
            self.build_features()
            features = [c for c in [
                "mom_24h","mom_72h","vol_24h","vol_72h",
                "parkinson_24h","rsi14","ret1","lag1","lag2","lag24",
                "regime_id","regime_vol","regime_mean"  # если ранее добавляла CPD-фичи
            ] if c in self.df.columns]

        assert "target_up" in self.df.columns, "Нужен target_up (вызови build_features())."

        X = self.df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        y = self.df["target_up"].astype(int)

        base_params = dict(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            random_state=42
        )
        if params:
            base_params.update(params)

        
        if scale:
            model = Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("gb", GradientBoostingClassifier(**base_params))
            ])
        else:
            model = GradientBoostingClassifier(**base_params)

        
        model.fit(X, y)
        if isinstance(model, Pipeline):
            proba = pd.Series(model.predict_proba(X)[:, 1], index=X.index, name="boost_proba")
        else:
            proba = pd.Series(model.predict_proba(X)[:, 1], index=X.index, name="boost_proba")

        
        sig = np.sign(proba - threshold).astype(float).rename("BOOST_SIMPLE")

        return (sig, proba) if return_proba else sig



#----------------------------------------------------------------------------------------------------------------
# CPD model
#----------------------------------------------------------------------------------------------------------------

# ---------- CPD helpers ----------
    @staticmethod
    def __robust_std(x: pd.Series) -> float:
        x = pd.Series(x).dropna().values
        if len(x) < 2:
            return 0.0
        
        q1, q3 = np.quantile(x, [0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 3*iqr, q3 + 3*iqr
        x = x[(x >= lo) & (x <= hi)]
        if len(x) < 2:
            return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        return float(np.std(x, ddof=1))

    @staticmethod
    def __annualized_changes(idx: pd.DatetimeIndex, n_changes: int) -> float:
        if len(idx) < 2 or n_changes <= 0:
            return 0.0
        total_days = (idx[-1] - idx[0]).days
        years = max(total_days / 365.0, 1e-6)
        return n_changes / years

 
    # ---------- penalty from metric  ----------
    @staticmethod
    def __cpd_base_penalty(series: pd.Series, metric: str = "bic") -> float:
        """
        Start score for metric: 'aic' | 'bic' | 'var'
        """
        s = series.dropna().values
        T = len(s)
        if T < 5:
            return 1.0
        

        sigma = AITradingSuite.__robust_std(series) 

        m = metric.lower() 
        if m == "bic":
            return max(1e-8, (sigma ** 2) * np.log(T))
        elif m == "var":
            return max(1e-8, sigma ** 2)
        elif m == "aic":
            
            num_params = 2
            with np.errstate(divide='ignore', invalid='ignore'):
                ll = -np.sum((s - np.nanmean(s)) ** 2) / (2 * (sigma ** 2 + 1e-8))
            aic = 2 * num_params - 2 * ll
            return max(1e-8, aic)
        else:
            raise ValueError(f"Unknown metric '{metric}'. Use 'aic'|'bic'|'var'.")
    
    @staticmethod
    def __build_algo(rpt, cfg: _CPDConfig):
        if cfg.method == "pelt":
            return rpt.Pelt(model=cfg.model, min_size=cfg.min_size, jump=cfg.jump)
        elif cfg.method == "binseg":
            return rpt.Binseg(model=cfg.model, min_size=cfg.min_size, jump=cfg.jump)
        elif cfg.method == "kernel":
            
            return rpt.KernelCPD()
        elif cfg.method == "wbs":
            # Window-based: width ~= min_size
            return rpt.Window(width=cfg.min_size, model=cfg.model)
        else:
            raise ValueError(f"Unknown CPD method '{cfg.method}'. Use 'pelt'|'binseg'|'kernel'|'wbs'.")
    
    @staticmethod
    def __predict_bkps(algo, x: np.ndarray, penalty: float) -> list:
        algo.fit(x)
        bkps = algo.predict(pen=penalty)
        
        if bkps and bkps[-1] == len(x):
            bkps = bkps[:-1]
        return bkps

    # ---------- PUBLIC CPD methods ----------
    def detect_change_points(self,
                            series: pd.Series,
                            # **kwargs
                            method: str = "pelt",
                            model: str = "rbf",
                            penalty: Optional[float] = None,
                            metric: str = "bic",
                            min_size: int = 10,
                            jump: int = 5,
                            target_changes_per_year: Optional[float] = None,
                            penalty_bounds: tuple[float, float] = (1e-6, 1e6),
                            max_iter_search: int = 18
                            ) -> pd.Series:
        """
        Adaptive ways to choose mode
        """
        s = series.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        x = s.values.reshape(-1, 1)

        # cfg = _CPDConfig(**kwargs)
        cfg = _CPDConfig(method=method.lower(), model=model, min_size=min_size, jump=jump)
        algo = self.__build_algo(rpt, cfg)

        
        base_pen = self.__cpd_base_penalty(s, metric=metric) if penalty is None else float(penalty)

        
        if target_changes_per_year is None:
            bkps = self.__predict_bkps(algo, x, penalty=base_pen)
            reg = np.zeros(len(s), dtype=int)
            seg_id, start = 0, 0
            for end in bkps:
                reg[start:end] = seg_id
                start = end
                seg_id += 1
            return pd.Series(reg, index=series.index, name="regime_id")

        
        
        target = float(target_changes_per_year)
        if target <= 0:
            
            big = penalty_bounds[1]
            bkps = self.__predict_bkps(algo, x, penalty=big)
            reg = np.zeros(len(s), dtype=int)
            seg_id, start = 0, 0
            for end in bkps:
                reg[start:end] = seg_id
                start = end
                seg_id += 1
            return pd.Series(reg, index=series.index, name="regime_id")

        
        lo, hi = penalty_bounds
        best_reg = None
        for _ in range(max_iter_search):
            mid = np.sqrt(lo * hi)
            pen_try = base_pen * mid
            bkps = self.__predict_bkps(algo, x, penalty=pen_try)
            chg_per_year = self.__annualized_changes(series.index, len(bkps))
            if chg_per_year > target:
              
                lo = mid
            else:
              
                hi = mid
            best_reg = bkps

        
        bkps = best_reg if best_reg is not None else []
        reg = np.zeros(len(s), dtype=int)
        seg_id, start = 0, 0
        for end in bkps:
            reg[start:end] = seg_id
            start = end
            seg_id += 1
        return pd.Series(reg, index=series.index, name="regime_id")


    def add_cpd_features(
        self,
        on: str = "ret1",
        *,
        method: str = "pelt",
        model: str = "rbf",
        penalty: float | None = None,
        metric: str = "bic",
        min_size: int = 10,
        jump: int = 5,
        target_changes_per_year: float | None = None,
        window_stat: int = 24,
        lag_features: int = 1,   # лагируем режимы, чтобы не видеть будущее
        **_,
    ) -> None:
        """
        CPD-features
        """
        s = self.df[on].astype(float)

        reg_full = self.detect_change_points(
            series=s,
            method=method,
            model=model,
            penalty=penalty,
            metric=metric,
            min_size=min_size,
            jump=jump,
            target_changes_per_year=target_changes_per_year,
        )

        reg = reg_full.shift(lag_features)  
        reg.name = "regime_id"
        self.df["regime_id"] = reg

        
        changed = (reg != reg.shift(1)).fillna(True)
        self.df["regime_len"] = (~changed).groupby(reg).cumsum().fillna(0).astype(int) + 1

        
        def _rolling_std(x: pd.Series) -> pd.Series:
            return x.rolling(window_stat, min_periods=1).std()

        def _rolling_mean(x: pd.Series) -> pd.Series:
            return x.rolling(window_stat, min_periods=1).mean()

        grp = self.df.groupby("regime_id", group_keys=False)
        self.df["regime_vol"]  = grp["ret1"].transform(_rolling_std)
        self.df["regime_mean"] = grp["ret1"].transform(_rolling_mean)

        # очистка
        for col in ["regime_id", "regime_len", "regime_vol", "regime_mean"]:
            self.df[col] = self.df[col].replace([np.inf, -np.inf], np.nan).ffill()



    def cpd_signal(
        self,
        cooldown: int = 12,                
        method: str = "pelt",
        model: str = "rbf",
        penalty: Optional[float] = None,
        metric: str = "bic",
        min_size: int = 10,
        jump: int = 5,
        target_changes_per_year: Optional[float] = None,
        regime_bias_window: int = 24*3,
        scale_by_vol: bool = True,         
        bias_mode: str = "ret_mean",       
    ) -> pd.Series:
        price = self.df["close"].astype(float)
        ret = price.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

        regime = self.detect_change_points(
            series=ret, method=method, model=model, penalty=penalty, metric=metric,
            min_size=min_size, jump=jump, target_changes_per_year=target_changes_per_year
        )
        changed = (regime != regime.shift(1)).fillna(True)

        if bias_mode == "ret_mean":
            bias = ret.rolling(regime_bias_window).mean().fillna(0.0)
        elif bias_mode == "price_trend":
            
            w = regime_bias_window
            z = price.rolling(w).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if x.notna().all() else 0.0, raw=False)
            bias = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            raise ValueError("bias_mode must be 'ret_mean' or 'price_trend'")

        sig = np.sign(bias).astype(float).rename("CPD")

        
        cooldown_mask = pd.Series(0.0, index=ret.index)
        chg_idx = np.where(changed.values)[0]
        for i in chg_idx:
            end = min(i + cooldown, len(cooldown_mask))
            if end > i:
                
                cooldown_mask.iloc[i:end] = np.linspace(0.0, 1.0, end - i, endpoint=False)

        # итоговая сила сигнала = base_strength * cooldown_strength
        strength = pd.Series(1.0, index=ret.index) * cooldown_mask.clip(0, 1)
        strength[strength == 0] = 1.0  # до смены режима сила = 1

        if scale_by_vol:
            vol = ret.rolling(regime_bias_window).std().replace(0, np.nan).bfill().ffill()
            strength = strength * (1.0 / (1.0 + 10 * vol)).clip(0.25, 1.0)

        sig = (sig * strength).clip(-1, 1)
        return sig.rename("CPD")



# ----------------------------------------------------------------------------------------------------
# CV for TS
# ----------------------------------------------------------------------------------------------------
    # @staticmethod
    def _ts_cv_splits(self, n: int, n_splits: int = 5, min_train_size: int = 24*60):
        tss = TimeSeriesSplit(n_splits=n_splits)
    
        base_idx = np.arange(n)
        start = min_train_size
        for tr, te in tss.split(base_idx[start:]):
            yield base_idx[start:][tr], base_idx[start:][te]

    def _search_threshold_by_sharpe(self, proba: pd.Series, price: pd.Series, grid=None):
        if grid is None:
            grid = np.linspace(0.4, 0.6, 21)
        proba = proba.dropna()
        price = price.reindex(proba.index)
        best = (0.5, -np.inf, None)
        for th in grid:
            sig = np.sign(proba - th).astype(float)
            eq = self.backtest(price, sig)
            m = self.metrics(eq)
            sc = m["Sharpe"] if pd.notna(m["Sharpe"]) else -np.inf
            if sc > best[1]:
                best = (float(th), float(sc), eq)
        return best  

    def _fit_predict_ml_fold(self, model, X, y, train_idx, test_idx):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = pd.Series(model.predict_proba(X.iloc[test_idx])[:,1], index=X.index[test_idx])
        return proba


    def cv_binary_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        price: pd.Series,
        model,                     
        n_splits: int = 5,
        min_train_size: int = 24*60,
        search_threshold: bool = True,
    ):
        oof = pd.Series(index=X.index, dtype=float)
        for tr, te in self._ts_cv_splits(len(X), n_splits=n_splits, min_train_size=min_train_size):
            model.fit(X.iloc[tr], y.iloc[tr])
            proba = pd.Series(model.predict_proba(X.iloc[te])[:, 1], index=X.index[te])
            oof.loc[proba.index] = proba

        valid = oof.dropna()
        yv = y.loc[valid.index].astype(int)
        auc = float(roc_auc_score(yv, valid)) if len(valid.unique()) > 1 else np.nan

        if search_threshold:
            best_th, best_sharpe, best_eq = self._search_threshold_by_sharpe(valid, price)
        else:
            best_th = 0.5
            sig = np.sign(valid - best_th).astype(float)
            best_eq = self.backtest(price.loc[valid.index], sig)
            best_sharpe = self.metrics(best_eq)["Sharpe"]

        rep = {
            "AUC_OOF": auc,
            "BestThreshold": best_th,
            "Sharpe_OOF": float(best_sharpe),
            "FinalEq_OOF": float(best_eq.iloc[-1]),
        }
        sig_oof = np.sign(valid - best_th).astype(float)
        return valid.rename("proba_oof"), sig_oof.rename("sig_oof"), rep


    def cv_logreg_strong(
        self,
        features: list[str] | None = None,
        C: float = 0.5,
        penalty: str = "l2",
        n_splits: int = 5,
        min_train_size: int = 24*60,
    ):
        if features is None:
            self.build_features()
            features = [c for c in [
                "mom_24h","mom_72h","vol_24h","vol_72h",
                "parkinson_24h","rsi14","ret1","lag1","lag2","lag24",
                "regime_id","regime_vol","regime_mean"
            ] if c in self.df.columns]

        assert "target_up" in self.df.columns
        X = self.df[features].astype(float).replace([np.inf,-np.inf], np.nan).fillna(0.0)
        y = self.df["target_up"]
        price = self.df["close"]

        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=1000, solver="liblinear", C=C, penalty=penalty))

        ])
        

# get probability from cv_binary_model
        proba_oof, _sig_unused, rep = self.cv_binary_model(
                X, y, price, pipe, n_splits=n_splits, min_train_size=min_train_size, search_threshold=False

            )
        

# new parameter to control signals from model
        HI = getattr(self, "_cv_hi", 0.62)
        LO = getattr(self, "_cv_lo", 0.38)
        MIN_HOLD = getattr(self, "_cv_min_hold", 24)
        ALLOW_SHORT = getattr(self, "_cv_allow_short", False)

       
        sig = self.__hysteresis_from_proba(proba_oof, hi=HI, lo=LO, allow_short=ALLOW_SHORT)
        if MIN_HOLD and MIN_HOLD > 0:
            sig = self.__enforce_min_hold(sig, min_hold=MIN_HOLD)
        sig = sig.rename("LOGREG_OOF")

        
        rep = dict(rep)
        rep.update({"postproc": {"hi": HI, "lo": LO, "min_hold": MIN_HOLD, "allow_short": ALLOW_SHORT}})
        return proba_oof.rename("proba_oof"), sig, rep
        



    def cv_boost_strong(
        self,
        features: list[str] | None = None,
        n_splits: int = 5,
        min_train_size: int = 24*60,
        params: dict | None = None,
    ):
        if features is None:
            self.build_features()
            features = [c for c in [
                "mom_24h","mom_72h","vol_24h","vol_72h",
                "parkinson_24h","rsi14","ret1","lag1","lag2","lag24",
                "regime_id","regime_vol","regime_mean"
            ] if c in self.df.columns]

        assert "target_up" in self.df.columns
        X = self.df[features].astype(float).replace([np.inf,-np.inf], np.nan).fillna(0.0)
        y = self.df["target_up"].astype(int)
        price = self.df["close"]

        base = dict(
            n_estimators=600,         
            learning_rate=0.02,      
            max_depth=2,             
            subsample=0.6,           
            random_state=42
        )
        if params: base.update(params)

        model = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("gb", GradientBoostingClassifier(**base))
        ])
       
        proba_oof, _sig_unused, rep = self.cv_binary_model(
        X, y, price, model, n_splits=n_splits, min_train_size=min_train_size, search_threshold=False
        )

        HI = getattr(self, "_cv_hi", 0.64)   
        LO = getattr(self, "_cv_lo", 0.36)
        MIN_HOLD = getattr(self, "_cv_min_hold", 24)
        ALLOW_SHORT = getattr(self, "_cv_allow_short", False)

        sig = self.__hysteresis_from_proba(proba_oof, hi=HI, lo=LO, allow_short=ALLOW_SHORT)
        if MIN_HOLD and MIN_HOLD > 0:
            sig = self.__enforce_min_hold(sig, min_hold=MIN_HOLD)
        sig = sig.rename("BOOST_OOF")

        rep = dict(rep)
        rep.update({"postproc": {"hi": HI, "lo": LO, "min_hold": MIN_HOLD, "allow_short": ALLOW_SHORT}})
        return proba_oof.rename("proba_oof"), sig, rep




    def garch_risk_scaler(
        self,
        window_model: int = 400,
        order: tuple[int,int] = (1,1),
        vol_quantile: float = 0.7,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
    ) -> pd.Series:
    
        r = self.df["ret1"].astype(float)
        scale = pd.Series(index=r.index, dtype=float)

        preds = []
        for t in range(window_model, len(r)):
            win = r.iloc[t-window_model:t].dropna()
            if len(win) < 40 or win.std(ddof=1) == 0:
                preds.append(np.nan); continue
            try:
                fit = arch_model(win, vol="Garch", p=order[0], q=order[1]).fit(disp="off")
                
                sigma = float(fit.conditional_volatility.iloc[-1])
                preds.append(sigma)
            except Exception:
                preds.append(np.nan)

        vol_series = pd.Series(preds, index=r.index[window_model:])
        vol_ref = vol_series.rolling(24*30).quantile(vol_quantile).bfill().ffill()
       
        z = (vol_series / (vol_ref + 1e-12)).clip(0.5, 2.0)
        scl = (1.0 / z).clip(min_scale, max_scale)
        scale.loc[vol_series.index] = scl
       
        scale = scale.fillna(1.0).clip(min_scale, max_scale).rename("risk_scale")
        return scale

    def apply_risk_scaling(self, signal: pd.Series, scale: pd.Series) -> pd.Series:
        sig = signal.reindex(scale.index).fillna(0.0) * scale
        return sig.clip(-1, 1).rename(signal.name + "_RS")


# ------------------------------------------------------------------------------------------------------------------------------
# FOR MODELS SIGNS FILTERING
# ------------------------------------------------------------------------------------------------------------------------------ 
    @staticmethod
    def __hysteresis_from_proba(proba: pd.Series, hi: float = 0.6, lo: float = 0.4, allow_short: bool = False) -> pd.Series:
       
        p = proba.astype(float).dropna()
        out = pd.Series(0.0, index=p.index)
        pos = 0.0
        for t, v in p.items():
            if pos == 1.0:
                if v < lo:
                    pos = -1.0 if allow_short else 0.0
            elif pos == -1.0:
                if v > hi:
                    pos = 1.0
            else:
                if v > hi:
                    pos = 1.0
                elif allow_short and v < lo:
                    pos = -1.0
                else:
                    pos = 0.0
            out.loc[t] = pos
        return out


    @staticmethod
    def __enforce_min_hold(sig: pd.Series, min_hold: int = 24) -> pd.Series:
      
        s = sig.astype(float).fillna(0.0)
        out = pd.Series(0.0, index=s.index)
        last_pos, hold = 0.0, 0
        for i, t in enumerate(s.index):
            desired = float(s.iloc[i])
            if desired == last_pos:
                hold += 1
            else:
                if hold >= min_hold:
                    last_pos = desired
                    hold = 1
                else:
                
                    pass
            out.iloc[i] = last_pos
        return out




    def cv_arima(
        self,
        window_model: int = 400,
        order: tuple[int,int,int] = (1,0,1),
        n_splits: int = 5,
        min_train_size: int = 24*60,
    ):
        r = self.df["ret1"].astype(float)
        preds = pd.Series(index=r.index, dtype=float)

        for tr, te in self._ts_cv_splits(len(r), n_splits=n_splits, min_train_size=min_train_size):
            # идём по тестовым точкам, каждый раз обучаясь на предыдущем окне train+история
            for t in te:
                start = max(0, t - window_model)
                win = r.iloc[start:t].dropna()
                if len(win) < 10 or win.std(ddof=1) == 0: 
                    continue
                try:
                    fit = ARIMA(win, order=order).fit(method_kwargs={"warn_convergence": False})
                    preds.iloc[t] = float(fit.forecast(steps=1).iloc[0])
                except Exception:
                    continue

        valid = preds.dropna()
        sig = np.sign(valid).astype(float).rename("ARIMA_CV")
        eq  = self.backtest(self.df["close"].loc[valid.index], sig)
        rep = self.metrics(eq).to_dict()
        return sig, pd.Series(rep)




    def gs_logreg_by_sharpe(
        self,
        C_grid=(0.1, 0.5, 1.0, 2.0, 5.0),
        penalties=("l1","l2"),
        features: list[str] | None = None,
        n_splits: int = 5,
        min_train_size: int = 24*60,
    ):
        best = (None, None, -np.inf, None)  
        for C in C_grid:
            for pen in penalties:
                _, rep = self.cv_logreg(
                    features=features, C=C, penalty=pen,
                    n_splits=n_splits, min_train_size=min_train_size, search_threshold=True
                )
                score = rep["Sharpe_OOF"] if rep["Sharpe_OOF"]==rep["Sharpe_OOF"] else -np.inf
                if score > best[2]:
                    best = (C, pen, score, rep)
        return {"best_C": best[0], "best_penalty": best[1], "Sharpe_OOF": best[2], "report": best[3]}



    # ---------- ensemble ----------

    def _rolling_edge(self, price: pd.Series, signal: pd.Series, 
                  window: int = 24*14) -> pd.Series:
        """
        Скользящая эффективность сигнала: IR = mean / std по стратегии (shift(1), без утечек).
        """
        ret = price.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        strat_ret = signal.shift(1).fillna(0.0) * ret

       
        alpha = getattr(self, "_edge_alpha", None)
        if alpha is not None and 0 < alpha < 1:
            mu = strat_ret.ewm(alpha=alpha, adjust=False, min_periods=max(5, int(1/alpha))).mean()
            sd = strat_ret.ewm(alpha=alpha, adjust=False, min_periods=max(5, int(1/alpha))).std(bias=False)
        else:
            mp = max(10, window // 4)
            mu = strat_ret.rolling(window, min_periods=mp).mean()
            sd = strat_ret.rolling(window, min_periods=mp).std()

        ir = mu / sd.replace(0.0, np.nan)
        return ir.fillna(0.0).clip(lower=-10, upper=10)

    @staticmethod
    # ensemble with equal weights
    def ensemble(signals: Dict[str, pd.Series], clip: bool = True) -> pd.Series:
      
        def _align_sum(sigdict):
            idx = None
            for s in sigdict.values():
                idx = s.index if idx is None else idx.union(s.index)
            fused = pd.Series(0.0, index=idx)
            for s in sigdict.values():
                fused = fused.add(s.reindex(idx).fillna(0.0), fill_value=0.0)
            return fused, idx

        def _min_hold_on_sign(x: pd.Series, min_hold: int) -> pd.Series:
            sign = np.sign(x).astype(float)
            out = pd.Series(0.0, index=sign.index)
            last, hold = 0.0, 0
            for i in range(len(sign)):
                desired = float(sign.iloc[i])
                if desired == last:
                    hold += 1
                else:
                    if hold >= min_hold:
                        last, hold = desired, 1
                out.iloc[i] = last
            return out

        fused, idx = _align_sum(signals)
       
        fused = fused / max(1, len(signals))

       
        alpha = 0.05   
        sm = fused.ewm(alpha=alpha, adjust=False).mean()

       
        band = 0.25
        sm_q = sm.where(sm.abs() > band, 0.0)

       
        min_hold = 24
        sign_hold = _min_hold_on_sign(sm_q, min_hold=min_hold)
        out = (sign_hold * sm_q.abs()).clip(-1, 1)

        return out if not clip else out.clip(-1, 1)
    

    def ensemble_vote_rolling(self,
                          price: pd.Series,
                          signals: dict[str, pd.Series],
                          window: int = 24*14,
                          min_edge: float = 0.0) -> pd.Series:
        """
        Majority vote среди моделей
        """
        idx = price.index
        aligned = {k: v.reindex(idx).fillna(0.0) for k, v in signals.items()}
        edges = {k: self._rolling_edge(price, v, window=window) for k, v in aligned.items()}

        votes = pd.Series(0.0, index=idx)
        for t in range(len(idx)):
            active_votes = []
            for k, sig in aligned.items():
                if edges[k].iloc[t] > min_edge:
                    active_votes.append(np.sign(sig.iloc[t]))
            votes.iloc[t] = np.sign(np.sum(active_votes)) if active_votes else 0.0

        
        min_hold = getattr(self, "_ens_vote_min_hold", 24)
        if min_hold > 0:
            
            out = pd.Series(0.0, index=votes.index)
            last, hold = 0.0, 0
            for i in range(len(votes)):
                desired = float(votes.iloc[i])
                if desired == last:
                    hold += 1
                else:
                    if hold >= min_hold:
                        last, hold = desired, 1
                    
                out.iloc[i] = last
            votes = out

        return votes.rename("ENS_VOTE")


    def ensemble_weighted(self,
                        price: pd.Series,
                        signals: dict[str, pd.Series],
                        window: int = 24*14,
                        method: str = "edge_softmax",
                        temperature: float = 0.05) -> pd.Series:
        """
        Веса только для моделей с положительным edge (IR>0), остальные = 0.
        """
        idx = price.index
        aligned = {k: v.reindex(idx).fillna(0.0) for k, v in signals.items()}
        edg = {k: self._rolling_edge(price, v, window=window) for k, v in aligned.items()}

        keys = list(aligned.keys())
        ens_raw = pd.Series(0.0, index=idx)

        min_edge = getattr(self, "_ens_min_edge", 0.0)  # можно поднять до 0.2..0.5

        for t in range(len(idx)):
            e = np.array([edg[k].iloc[t] for k in keys], dtype=float)
            S = np.array([aligned[k].iloc[t] for k in keys], dtype=float)

            # маска только «хороших» на этот момент
            m = e > min_edge
            if not np.any(m):
                ens_raw.iloc[t] = 0.0
                continue
            e = e.copy(); S = S.copy()
            e[~m] = -np.inf if method == "edge_softmax" else 0.0
            S[~m] = 0.0

            if method == "edge_linear":
                w = np.maximum(e, 0.0)  # уже отфильтрованы по min_edge
                w = w / w.sum() if w.sum() > 0 else np.zeros_like(w)
            elif method == "edge_softmax":
                z = e / max(1e-6, temperature)
                z -= np.nanmax(z)
                w = np.exp(z)
                w[~np.isfinite(w)] = 0.0
                w = w / w.sum() if w.sum() > 0 else np.zeros_like(w)
            else:
                raise ValueError("unknown method")

            ens_raw.iloc[t] = float(np.dot(w, S))

      
        alpha = getattr(self, "_ens_alpha", 0.05)
        band  = getattr(self, "_ens_band", 0.25)
        min_hold = getattr(self, "_ens_min_hold", 24)

        sm = ens_raw.ewm(alpha=alpha, adjust=False).mean()
        sm_q = sm.where(sm.abs() > band, 0.0)

        if min_hold > 0:
            out = pd.Series(0.0, index=sm_q.index)
            last, hold = 0.0, 0
            for i in range(len(sm_q)):
                desired = float(np.sign(sm_q.iloc[i]))
                if desired == last:
                    hold += 1
                else:
                    if hold >= min_hold:
                        last, hold = desired, 1
                  
                out.iloc[i] = last
            ens = (out * sm_q.abs()).clip(-1, 1)
        else:
            ens = sm_q.clip(-1, 1)

        return ens.rename("ENS_WEIGHTED")




    def ensemble_vote_filtered(
        self,
        price: pd.Series,
        signals: dict[str, pd.Series],
        window: int = 24*14,
        min_edge: float = 0.0,
    ) -> pd.Series:
        """
        Голосуют только модели, у которых edge > min_edge на протяжении последних K баров.
        """
        idx = price.index
        aligned = {k: v.reindex(idx).fillna(0.0) for k, v in signals.items()}
        edges = {k: self._rolling_edge(price, v, window=window) for k, v in aligned.items()}

        K = getattr(self, "_ens_edge_persist", 3) 
        votes = pd.Series(0.0, index=idx)

        
        ok = {}
        for k, e in edges.items():
            over = (e > min_edge).astype(int)
            ok[k] = over.rolling(K, min_periods=K).sum() == K

        for t in range(len(idx)):
            active = []
            for k, sig in aligned.items():
                if ok[k].iloc[t]:
                    active.append(np.sign(sig.iloc[t]))
            votes.iloc[t] = np.sign(np.sum(active)) if active else 0.0

        min_hold = getattr(self, "_ens_vote_min_hold", 24)
        if min_hold > 0:
            out = pd.Series(0.0, index=votes.index)
            last, hold = 0.0, 0
            for i in range(len(votes)):
                desired = float(votes.iloc[i])
                if desired == last:
                    hold += 1
                else:
                    if hold >= min_hold:
                        last, hold = desired, 1
                out.iloc[i] = last
            votes = out

        return votes.rename("ENS_VOTE_FILT")




    



