"""
src/features/elliott.py
════════════════════════
Elliott Wave detection + tính features cho RL agent.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


FIB_RATIOS = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 1.272, 1.618, 2.618]


@dataclass
class Pivot:
    idx:   int
    price: float
    kind:  str   # 'H' | 'L'


@dataclass
class WavePattern:
    pivots:      list[Pivot]
    direction:   str             # 'bullish' | 'bearish' | 'abc_bull' | 'abc_bear'
    pattern:     str             # 'impulse' | 'abc'
    fib_levels:  dict[str, float] = field(default_factory=dict)
    target:      float = 0.0    # Target price (161.8% extension)
    support:     float = 0.0
    resistance:  float = 0.0
    confidence:  float = 0.0

    @property
    def start_idx(self): return self.pivots[0].idx
    @property
    def end_idx(self):   return self.pivots[-1].idx


# ─── Pivot detection ─────────────────────────────────────────────────────

def _find_pivots(prices: np.ndarray, order: int = 5) -> list[Pivot]:
    pivots = []
    n = len(prices)
    for i in range(order, n - order):
        w = prices[i - order: i + order + 1]
        if prices[i] == w.max(): pivots.append(Pivot(i, float(prices[i]), 'H'))
        elif prices[i] == w.min(): pivots.append(Pivot(i, float(prices[i]), 'L'))
    return pivots


def _filter_alt(pivots: list[Pivot]) -> list[Pivot]:
    f: list[Pivot] = []
    for p in pivots:
        if not f:
            f.append(p)
        elif p.kind != f[-1].kind:
            f.append(p)
        elif p.kind == 'H' and p.price > f[-1].price:
            f[-1] = p
        elif p.kind == 'L' and p.price < f[-1].price:
            f[-1] = p
    return f


# ─── Fibonacci ────────────────────────────────────────────────────────────

def _fib_retrace(high: float, low: float) -> dict[str, float]:
    rng = high - low
    return {f"{int(r*100)}%": high - r * rng for r in [0.236, 0.382, 0.5, 0.618, 0.786]}


def _fib_extend(wave_start: float, wave_end: float, ref: float) -> float:
    """Target price = ref + 1.618 × wave_length (Wave 3 / Wave C target)."""
    return ref + 1.618 * abs(wave_end - wave_start)


# ─── Impulse wave validation (EWP rules) ─────────────────────────────────

def _impulse_confidence(pts: list[Pivot]) -> float:
    """
    Trả về confidence score 0–1 cho 5-sóng impulse.
    Rules:
      1. Wave 2 không vượt quá start của Wave 1
      2. Wave 3 không phải sóng ngắn nhất
      3. Wave 4 không overlap Wave 1
    """
    if len(pts) < 6: return 0.0
    kinds = [p.kind for p in pts]
    if kinds == ['L', 'H', 'L', 'H', 'L', 'H']:
        w = [p.price for p in pts]
        # Rule 1
        if w[2] <= w[0]: return 0.0
        # Rule 3
        if w[4] <= w[1]: return 0.0
        w1 = w[1] - w[0]; w2_ret = w[1] - w[2]
        w3 = w[3] - w[2]; w4_ret = w[3] - w[4]; w5 = w[5] - w[4]
        # Rule 2
        if w3 <= min(w1, w5) or w3 <= 0: return 0.0
        conf = 0.4
        if 0.38 <= w2_ret/(w1+1e-9) <= 0.79: conf += 0.2
        if 1.2 <= w3/(w1+1e-9) <= 3.0:        conf += 0.25
        if 0.23 <= w4_ret/(w3+1e-9) <= 0.62:  conf += 0.15
        return min(1.0, conf)
    elif kinds == ['H', 'L', 'H', 'L', 'H', 'L']:
        # Bearish impulse (mirror)
        w = [-p.price for p in pts]
        if w[2] <= w[0] or w[4] <= w[1]: return 0.0
        w1 = w[1]-w[0]; w3 = w[3]-w[2]; w5 = w[5]-w[4]
        if w3 <= min(w1, w5) or w3 <= 0: return 0.0
        conf = 0.4 + (0.3 if 1.2 <= w3/(w1+1e-9) <= 3.0 else 0.0)
        return min(1.0, conf)
    return 0.0


# ─── Pattern detection ────────────────────────────────────────────────────

def detect_patterns(pivots: list[Pivot]) -> list[WavePattern]:
    patterns: list[WavePattern] = []
    n = len(pivots)

    # 5-wave impulse
    for i in range(n - 5):
        pts  = pivots[i: i + 6]
        conf = _impulse_confidence(pts)
        if conf < 0.4: continue
        kinds = [p.kind for p in pts]
        if kinds[0] == 'L':
            direction = 'bullish'
            fib  = _fib_retrace(pts[1].price, pts[2].price)
            tgt  = _fib_extend(pts[2].price, pts[3].price, pts[4].price)
            sup  = pts[2].price; res = pts[3].price
        else:
            direction = 'bearish'
            fib  = _fib_retrace(pts[0].price, pts[1].price)
            tgt  = _fib_extend(pts[2].price, pts[3].price, pts[4].price)
            sup  = pts[3].price; res = pts[2].price
        patterns.append(WavePattern(
            pivots=pts, direction=direction, pattern='impulse',
            fib_levels=fib, target=tgt, support=sup, resistance=res, confidence=conf))

    # ABC correction
    for i in range(n - 3):
        pts  = pivots[i: i + 4]
        kinds = [p.kind for p in pts]
        if kinds == ['H', 'L', 'H', 'L']:
            wA = pts[0].price - pts[1].price
            wC = pts[2].price - pts[3].price
            if wA <= 0 or wC <= 0: continue
            ratio = wC / (wA + 1e-9)
            conf  = 0.5 + (0.3 if 0.8 <= ratio <= 1.3 else 0.1 if 1.3 < ratio <= 1.8 else 0.0)
            fib   = _fib_retrace(pts[0].price, pts[1].price)
            patterns.append(WavePattern(
                pivots=pts, direction='abc_bear', pattern='abc',
                fib_levels=fib, target=pts[1].price, support=pts[3].price,
                resistance=pts[2].price, confidence=min(1.0, conf)))
        elif kinds == ['L', 'H', 'L', 'H']:
            wA = pts[1].price - pts[0].price
            wC = pts[3].price - pts[2].price
            if wA <= 0 or wC <= 0: continue
            ratio = wC / (wA + 1e-9)
            conf  = 0.5 + (0.3 if 0.8 <= ratio <= 1.3 else 0.1 if 1.3 < ratio <= 1.8 else 0.0)
            fib   = _fib_retrace(pts[1].price, pts[0].price)
            patterns.append(WavePattern(
                pivots=pts, direction='abc_bull', pattern='abc',
                fib_levels=fib, target=pts[1].price, support=pts[2].price,
                resistance=pts[3].price, confidence=min(1.0, conf)))
    return patterns


# ─── Feature computation ──────────────────────────────────────────────────

def compute_features(df: pd.DataFrame, patterns: list[WavePattern],
                     pivots: list[Pivot]) -> pd.DataFrame:
    """
    Tính 9 Elliott features cho mỗi bar.
    Dùng pattern active tại thời điểm đó (confidence cao nhất).
    """
    n = len(df)
    c = df["close"].values

    wave_pos  = np.zeros(n, np.float32)
    wave_dir  = np.zeros(n, np.float32)
    fib618    = np.zeros(n, np.float32)
    fib382    = np.zeros(n, np.float32)
    sup_d     = np.zeros(n, np.float32)
    res_d     = np.zeros(n, np.float32)
    tgt_d     = np.zeros(n, np.float32)
    conf      = np.zeros(n, np.float32)
    signal    = np.zeros(n, np.float32)

    # Sắp xếp theo confidence giảm dần
    sorted_pats = sorted(patterns, key=lambda p: p.confidence, reverse=True)

    for i in range(n):
        price = c[i]
        # Tìm pattern active tốt nhất
        best = None
        for pat in sorted_pats:
            if pat.start_idx <= i <= pat.end_idx + 15:
                best = pat; break
        if best is None: continue

        pvs = best.pivots
        pos = 0
        for j, pv in enumerate(pvs):
            if i >= pv.idx: pos = j
        wave_pos[i] = pos / max(len(pvs) - 1, 1)

        dir_map = {'bullish': 1.0, 'bearish': -1.0, 'abc_bull': 0.5, 'abc_bear': -0.5}
        wave_dir[i] = dir_map.get(best.direction, 0.0)

        f618 = best.fib_levels.get("61%", best.fib_levels.get("62%", 0)) or list(best.fib_levels.values())[3] if len(best.fib_levels) >= 4 else price
        f382 = best.fib_levels.get("38%", best.fib_levels.get("39%", 0)) or list(best.fib_levels.values())[1] if len(best.fib_levels) >= 2 else price

        fib618[i] = np.clip((price - f618) / (price + 1e-9), -0.2, 0.2)
        fib382[i] = np.clip((price - f382) / (price + 1e-9), -0.2, 0.2)

        if best.support > 0:
            sup_d[i] = np.clip((price - best.support) / (price + 1e-9), -0.2, 0.2)
        if best.resistance > 0:
            res_d[i] = np.clip((best.resistance - price) / (price + 1e-9), -0.2, 0.2)
        if best.target > 0:
            tgt_d[i] = np.clip((best.target - price) / (price + 1e-9), -0.5, 0.5)
        conf[i] = best.confidence

        # Composite signal
        s = 0.0
        if best.direction == 'bullish':
            if wave_pos[i] > 0.6 and conf[i] > 0.5: s += 0.4  # Wave 4 → expect wave 5
            if wave_pos[i] > 0.9: s -= 0.3           # end of wave 5 → sell
        elif best.direction == 'bearish':
            if wave_pos[i] > 0.6: s -= 0.4
        elif best.direction == 'abc_bull':
            if wave_pos[i] > 0.85: s += 0.5          # End of C → reversal
        elif best.direction == 'abc_bear':
            if wave_pos[i] > 0.85: s -= 0.5
        signal[i] = np.clip(s, -1, 1)

    df = df.copy()
    df["wave_position"]  = wave_pos
    df["wave_direction"] = wave_dir
    df["fib_dist_618"]   = fib618
    df["fib_dist_382"]   = fib382
    df["support_dist"]   = sup_d
    df["resistance_dist"]= res_d
    df["target_dist"]    = tgt_d
    df["pattern_conf"]   = conf
    df["elliott_signal"] = signal
    return df


def run_pipeline(df: pd.DataFrame,
                 pivot_order: int = 5) -> tuple[pd.DataFrame, list[WavePattern], list[Pivot]]:
    """Chạy toàn bộ Elliott pipeline."""
    prices  = df["close"].values
    raw_pvs = _find_pivots(prices, order=pivot_order)
    pivots  = _filter_alt(raw_pvs)
    pats    = detect_patterns(pivots)
    df_out  = compute_features(df, pats, pivots)
    n_imp   = sum(1 for p in pats if p.pattern == 'impulse')
    n_abc   = sum(1 for p in pats if p.pattern == 'abc')
    print(f"[Elliott] Pivots={len(pivots)} | Impulse={n_imp} | ABC={n_abc}")
    return df_out, pats, pivots
