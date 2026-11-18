"""
Retrospective longitudinal calibration collection.

For each resolved market:
  1. Fetch full daily CLOB price history
  2. Extract prices at T-7, T-14, T-30, T-60, T-90, T-180 days before resolution
  3. Resolution outcome from lastTradePrice (1=YES, 0=NO)

Output: data/longitudinal_calibration.json
"""
from __future__ import annotations

import json
import time
import requests
import datetime
from pathlib import Path
from collections import defaultdict

DATA_FILE  = Path(__file__).parent / "raw" / "markets_20260401_215624.json"
HIST_DIR   = Path(__file__).parent / "raw" / "clob_history_resolved"
OUT_FILE   = Path(__file__).parent / "longitudinal_calibration.json"
HIST_DIR.mkdir(exist_ok=True)

HORIZONS   = [7, 14, 30, 60, 90, 180]   # days before resolution
N_SAMPLE   = 500                          # total markets to sample
CLOB_API   = "https://clob.polymarket.com"


def get_token(m: dict) -> str | None:
    raw = m.get("clobTokenIds")
    if not raw: return None
    try:
        toks = json.loads(raw) if isinstance(raw, str) else raw
        return str(toks[0]) if isinstance(toks, list) and toks else None
    except Exception:
        return None


def fetch_daily_history(token: str) -> list[dict]:
    r = requests.get(f"{CLOB_API}/prices-history",
                     params={"market": token, "interval": "max", "fidelity": 1440},
                     timeout=25)
    r.raise_for_status()
    return r.json().get("history", [])


def price_at_horizon(history: list[dict], t_resolve: int, days: int) -> float | None:
    """Return price closest to (t_resolve - days*86400)."""
    target = t_resolve - days * 86400
    if not history or history[0]["t"] > target:
        return None  # history doesn't reach that far back
    best = None
    best_dt = float("inf")
    for h in history:
        dt = abs(h["t"] - target)
        if dt < best_dt:
            best_dt = dt
            best = h["p"]
    return best


def load_resolved_markets() -> tuple[list[dict], list[dict]]:
    """Return (yes_markets, no_markets) sorted by volume desc."""
    data   = json.loads(DATA_FILE.read_text())
    yes_ms, no_ms = [], []
    for m in data:
        if not m.get("closed"): continue
        token = get_token(m)
        if not token: continue
        try:
            vol = float(m.get("volume") or 0)
            # Use outcomePrices[0] = YES token final settlement price
            op_raw = m.get("outcomePrices", "[]")
            op = json.loads(op_raw) if isinstance(op_raw, str) else op_raw
            yes_final = float(op[0]) if op else -1
        except Exception:
            continue
        if vol < 10_000: continue
        if yes_final >= 0.95:
            yes_ms.append({**m, "_outcome": 1, "_token": token})
        elif yes_final <= 0.05:
            no_ms.append({**m, "_outcome": 0, "_token": token})

    yes_ms.sort(key=lambda x: -float(x.get("volume", 0)))
    no_ms.sort(key=lambda x:  -float(x.get("volume", 0)))
    print(f"Resolved YES: {len(yes_ms):,}  |  Resolved NO: {len(no_ms):,}")
    return yes_ms, no_ms


def process_market(m: dict) -> dict | None:
    mid     = m.get("id") or m.get("conditionId")
    token   = m["_token"]
    outcome = m["_outcome"]
    hist_path = HIST_DIR / f"{mid}.json"

    # Load or fetch
    if hist_path.exists():
        history = json.loads(hist_path.read_text())
    else:
        try:
            history = fetch_daily_history(token)
            hist_path.write_text(json.dumps(history))
            time.sleep(0.3)
        except Exception as e:
            return None

    if len(history) < 10:
        return None

    # Resolution timestamp = last data point
    t_resolve = history[-1]["t"]
    span_days = (t_resolve - history[0]["t"]) / 86400

    record = {
        "id":         mid,
        "question":   m.get("question", "")[:100],
        "volume":     float(m.get("volume") or 0),
        "outcome":    outcome,
        "t_resolve":  t_resolve,
        "span_days":  round(span_days, 1),
        "n_points":   len(history),
        "predictions": {},
    }

    for days in HORIZONS:
        p = price_at_horizon(history, t_resolve, days)
        if p is not None and 0.001 < p < 0.999:
            record["predictions"][str(days)] = round(p, 4)

    # Only keep if at least one horizon is available
    if not record["predictions"]:
        return None

    return record


def main():
    print("Loading resolved markets...")
    yes_ms, no_ms = load_resolved_markets()

    # Balanced sample: half YES, half NO
    half = N_SAMPLE // 2
    sample = yes_ms[:half] + no_ms[:half]
    print(f"Sampling {len(sample)} markets ({half} YES + {half} NO)")

    records = []
    errors  = 0

    for i, m in enumerate(sample):
        rec = process_market(m)
        if rec:
            records.append(rec)
        else:
            errors += 1

        if (i + 1) % 50 == 0:
            pct = (i + 1) / len(sample) * 100
            print(f"  {i+1}/{len(sample)} ({pct:.0f}%) — {len(records)} usable, {errors} errors")

    print(f"\nDone: {len(records)} usable markets, {errors} failures")

    # Summary by horizon
    for days in HORIZONS:
        k = str(days)
        n = sum(1 for r in records if k in r["predictions"])
        print(f"  T-{days:3d}d:  {n:3d} markets with prediction")

    OUT_FILE.write_text(json.dumps(records, indent=2))
    print(f"\nSaved → {OUT_FILE}")
    return records


if __name__ == "__main__":
    main()
