"""
Daily prospective snapshot for longitudinal calibration.

Runs every day via cron. For each active market tracked:
  - Records current YES-token price
  - Records timestamp
  - Appends to data/prospective/{market_id}.jsonl

When a market resolves (outcomePrices[0] reaches 0 or 1), marks it resolved.
After 180 days of tracking, each market provides T-7/30/90/180 calibration pairs.

Usage: python3 daily_snapshot.py
Cron:  0 9 * * * cd /path/to/prediction-market-efficiency && python3 data/daily_snapshot.py >> data/daily_snapshot.log 2>&1
"""
from __future__ import annotations

import json
import time
import requests
import datetime
from pathlib import Path

DATA_FILE  = Path(__file__).parent / "raw" / "markets_20260401_215624.json"
SNAP_DIR   = Path(__file__).parent / "prospective"
INDEX_FILE = Path(__file__).parent / "prospective_index.json"
GAMMA_API  = "https://gamma-api.polymarket.com"
N_TRACK    = 300   # number of active markets to track
SNAP_DIR.mkdir(exist_ok=True)


def get_token(m: dict) -> str | None:
    raw = m.get("clobTokenIds")
    if not raw: return None
    try:
        toks = json.loads(raw) if isinstance(raw, str) else raw
        return str(toks[0]) if isinstance(toks, list) and toks else None
    except Exception:
        return None


def fetch_current_price(token: str) -> float | None:
    try:
        r = requests.get("https://clob.polymarket.com/book",
                         params={"token_id": token}, timeout=10)
        r.raise_for_status()
        book = r.json()
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        if bids and asks:
            return (float(bids[0]["price"]) + float(asks[0]["price"])) / 2
        elif bids:
            return float(bids[0]["price"])
        elif asks:
            return float(asks[0]["price"])
    except Exception:
        return None


def load_or_init_index() -> dict:
    if INDEX_FILE.exists():
        return json.loads(INDEX_FILE.read_text())
    # First run: select markets to track
    data = json.loads(DATA_FILE.read_text())
    active = sorted(
        [m for m in data if not m.get("closed") and get_token(m)
         and float(m.get("volume") or 0) >= 50_000],
        key=lambda m: -float(m.get("volume") or 0),
    )[:N_TRACK]

    index = {}
    for m in active:
        mid = m.get("id") or m.get("conditionId")
        token = get_token(m)
        op_raw = m.get("outcomePrices","[]")
        try:
            op = json.loads(op_raw) if isinstance(op_raw, str) else op_raw
            yes_init = float(op[0]) if op else None
        except Exception:
            yes_init = None
        index[mid] = {
            "question":   m.get("question","")[:100],
            "token":      token,
            "volume":     float(m.get("volume") or 0),
            "start_date": datetime.date.today().isoformat(),
            "resolved":   False,
            "outcome":    None,
            "yes_init":   yes_init,
            "end_date":   m.get("endDateIso") or m.get("endDate"),
        }
    INDEX_FILE.write_text(json.dumps(index, indent=2))
    print(f"Initialized tracking for {len(index)} markets")
    return index


def update_market_resolution(mid: str, meta: dict) -> bool:
    """Check if market resolved since last snapshot. Returns True if newly resolved."""
    try:
        r = requests.get(f"{GAMMA_API}/markets/{mid}", timeout=10)
        r.raise_for_status()
        m = r.json()
        if m.get("closed"):
            op_raw = m.get("outcomePrices","[]")
            op = json.loads(op_raw) if isinstance(op_raw, str) else op_raw
            if op:
                yes_final = float(op[0])
                if yes_final >= 0.95:
                    meta["resolved"] = True
                    meta["outcome"]  = 1
                    meta["resolve_date"] = datetime.date.today().isoformat()
                    return True
                elif yes_final <= 0.05:
                    meta["resolved"] = True
                    meta["outcome"]  = 0
                    meta["resolve_date"] = datetime.date.today().isoformat()
                    return True
    except Exception:
        pass
    return False


def main():
    today = datetime.date.today().isoformat()
    now_ts = int(datetime.datetime.now().timestamp())
    print(f"[{today}] Daily snapshot starting...")

    index = load_or_init_index()
    active = {mid: meta for mid, meta in index.items() if not meta.get("resolved")}

    prices_recorded = 0
    newly_resolved  = 0
    errors          = 0

    for mid, meta in active.items():
        token = meta.get("token")
        if not token: continue

        # Check if resolved
        if update_market_resolution(mid, meta):
            newly_resolved += 1
            time.sleep(0.2)
            continue

        # Record current price
        price = fetch_current_price(token)
        if price is None:
            errors += 1
            time.sleep(0.1)
            continue

        snap_path = SNAP_DIR / f"{mid}.jsonl"
        with open(snap_path, "a") as f:
            f.write(json.dumps({"t": now_ts, "date": today, "p": round(price, 4)}) + "\n")

        prices_recorded += 1
        time.sleep(0.15)

    # Save updated index
    INDEX_FILE.write_text(json.dumps(index, indent=2))

    total_resolved = sum(1 for m in index.values() if m.get("resolved"))
    print(f"[{today}] Done: {prices_recorded} prices, {newly_resolved} new resolutions, "
          f"{errors} errors | {total_resolved}/{len(index)} total resolved")


if __name__ == "__main__":
    main()
