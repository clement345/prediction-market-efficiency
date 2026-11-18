"""
Data collection script — Polymarket markets + CLOB prices
Section 3.1 of the paper
"""
from __future__ import annotations

import json
import time
import requests
import psycopg2
from datetime import datetime
from pathlib import Path

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"
DB_URL    = "postgresql://clement@localhost:5432/polymarket"

OUTPUT_DIR = Path(__file__).parent / "raw"
OUTPUT_DIR.mkdir(exist_ok=True)


def fetch_all_markets(min_volume: float = 10_000) -> list[dict]:
    """Fetch all markets above volume threshold (paginated)."""
    markets = []
    offset  = 0
    limit   = 500

    while True:
        r = requests.get(
            f"{GAMMA_API}/markets",
            params={"limit": limit, "offset": offset, "volume_num_min": min_volume},
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        markets.extend(batch)
        offset += limit
        print(f"  fetched {len(markets)} markets...")
        time.sleep(0.3)

    return markets


def fetch_clob_price(token_id: str) -> dict | None:
    """Fetch current best bid/ask for a given token."""
    try:
        r = requests.get(
            f"{CLOB_API}/book",
            params={"token_id": token_id},
            timeout=10,
        )
        r.raise_for_status()
        book = r.json()
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        best_bid = float(bids[0]["price"]) if bids else None
        best_ask = float(asks[0]["price"]) if asks else None
        mid      = (best_bid + best_ask) / 2 if best_bid and best_ask else None
        return {"best_bid": best_bid, "best_ask": best_ask, "mid": mid}
    except Exception:
        return None


def save_snapshot(markets: list[dict], timestamp: str) -> None:
    """Save market snapshot to JSON."""
    path = OUTPUT_DIR / f"markets_{timestamp}.json"
    path.write_text(json.dumps(markets, indent=2))
    print(f"Saved {len(markets)} markets to {path}")


if __name__ == "__main__":
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    print(f"Fetching Polymarket data at {ts}...")
    markets = fetch_all_markets(min_volume=10_000)
    save_snapshot(markets, ts)
    print(f"Done. {len(markets)} markets collected.")
