"""
Collect hourly CLOB price history for a sample of active markets.
Used for: information half-life estimation (Section 4.3 of paper).

Strategy:
  - Take top-200 active markets by volume
  - Fetch full price history (interval=max, fidelity=60 = hourly)
  - Save to data/raw/clob_history_{market_id}.json
"""
from __future__ import annotations
import json, time, requests
from pathlib import Path

CLOB_API  = "https://clob.polymarket.com"
DATA_FILE = Path(__file__).parent / "raw" / "markets_20260401_215624.json"
OUT_DIR   = Path(__file__).parent / "raw" / "clob_history"
OUT_DIR.mkdir(exist_ok=True)
N_MARKETS = 200


def get_token(m: dict) -> str | None:
    raw = m.get("clobTokenIds")
    if not raw: return None
    try:
        toks = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(toks, list) and toks:
            return str(toks[0])
    except Exception:
        return None


def fetch_history(token: str) -> list[dict] | None:
    try:
        r = requests.get(f"{CLOB_API}/prices-history",
                         params={"market": token, "interval": "max", "fidelity": 60},
                         timeout=20)
        r.raise_for_status()
        return r.json().get("history", [])
    except Exception as e:
        return None


def main():
    print("Loading markets...")
    data = json.loads(DATA_FILE.read_text())

    # Top-N active markets by volume
    active = sorted(
        [m for m in data if not m.get("closed") and get_token(m)],
        key=lambda m: float(m.get("volume") or 0),
        reverse=True,
    )[:N_MARKETS]

    print(f"Fetching CLOB history for {len(active)} markets...")
    results = {}
    errors  = 0

    for i, m in enumerate(active):
        mid   = m.get("id") or m.get("conditionId")
        token = get_token(m)
        out   = OUT_DIR / f"{mid}.json"

        if out.exists():
            history = json.loads(out.read_text())
        else:
            history = fetch_history(token)
            if history is None:
                errors += 1
                continue
            out.write_text(json.dumps(history))
            time.sleep(0.25)

        if len(history) >= 10:
            results[mid] = {
                "question":  m["question"],
                "volume":    float(m.get("volume") or 0),
                "n_points":  len(history),
                "first_t":   history[0]["t"]  if history else None,
                "last_t":    history[-1]["t"]  if history else None,
                "first_p":   history[0]["p"]   if history else None,
                "last_p":    history[-1]["p"]   if history else None,
            }

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(active)} done ({errors} errors)")

    print(f"\nDone: {len(results)} markets with >=10 price points, {errors} errors")
    out_path = Path(__file__).parent / "clob_history_index.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Index saved → {out_path}")
    return results


if __name__ == "__main__":
    main()
