"""
Collect resolved binary markets from Manifold Markets for calibration comparison.
API: https://api.manifold.markets/v0/search-markets

Fetches resolved BINARY markets, extracts prediction probability and outcome.
"""
from __future__ import annotations
import json, time, requests
from pathlib import Path

API_BASE = "https://api.manifold.markets/v0"
OUT_FILE = Path(__file__).parent / "manifold_resolved.json"
N_BATCHES = 30   # 100 per batch = up to 3000 questions


def fetch_batch(before_id: str | None = None) -> list[dict]:
    params = {
        "limit":        100,
        "filter":       "resolved",
        "contractType": "BINARY",
        "sort":         "score",
    }
    if before_id:
        params["before"] = before_id
    r = requests.get(f"{API_BASE}/search-markets", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def extract(m: dict) -> dict | None:
    try:
        res = m.get("resolution")
        if res not in ("YES", "NO"):
            return None
        pred = m.get("resolutionProbability") or m.get("probability")
        if pred is None:
            return None
        return {
            "id":         m["id"],
            "question":   m.get("question", ""),
            "resolution": 1 if res == "YES" else 0,
            "prediction": float(pred),
            "volume":     float(m.get("volume") or 0),
            "n_traders":  m.get("uniqueBettorCount", 0),
            "close_time": m.get("closeTime"),
        }
    except Exception:
        return None


def main():
    questions = []
    last_id   = None
    print("Fetching Manifold resolved binary markets...")

    for batch_n in range(N_BATCHES):
        try:
            markets = fetch_batch(last_id)
        except Exception as e:
            print(f"  Error on batch {batch_n}: {e}")
            break
        if not markets:
            print("  No more results.")
            break

        for m in markets:
            ex = extract(m)
            if ex:
                questions.append(ex)

        last_id = markets[-1]["id"]
        print(f"  Batch {batch_n+1}: {len(markets)} fetched → {len(questions)} usable total")
        time.sleep(0.3)

    print(f"\nTotal usable: {len(questions)}")
    OUT_FILE.write_text(json.dumps(questions, indent=2))
    print(f"Saved → {OUT_FILE}")
    return questions


if __name__ == "__main__":
    main()
