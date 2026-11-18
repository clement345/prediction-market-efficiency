"""
Collect resolved Metaculus questions for calibration comparison.
API: https://www.metaculus.com/api2/questions/

Fetches binary questions (type=forecast) that are resolved,
with community predictions and actual outcomes.
"""
from __future__ import annotations
import json, time, requests
from pathlib import Path

API_BASE = "https://www.metaculus.com/api2"
OUT_FILE = Path(__file__).parent / "metaculus_resolved.json"
N_PAGES  = 20   # 100 per page = up to 2000 questions


def fetch_page(offset: int, limit: int = 100) -> dict:
    r = requests.get(
        f"{API_BASE}/questions/",
        params={
            "format":       "json",
            "status":       "resolved",
            "type":         "forecast",
            "limit":        limit,
            "offset":       offset,
            "order_by":     "-votes",
        },
        headers={"Accept": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def extract_question(q: dict) -> dict | None:
    """Extract calibration-relevant fields from a Metaculus question."""
    try:
        res = q.get("resolution")
        if res not in (0, 1):   # binary: 0=No, 1=Yes
            return None

        # Community prediction at close (best available)
        cp = q.get("community_prediction", {})
        pred = None
        for key in ("full", "unweighted"):
            full = cp.get(key, {})
            if full:
                pred = full.get("q2") or full.get("mean")
                if pred is not None:
                    break

        if pred is None:
            return None

        return {
            "id":         q["id"],
            "title":      q.get("title", ""),
            "resolution": int(res),
            "prediction": float(pred),
            "n_forecasters": q.get("number_of_predictions", 0),
            "close_time": q.get("close_time"),
            "resolve_time": q.get("resolve_time"),
        }
    except Exception:
        return None


def main():
    questions = []
    print("Fetching Metaculus resolved questions...")

    for page in range(N_PAGES):
        offset = page * 100
        try:
            data = fetch_page(offset)
        except Exception as e:
            print(f"  Error on page {page}: {e}")
            break

        results = data.get("results", [])
        if not results:
            print(f"  No more results at offset {offset}")
            break

        for q in results:
            extracted = extract_question(q)
            if extracted:
                questions.append(extracted)

        print(f"  Page {page+1}: {len(results)} raw → {len(questions)} usable total")
        time.sleep(0.4)

        if not data.get("next"):
            break

    print(f"\nTotal usable questions: {len(questions)}")
    OUT_FILE.write_text(json.dumps(questions, indent=2))
    print(f"Saved → {OUT_FILE}")
    return questions


if __name__ == "__main__":
    main()
