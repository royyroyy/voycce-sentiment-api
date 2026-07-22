#!/usr/bin/env python3
"""
aggregate_timeseries.py — Phase E component 3: build the sentiment time series.

Aggregates `article_scores` into `topic_sentiment_timeseries`: one doc per
(story_id, date) with the mean sentiment, volume, and the contributing article
ids for drill-down. This is the artifact the UI reads to draw a story's line.

Per spec stage1a-phaseE-data-design.md §5 (Invariant 9, revised): the time series
is a MATERIALIZED, recomputable aggregation over article_scores (the source of
truth) — NOT itself the system of record. Safe to drop and rebuild anytime.

AGGREGATION (agg-v1, decided 2026-07-21):
  - mean_score = simple unweighted mean (every article counts equally). Primary metric.
  - article_count = volume (distinguishes "sentiment moved" from "coverage spiked").
  - contributing_article_ids / _score_doc_ids = the drill-down join (non-negotiable).
  - confidence data is preserved on the atoms, so a confidence-weighted mean can be
    added later without re-running (see Notion; UI-design decision). Not computed here.
  - MUST exclude is_eval_seed == true (the 30 eval rows must never pollute a real line).

Bucketing: by article_published_at date (UTC). Falls back to scored_at if an atom
lacks a published date. Only current (model_version, prompt_version) atoms contribute.

SAFETY: dry-run by DEFAULT — prints the buckets it WOULD write, writes nothing.
--commit writes. Idempotent: doc id = {story_id}_{YYYY-MM-DD}, set overwrites, so a
rebuild is clean. --rebuild deletes existing timeseries docs first (full recompute).

AUTH: gcloud ADC (Firestore). No Claude, no Voyage — pure aggregation over stored data.
Run from voycce-sentiment-api/ with .venv active. Roy commits; Claude does not run git.
"""
import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone

import firebase_admin
from firebase_admin import credentials, firestore

PROJECT_ID = "voycce-mvp"
SRC = "article_scores"
DST = "topic_sentiment_timeseries"
AGG_VERSION = "agg-v1"
# only atoms from the current scorer version feed the live series
CUR_MODEL = "claude-sonnet-5"
CUR_PROMPT = "snp-v1"


def get_db():
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.ApplicationDefault(), {"projectId": PROJECT_ID})
    return firestore.client()


def atom_date(a, pub_date_by_doc):
    """Bucket date (UTC, YYYY-MM-DD). The scoring atom does NOT carry the publish
    date, so we join to the articles doc (via article_doc_id) to get published_at —
    the article is the source of truth for when it was published. Falls back to the
    atom's scored_at only if the article has no usable published_at."""
    pub = pub_date_by_doc.get(a.get("article_doc_id"))
    if pub:
        return pub
    v = a.get("scored_at")
    if v is not None:
        if isinstance(v, str):
            try:
                v = datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                return None
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc).strftime("%Y-%m-%d")
    return None


def _to_date_str(v):
    """Normalize a Firestore datetime / ISO string to UTC YYYY-MM-DD, or None."""
    if v is None:
        return None
    if isinstance(v, str):
        try:
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None
    if v.tzinfo is None:
        v = v.replace(tzinfo=timezone.utc)
    return v.astimezone(timezone.utc).strftime("%Y-%m-%d")


def load_publish_dates(db):
    """article doc id -> published_at date string. The date lives on the article
    (source of truth), not the score atom, so C3 reads it here."""
    out = {}
    for d in db.collection("articles").stream():
        out[d.id] = _to_date_str(d.to_dict().get("published_at"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="write buckets (default: dry-run)")
    ap.add_argument("--rebuild", action="store_true", help="delete existing timeseries docs first")
    args = ap.parse_args()

    db = get_db()

    # publish date lives on the article, not the score atom — join via article_doc_id
    pub_date_by_doc = load_publish_dates(db)

    # ---- gather atoms, excluding eval seeds and non-current versions ----
    buckets = defaultdict(list)   # (story_id, date) -> [atom-with-doc-id]
    total, skipped_eval, skipped_ver, skipped_nodate, skipped_nostory = 0, 0, 0, 0, 0
    for d in db.collection(SRC).stream():
        a = d.to_dict()
        total += 1
        if a.get("is_eval_seed") is True:
            skipped_eval += 1; continue
        if a.get("model_version") != CUR_MODEL or a.get("prompt_version") != CUR_PROMPT:
            skipped_ver += 1; continue
        sid = a.get("story_id")
        if not sid:
            skipped_nostory += 1; continue
        date = atom_date(a, pub_date_by_doc)
        if not date:
            skipped_nodate += 1; continue
        buckets[(sid, date)].append({**a, "_score_doc_id": d.id})

    if not buckets:
        sys.exit(f"No aggregatable atoms. (scanned {total}: {skipped_eval} eval, "
                 f"{skipped_ver} wrong-version, {skipped_nostory} no-story, {skipped_nodate} no-date)")

    # ---- compute buckets ----
    now = datetime.now(timezone.utc)
    rows = []
    for (sid, date), atoms in buckets.items():
        scores = [x["score"] for x in atoms]
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        rows.append({
            "doc_id": f"{sid}_{date}",
            "canonical_topic_id": sid,      # story is the time-series atom
            "story_id": sid,
            "date": date,
            "mean_score": round(mean, 4),
            "score_stddev": round(var ** 0.5, 4),
            "article_count": len(atoms),
            "contributing_article_ids": [x.get("article_doc_id") for x in atoms],
            "contributing_score_doc_ids": [x["_score_doc_id"] for x in atoms],
            "aggregation_version": AGG_VERSION,
            "model_version": CUR_MODEL,
            "prompt_version": CUR_PROMPT,
            "computed_at": now,
        })

    # ---- report ----
    multi_day = defaultdict(int)
    for r in rows:
        multi_day[r["story_id"]] += 1
    stories_with_movement = sum(1 for v in multi_day.values() if v > 1)
    print("=" * 68)
    print(f"scanned {total} atoms → {len(rows)} (story,date) buckets across {len(multi_day)} stories")
    print(f"  excluded: {skipped_eval} eval-seed, {skipped_ver} wrong-version, "
          f"{skipped_nodate} no-date, {skipped_nostory} no-story")
    print(f"  stories with >1 day of coverage (a line with MOVEMENT): {stories_with_movement}")
    print("=" * 68)

    # show the buckets for the biggest multi-day stories
    by_story = defaultdict(list)
    for r in rows:
        by_story[r["story_id"]].append(r)
    movers = sorted((kv for kv in by_story.items() if len(kv[1]) > 1),
                    key=lambda kv: -sum(x["article_count"] for x in kv[1]))
    if movers:
        print("\n--- stories with multi-day movement (the actual sentiment lines) ---")
        for sid, rs in movers[:6]:
            rs = sorted(rs, key=lambda x: x["date"])
            print(f"\n  {sid}")
            for r in rs:
                bar = "+" if r["mean_score"] >= 0 else "-"
                print(f"    {r['date']}  mean={r['mean_score']:+.2f}  n={r['article_count']}  {bar}")
    else:
        print("\n  (no multi-day stories yet — corpus is a snapshot; movement fills in over time)")

    if not args.commit:
        print(f"\nDRY RUN — nothing written. {len(rows)} buckets ready. Re-run with --commit.")
        return

    if args.rebuild:
        print("\n--rebuild: deleting existing timeseries docs…")
        deleted = 0
        for d in db.collection(DST).stream():
            d.reference.delete(); deleted += 1
        print(f"  deleted {deleted}.")

    print("\n--- COMMIT: writing buckets ---")
    n = 0
    for r in rows:
        doc = dict(r); doc.pop("doc_id")
        db.collection(DST).document(r["doc_id"]).set(doc)
        n += 1
        if n % 100 == 0:
            print(f"  … {n}/{len(rows)} written")
    print(f"  wrote {n} buckets to {DST}.")


if __name__ == "__main__":
    main()
