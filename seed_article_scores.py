#!/usr/bin/env python3
"""
seed_article_scores.py — Phase E, component 1.

Seeds the local Path B corpus (score_cache_llm.json, the 30 scored eval atoms) into
the Firestore `article_scores` collection, per the Phase E data-design spec (§3/§4).
This is the one-time "move the local cache into the cloud" step.

Each corpus row = a scoring atom (from the cache) JOINED with per-article metadata
(canonical_topic_id, topic_role, published_at, source) from the eval article JSONs,
so the rows are complete and aggregatable by component 3.

AUTH: local gcloud Application Default Credentials (no key file). Run once first:
    gcloud auth application-default login
    (sign in with the Google account that owns the voycce-mvp Firebase project)

SAFETY: dry-run by DEFAULT — prints exactly what it would write, touches nothing.
Pass --commit to actually write. --commit writes to PRODUCTION Firestore (voycce-mvp);
the `article_scores` collection is server-only-write (Admin SDK bypasses rules).

Run from voycce-sentiment-api/. Imports evaluate_v1 to locate the eval article JSONs.
Claude does not run git — Roy commits.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import evaluate_v1 as base  # for LABELS_DIR (eval article JSON location)

PROJECT_ID_DEFAULT = "voycce-mvp"
COLLECTION = "article_scores"
CACHE_DEFAULT = Path(__file__).resolve().parent / "score_cache_llm.json"


# ---------------------------------------------------------------- metadata join
def load_article_metadata():
    """article_id -> {canonical_topic_id, aspect_text, topic_role, source, published_at}
    from the primary label of each eval article JSON."""
    meta = {}
    for f in sorted(base.LABELS_DIR.glob("article-*.json")):
        art = json.loads(f.read_text())
        primary = next((l for l in art.get("labels", []) if l.get("topicRole") == "primary"),
                       art["labels"][0] if art.get("labels") else None)
        if not primary:
            continue
        meta[art["articleId"]] = {
            "canonical_topic_id": primary.get("canonicalTopicId"),
            "aspect_text": primary.get("canonicalTopicTitle"),
            "topic_role": primary.get("topicRole", "primary"),
            "source": art.get("source"),
            "published_at": art.get("publishedAt"),
        }
    return meta


def parse_ts(s):
    """ISO8601 -> tz-aware datetime (Firestore stores as timestamp). None-safe."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def build_docs(cache, meta):
    """Join cache atoms + article metadata -> (doc_id, doc_dict) list. Warns on gaps."""
    now = datetime.now(timezone.utc)
    docs, warnings = [], []
    for cache_key, atom in cache.items():
        aid = atom["article_id"]
        m = meta.get(aid)
        if not m:
            warnings.append(f"no eval metadata for {aid} — skipped")
            continue
        if m["aspect_text"] and atom.get("aspect") and m["aspect_text"] != atom["aspect"]:
            warnings.append(f"{aid}: aspect mismatch cache={atom['aspect']!r} vs meta={m['aspect_text']!r}")
        doc_id = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
        docs.append((doc_id, {
            "cache_key":             cache_key,
            "content_hash":          cache_key.split("|", 1)[0],   # first segment = sha256(norm(text))
            "article_id":            aid,
            "canonical_topic_id":    m["canonical_topic_id"],
            "aspect_text":           atom.get("aspect"),
            "score":                 atom["score"],
            "confidence":            atom["confidence"],
            "reason":                atom["reason"],
            "evidence_spans":        atom.get("evidence_spans", []),
            "topic_role":            m["topic_role"],
            "is_canonical_original": True,          # dedup deferred (§13.3); default original
            "syndication_cluster_id": None,         # populated by later near-dup pass
            "is_eval_seed":          True,          # test fixture: placeholder canonical_topic_id
                                                    # (manual_topic_*), NOT real corpus. Component 3
                                                    # aggregation MUST filter `is_eval_seed != true`.
            "model_version":         atom["model_version"],
            "prompt_version":        atom["prompt_version"],
            "rubric_version":        atom["rubric_version"],
            "scorer_latency_ms":     atom.get("scorer_latency_ms"),
            "article_published_at":  parse_ts(m["published_at"]),
            "article_source":        m["source"],
            "scored_at":             now,
        }))
    return docs, warnings


# ---------------------------------------------------------------- firestore (commit only)
def commit_to_firestore(docs, project_id):
    import firebase_admin                       # lazy: dry-run needs no Firebase deps
    from firebase_admin import credentials, firestore
    if not firebase_admin._apps:
        # ADC + explicit projectId (required for gcloud end-user credentials)
        firebase_admin.initialize_app(credentials.ApplicationDefault(), {"projectId": project_id})
    db = firestore.client()

    batch = db.batch()                           # 30 docs << 500-op batch limit
    for doc_id, doc in docs:
        batch.set(db.collection(COLLECTION).document(doc_id), doc)  # set = idempotent upsert
    batch.commit()
    print(f"  committed {len(docs)} docs to {COLLECTION} (project {project_id})")

    # verification read-back (proves the write + our read path)
    first_id = docs[0][0]
    snap = db.collection(COLLECTION).document(first_id).get()
    if snap.exists:
        d = snap.to_dict()
        print(f"  verify: {first_id[:12]}… exists — "
              f"{d['article_id']} score={d['score']} topic={d['canonical_topic_id']}")
    else:
        print("  verify FAILED: read-back not found — investigate before trusting the write")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true",
                    help="actually write to Firestore (default: dry-run, writes nothing)")
    ap.add_argument("--cache", type=Path, default=CACHE_DEFAULT)
    ap.add_argument("--project", default=PROJECT_ID_DEFAULT)
    args = ap.parse_args()

    if not args.cache.exists():
        sys.exit(f"cache not found: {args.cache}")
    cache = json.loads(args.cache.read_text())
    meta = load_article_metadata()
    docs, warnings = build_docs(cache, meta)

    print(f"corpus: {len(cache)} cached atoms -> {len(docs)} article_scores docs "
          f"(project {args.project}, collection {COLLECTION})")
    for w in warnings:
        print(f"  WARN: {w}")
    if not docs:
        sys.exit("nothing to write.")

    if not args.commit:
        print("\n--- DRY RUN (no writes). Sample doc: ---")
        sample_id, sample = docs[0]
        printable = {**sample,
                     "article_published_at": str(sample["article_published_at"]),
                     "scored_at": str(sample["scored_at"])}
        print(f"doc id: {sample_id}")
        print(json.dumps(printable, indent=2, ensure_ascii=False))
        print(f"\n... and {len(docs)-1} more. Doc IDs:")
        for did, d in docs:
            print(f"  {did[:16]}…  {d['article_id']:<14} score={d['score']:+.2f} "
                  f"topic={d['canonical_topic_id']}")
        print(f"\n{len(docs)} docs ready. Re-run with --commit to write to Firestore.")
        return

    print("\n--- COMMIT: writing to PRODUCTION Firestore ---")
    commit_to_firestore(docs, args.project)


if __name__ == "__main__":
    main()
