#!/usr/bin/env python3
"""
score_corpus.py — Phase E component 2: score the real corpus.

Scores every article that has a story_id, measuring sentiment TOWARD ITS STORY
(aspect = the story's label). Writes scoring atoms to Firestore `article_scores`.
This is the Path B scorer (validated ρ=0.902 on eval) pointed at real corpus.

DESIGN (confirmed with Roy 2026-07-21):
  - aspect = the story's `label` (clean, specific, already on the story doc).
    "How does this article frame THIS story?" — the narrative sentiment that rolls
    up into a story's time series.
  - EVERY article scored (not one per story): per-source framing differences are
    the product (drill-down: Fox vs Al Jazeera on the same story).
  - Singletons scored too: a single-article story may gain coverage later; scoring
    now avoids a backfill. One point today, a line when the story grows.

REUSES score_news_llm.py unchanged: call_claude, atom_from, cache_key, norm — same
model (claude-sonnet-5), same schema, same versioning, thinking disabled, no temperature.

article_scores doc = the scoring atom + provenance. Doc id = sha256(cache_key), so
re-scoring the same (text, aspect, prompt_version, model_version) is idempotent and
a version bump preserves old atoms (audit trail). content_hash on the atom is the
SCORER's text hash (cache_key's first segment) — distinct from the articles doc id
(which hashes headline+full_content); both are kept, they serve different joins.
article_scores also carries story_id + article doc id for the C3 aggregation join.

SAFETY: dry-run by DEFAULT — scores a small --limit sample and prints score/reason/
evidence, writes nothing. --commit scores all and writes. Skips already-cached atoms
(by doc id) so re-runs only score new articles. Filters is_eval_seed rows out by
writing to distinct doc ids (real corpus is never flagged is_eval_seed).

AUTH: ANTHROPIC_API_KEY (Claude) + gcloud ADC (Firestore).
Run from voycce-sentiment-api/ with .venv active. Roy commits; Claude does not run git.
"""
import argparse
import hashlib
import sys
from datetime import datetime, timezone

import firebase_admin
from firebase_admin import credentials, firestore

import score_news_llm as scorer  # reuse call_claude, atom_from, cache_key, norm, versions
import evaluate_v1 as base       # extract_lead (same lead extraction as the eval)

PROJECT_ID = "voycce-mvp"
COLLECTION = "article_scores"


def get_db():
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.ApplicationDefault(), {"projectId": PROJECT_ID})
    return firestore.client()


def build_text(article):
    """Same shape the eval scorer used: headline + ". " + lead(full_content).
    Keeps cache keys comparable to the eval and lean on tokens."""
    headline = article.get("headline", "") or ""
    body = article.get("full_content", "") or ""
    lead = base.extract_lead(body) if body else ""
    return f"{headline}. {lead}".strip()


def load_story_labels(db):
    """story_id -> label. The label is the aspect we score toward."""
    out = {}
    for s in db.collection("stories").stream():
        d = s.to_dict()
        out[d.get("story_id", s.id)] = d.get("label") or d.get("subject") or "the story"
    return out


def load_scoreable(db, story_labels):
    """Every article with a story_id, shaped for the scorer: {article_id, aspect, text}
    plus the article doc id and story_id for the write."""
    items = []
    for d in db.collection("articles").stream():
        a = d.to_dict()
        sid = a.get("story_id")
        if not sid:
            continue
        label = story_labels.get(sid)
        if not label:
            continue
        items.append({
            "doc_id": d.id,
            "story_id": sid,
            "article_id": a.get("source_article_id") or d.id,
            "aspect": label,
            "text": build_text(a),
        })
    return items


def score_doc_id(cache_key):
    return hashlib.sha256(cache_key.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="score all + write (default: dry-run sample)")
    ap.add_argument("--limit", type=int, default=5, help="dry-run sample size (ignored with --commit)")
    ap.add_argument("--batch-size", type=int, default=scorer.BATCH_SIZE)
    args = ap.parse_args()

    db = get_db()
    story_labels = load_story_labels(db)
    items = load_scoreable(db, story_labels)
    if not items:
        sys.exit("No scoreable articles (need articles with story_id). Run cluster_stories.py first.")

    # skip already-scored atoms (idempotent re-runs only score new work)
    def already(it):
        ck = scorer.cache_key(it["text"], it["aspect"])
        return db.collection(COLLECTION).document(score_doc_id(ck)).get().exists

    print(f"{len(items)} articles have a story_id and can be scored.")

    from anthropic import Anthropic
    client = Anthropic()

    if not args.commit:
        sample = items[:args.limit]
        print(f"\n--- DRY RUN: scoring {len(sample)} sample article(s), writing nothing ---\n")
        results, _, _ = scorer.call_claude(client, sample)
        for it in sample:
            r = results.get(it["article_id"])
            if not r:
                print(f"  [no result] {it['article_id']}")
                continue
            atom = scorer.atom_from(r, it, 0)
            print(f"  story: {it['aspect']}")
            print(f"    {it['article_id']}  score={atom['score']:+.2f}  conf={atom['confidence']:.2f}")
            print(f"    reason: {atom['reason']}")
            print(f"    evidence: {atom['evidence_spans']}\n")
        print("Inspect scores/reasons above, then --commit.")
        return

    # ---- commit: score all uncached, write atoms one at a time ----
    todo = [it for it in items if not already(it)]
    print(f"{len(todo)} to score ({len(items)-len(todo)} already cached).")
    if not todo:
        print("nothing new to score."); return

    now = datetime.now(timezone.utc)
    written = 0
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i:i + args.batch_size]
        try:
            results, latency, _ = scorer.call_claude(client, batch)
        except RuntimeError as e:
            print(f"  batch {i}: {e} — retrying singly")
            results = {}
            for it in batch:
                try:
                    r1, _, _ = scorer.call_claude(client, [it])
                    results.update(r1)
                except RuntimeError as e2:
                    print(f"    skip {it['article_id']}: {e2}")
        for it in batch:
            r = results.get(it["article_id"])
            if not r:
                continue
            atom = scorer.atom_from(r, it, latency if 'latency' in dir() else 0)
            ck = scorer.cache_key(it["text"], it["aspect"])
            doc = {
                **atom,
                "cache_key": ck,
                "content_hash": ck.split("|", 1)[0],   # scorer text hash (distinct from articles doc id)
                "article_doc_id": it["doc_id"],
                "story_id": it["story_id"],
                "is_eval_seed": False,
                "scored_at": now,
            }
            db.collection(COLLECTION).document(score_doc_id(ck)).set(doc, merge=True)
            written += 1
            if written % 50 == 0:
                print(f"  … {written}/{len(todo)} scored")
    print(f"  wrote {written} scoring atoms to {COLLECTION}.")


if __name__ == "__main__":
    main()
