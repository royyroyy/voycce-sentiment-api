#!/usr/bin/env python3
"""
cluster_stories.py — Phase E story clustering via embeddings.

THE JOB: group the articles in Firestore `articles` into STORIES (many articles ->
one story), so the V2 sentiment time series has multiple articles per line.
Replaces the dead entity-pair anchor. Per docs/architecture/story-clustering-embeddings.md.

HOW: embed each article (Voyage voyage-4) -> assign to a story by cosine similarity
vs. existing story centroids -> above threshold joins, below mints a new story.
Voyage vectors are unit-normalized, so dot product == cosine similarity.

GOVERNING PRINCIPLE (Open 8, Roy): precision over recall. A story fragmenting into
two is recoverable; two stories wrongly merged silently corrupts a sentiment line.
Threshold is deliberately STRICT. The right value is unknown for voyage-4-on-news —
this script REPORTS the similarity distribution so it's tuned from data, not guessed.

SAFETY: dry-run by DEFAULT. Embeds (cheap) and logs the clustering it WOULD write,
but writes NOTHING to Firestore. --commit writes embedding + story_id back.
Embedding is versioned (embedding_model_version) so the provider is swappable later.

AUTH: Voyage key via VOYAGE_API_KEY env. Firestore via gcloud ADC.
Run from voycce-sentiment-api/ with .venv active. Roy commits; Claude does not run git.
"""
import argparse
import os
import sys
from collections import defaultdict

import firebase_admin
from firebase_admin import credentials, firestore

PROJECT_ID = "voycce-mvp"
EMBED_MODEL = "voyage-4"
EMBED_VERSION = "voyage-4"          # written to each vector; the swap-safety tag
EMBED_DIM = 1024                    # voyage-4 default
VOYAGE_BATCH = 128                  # SDK per-call limit
# Strict starting threshold (precision > recall). NOT authoritative — tune from the
# reported distribution before trusting live merges.
SIM_THRESHOLD = 0.75


# ---------------------------------------------------------------- embedding (swappable)
def embed_texts(texts):
    """The single provider-boundary function. Swap Voyage->OpenAI here only.
    Returns list of unit-normalized vectors (Voyage guarantees unit norm)."""
    import voyageai
    if not os.environ.get("VOYAGE_API_KEY"):
        sys.exit("VOYAGE_API_KEY not set. Add it to voycce-sentiment-api/.env")
    vo = voyageai.Client()  # reads VOYAGE_API_KEY
    out = []
    for i in range(0, len(texts), VOYAGE_BATCH):
        batch = texts[i:i + VOYAGE_BATCH]
        out.extend(vo.embed(batch, model=EMBED_MODEL, input_type="document").embeddings)
    return out


# ---------------------------------------------------------------- similarity
def cosine(a, b):
    # Voyage vectors are unit-norm -> dot product == cosine. No normalization needed.
    return sum(x * y for x, y in zip(a, b))


def assign_to_stories(items, threshold):
    """Greedy online clustering (same shape as the production ingestion path would use).
    items: list of dicts with 'embedding'. Returns (stories, decisions).
    A story's centroid is the running mean of its members' vectors."""
    stories = []   # each: {"centroid": [...], "members": [idx...], "sum": [...]}
    decisions = []
    for idx, it in enumerate(items):
        emb = it["embedding"]
        best_sim, best_story = -1.0, None
        for s_i, s in enumerate(stories):
            sim = cosine(emb, s["centroid"])
            if sim > best_sim:
                best_sim, best_story = sim, s_i
        if best_story is not None and best_sim >= threshold:
            s = stories[best_story]
            s["members"].append(idx)
            s["sum"] = [a + b for a, b in zip(s["sum"], emb)]          # running sum
            n = len(s["members"])
            s["centroid"] = [v / n for v in s["sum"]]                  # -> running mean
            decisions.append({"idx": idx, "action": "join", "story": best_story, "sim": best_sim})
        else:
            stories.append({"centroid": list(emb), "sum": list(emb), "members": [idx]})
            decisions.append({"idx": idx, "action": "new", "story": len(stories) - 1,
                              "sim": best_sim if best_sim >= 0 else None})
    return stories, decisions


# ---------------------------------------------------------------- firestore
def get_db():
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.ApplicationDefault(), {"projectId": PROJECT_ID})
    return firestore.client()


def story_doc_id(seed_content_hash):
    return f"st_{seed_content_hash[:16]}"   # stable id seeded from first member's hash


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="write embeddings + story_id (default: dry-run)")
    ap.add_argument("--threshold", type=float, default=SIM_THRESHOLD)
    ap.add_argument("--limit", type=int, default=None, help="cap articles (testing)")
    args = ap.parse_args()

    db = get_db()
    q = db.collection("articles")
    docs = list(q.stream())
    if args.limit:
        docs = docs[:args.limit]
    if not docs:
        sys.exit("No articles found.")

    items = []
    for d in docs:
        a = d.to_dict()
        text = f"{a.get('headline','')}\n{(a.get('full_content') or '')[:1500]}".strip()
        items.append({"doc_id": d.id, "content_hash": a.get("content_hash", d.id),
                      "headline": a.get("headline", "")[:70], "source": a.get("source"),
                      "text": text})

    print(f"embedding {len(items)} articles via {EMBED_MODEL} ({EMBED_DIM}d)…")
    vecs = embed_texts([it["text"] for it in items])
    for it, v in zip(items, vecs):
        it["embedding"] = v

    stories, decisions = assign_to_stories(items, args.threshold)

    # ---- distribution report (the reason this script exists: tune threshold from data)
    joins = [d["sim"] for d in decisions if d["action"] == "join"]
    sizes = sorted((len(s["members"]) for s in stories), reverse=True)
    multi = [s for s in stories if len(s["members"]) > 1]
    print("=" * 72)
    print(f"threshold {args.threshold}: {len(items)} articles -> {len(stories)} stories")
    print(f"  multi-article stories: {len(multi)}  |  singletons: {len(stories)-len(multi)}")
    print(f"  largest stories (by size): {sizes[:8]}")
    if joins:
        joins_sorted = sorted(joins)
        print(f"  join-similarity: min {min(joins):.3f}  median {joins_sorted[len(joins)//2]:.3f}  max {max(joins):.3f}")
    print("=" * 72)

    if multi:
        print("\n--- multi-article stories (INSPECT: real convergence or over-merge?) ---")
        for s in sorted(multi, key=lambda x: -len(x["members"]))[:8]:
            print(f"\n  story ({len(s['members'])} articles):")
            for idx in s["members"]:
                it = items[idx]
                print(f"    [{it['source'] or '?'}] {it['headline']}")
    else:
        print("\n  ** no multi-article stories at this threshold — likely too strict. **")

    if not args.commit:
        print(f"\nDRY RUN — nothing written. Inspect above, tune --threshold, then --commit.")
        return

    # ---- commit: write embedding + story_id back to each article, and stories/ docs
    # Embeddings are large Firestore arrays whose serialized size is hard to predict,
    # so any batch size is a gamble against the ~10MB transaction limit. Write ONE doc
    # at a time instead: a single doc with one embedding is well under Firestore's
    # ~1MB per-DOCUMENT cap, so this cannot hit the transaction-size error. Slower,
    # but this is a one-time script over a few hundred docs — seconds, not throughput.
    print("\n--- COMMIT: writing embeddings + story_id (one doc at a time) ---")
    written = 0
    for s in stories:
        seed = items[s["members"][0]]["content_hash"]
        sid = story_doc_id(seed)
        member_ids = [items[i]["doc_id"] for i in s["members"]]
        db.collection("stories").document(sid).set({
            "story_id": sid,
            "centroid": s["centroid"],
            "article_ids": member_ids,
            "embedding_model_version": EMBED_VERSION,
            "size": len(member_ids),
        })
        for i in s["members"]:
            it = items[i]
            sim = next((d["sim"] for d in decisions if d["idx"] == i), None)
            db.collection("articles").document(it["doc_id"]).set({
                "embedding": it["embedding"],
                "embedding_model_version": EMBED_VERSION,
                "story_id": sid,
                "story_similarity": sim,
            }, merge=True)
            written += 1
            if written % 50 == 0:
                print(f"  … {written}/{len(items)} articles written")
    print(f"  wrote {len(stories)} stories over {len(items)} articles.")


if __name__ == "__main__":
    main()
