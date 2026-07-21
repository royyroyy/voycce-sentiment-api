#!/usr/bin/env python3
"""
label_stories.py — Phase E story labelling.

Gives each story in Firestore `stories` a human-readable {label, subject}:
  - MULTI-article stories (size > 1): one Claude call each, synthesizing a label +
    parent subject from the member headlines. This is where synthesis across many
    headlines produces something no single headline captures.
  - SINGLETON stories (size == 1): NO LLM call. label = the article's headline,
    subject = its `category`. Spending a call to "summarize" one headline is waste.

Mirrors score_news_llm.py's Claude plumbing exactly: Anthropic() reads
ANTHROPIC_API_KEY; model=claude-sonnet-5; NATIVE structured outputs via
output_config.format; NO temperature param (Sonnet 5 rejects it); checks stop_reason.

SAFETY: dry-run by DEFAULT — prints the {label, subject} it WOULD write per story,
writes nothing. --commit writes to the stories docs. Idempotent (set merge=True).

NOTE on subject: emitted per-story by the LLM, so subjects will be INCONSISTENT
across stories (one "Iran", another "Middle East conflict"). That's accepted for now;
consistent subjects come later from story-level embedding clustering (see Notion +
spec story-clustering-embeddings.md §8.5). This script does the story layer only.

AUTH: ANTHROPIC_API_KEY (Claude) + gcloud ADC (Firestore).
Run from voycce-sentiment-api/ with .venv active. Roy commits; Claude does not run git.
"""
import argparse
import os
import sys

import firebase_admin
from firebase_admin import credentials, firestore

MODEL_VERSION = os.environ.get("VOYCCE_SCORER_MODEL", "claude-sonnet-5")
LABEL_PROMPT_VERSION = "label-v1"
MAX_TOKENS = 1024
PROJECT_ID = "voycce-mvp"

# ---- structured-output schema (model-produced fields only) ----
LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "label":   {"type": "string", "description": "concise story name, <= 8 words, no trailing period"},
        "subject": {"type": "string", "description": "the broad parent theme this story rolls up to, 1-4 words (e.g. 'Iran', 'US-Canada trade', 'UK politics')"},
    },
    "required": ["label", "subject"],
    "additionalProperties": False,
}

SYSTEM = (
    "You name news stories for a media-monitoring product. Given the headlines of "
    "articles that all cover ONE story, produce: (1) a concise, neutral label naming "
    "that specific story, and (2) the broad subject/theme it belongs to. The label is "
    "the specific event; the subject is the umbrella it sits under. Be neutral and "
    "factual — no editorializing."
)


# ---------------------------------------------------------------- claude (multi-article only)
def call_claude(client, headlines):
    user = "Headlines covering one story:\n" + "\n".join(f"- {h}" for h in headlines)
    kwargs = dict(
        model=MODEL_VERSION,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": LABEL_SCHEMA}},
    )
    resp = client.messages.create(**kwargs)
    stop = getattr(resp, "stop_reason", None)
    if stop in ("refusal", "max_tokens"):
        return None
    import json
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            return json.loads(block.text)
    return None


# ---------------------------------------------------------------- firestore
def get_db():
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.ApplicationDefault(), {"projectId": PROJECT_ID})
    return firestore.client()


def headlines_for(db, article_ids, limit=20):
    """Fetch member headlines. Cap to keep the prompt lean on huge stories."""
    out = []
    for aid in article_ids[:limit]:
        snap = db.collection("articles").document(aid).get()
        if snap.exists:
            d = snap.to_dict()
            out.append({"headline": d.get("headline", ""), "category": d.get("category", "general")})
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="write labels (default: dry-run)")
    ap.add_argument("--limit", type=int, default=None, help="cap stories processed (testing)")
    ap.add_argument("--multi-only", action="store_true", help="skip singletons entirely")
    args = ap.parse_args()

    db = get_db()
    stories = list(db.collection("stories").stream())
    if args.limit:
        stories = stories[:args.limit]
    if not stories:
        sys.exit("No stories found. Run cluster_stories.py --commit first.")

    client = None  # lazy — only built if there's a multi-article story to label

    labeled, multi_n, single_n, failures = [], 0, 0, 0
    for s in stories:
        sd = s.to_dict()
        sid = sd.get("story_id", s.id)
        article_ids = sd.get("article_ids", [])
        size = sd.get("size", len(article_ids))

        members = headlines_for(db, article_ids)
        if not members:
            failures += 1
            continue

        if size > 1:
            if client is None:
                from anthropic import Anthropic
                client = Anthropic()  # reads ANTHROPIC_API_KEY
            result = call_claude(client, [m["headline"] for m in members])
            if not result:
                failures += 1
                continue
            label, subject, via = result["label"], result["subject"], "llm"
            multi_n += 1
        else:
            # singleton: headline IS the label; category IS the subject. No LLM.
            # Normalize casing only — outlets are inconsistent (general/General). We do
            # NOT try to fix miscategorized values (e.g. US politics tagged 'world');
            # that's the article's own bad category, and subject-level clustering
            # (deferred) replaces this whole field with consistent embedding-derived
            # subjects anyway.
            raw_subject = members[0]["category"] or "general"
            label, subject, via = members[0]["headline"], raw_subject.strip().title(), "headline"
            single_n += 1

        labeled.append({"sid": sid, "size": size, "label": label, "subject": subject, "via": via})

        if not args.commit:
            tag = "LLM " if via == "llm" else "hdl "
            print(f"  [{tag}size {size:>2}] {label}   →  subject: {subject}")

    print("\n" + "=" * 68)
    print(f"{len(stories)} stories: {multi_n} LLM-labeled, {single_n} headline-labeled, {failures} skipped")
    print("=" * 68)

    if not args.commit:
        print("\nDRY RUN — nothing written. Inspect labels above, then --commit.")
        return

    print("\n--- COMMIT: writing labels to stories docs ---")
    n = 0
    for it in labeled:
        db.collection("stories").document(it["sid"]).set({
            "label": it["label"],
            "subject": it["subject"],
            "label_via": it["via"],
            "label_prompt_version": LABEL_PROMPT_VERSION,
        }, merge=True)
        n += 1
        if n % 50 == 0:
            print(f"  … {n}/{len(labeled)} written")
    print(f"  wrote labels to {n} stories.")


if __name__ == "__main__":
    main()
