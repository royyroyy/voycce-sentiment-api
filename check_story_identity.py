#!/usr/bin/env python3
"""
check_story_identity.py — the payoff measurement (read-only, writes nothing).

Was: 279 articles -> 279 canonical_topic_id (zero convergence, no time series possible).
Now: measure articles-per-STORY. This is the number that was broken since the blocker
was found. Multi-article stories = the sentiment time series can finally exist.
"""
import sys
from collections import Counter, defaultdict
import firebase_admin
from firebase_admin import credentials, firestore

PROJECT_ID = "voycce-mvp"
if not firebase_admin._apps:
    firebase_admin.initialize_app(credentials.ApplicationDefault(), {"projectId": PROJECT_ID})
db = firestore.client()

docs = list(db.collection("articles").stream())
if not docs:
    sys.exit("No articles.")

by_story = defaultdict(list)
no_story = 0
for d in docs:
    a = d.to_dict()
    sid = a.get("story_id")
    if not sid:
        no_story += 1
        continue
    by_story[sid].append(a.get("headline", "")[:60])

sizes = Counter(len(v) for v in by_story.values())
multi = {k: v for k, v in by_story.items() if len(v) > 1}
articles_in_multi = sum(len(v) for v in multi.values())

print("=" * 68)
print(f"{len(docs)} articles  ->  {len(by_story)} stories"
      + (f"  ({no_story} not yet clustered)" if no_story else ""))
print("=" * 68)
print("\narticles-per-story:")
for n in sorted(sizes):
    print(f"  {sizes[n]:>4} stor{'y' if sizes[n]==1 else 'ies'} with {n:>2} article{'s' if n>1 else ''}")

pct = len(multi) / len(by_story) * 100 if by_story else 0
print(f"\nmulti-article stories: {len(multi)} / {len(by_story)} ({pct:.0f}% of stories)")
print(f"articles living in a multi-article story: {articles_in_multi} / {len(docs)} "
      f"({articles_in_multi/len(docs)*100:.0f}% of the corpus)")

if multi:
    biggest = max(multi.items(), key=lambda kv: len(kv[1]))
    print(f"\nlargest story: {biggest[0]} — {len(biggest[1])} articles")
    print("\n** CONVERGENCE ACHIEVED. The sentiment time series can now exist:")
    print("   any story with >1 article across >1 day is a line with movement. **")
else:
    print("\n** still no convergence — investigate. **")
