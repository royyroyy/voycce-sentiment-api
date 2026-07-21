#!/usr/bin/env python3
"""One-off: print the largest labeled story doc to confirm full shape. Read-only."""
import firebase_admin
from firebase_admin import credentials, firestore

if not firebase_admin._apps:
    firebase_admin.initialize_app(credentials.ApplicationDefault(), {"projectId": "voycce-mvp"})
db = firestore.client()

stories = sorted(
    (s.to_dict() for s in db.collection("stories").stream()),
    key=lambda d: d.get("size", 0), reverse=True,
)
print(f"total stories: {len(stories)}")
labeled = [s for s in stories if s.get("label")]
llm = [s for s in stories if s.get("label_via") == "llm"]
print(f"with label: {len(labeled)}  |  llm-labeled: {len(llm)}")

print("\n--- top 5 stories (label / subject / size / via) ---")
for s in stories[:5]:
    print(f"  [{s.get('label_via','?'):7}] size {s.get('size'):>2}  {s.get('label','')}")
    print(f"            subject: {s.get('subject','')}")

top = stories[0]
print("\n--- full field shape of the largest story ---")
for k, v in top.items():
    if k == "centroid":
        print(f"  centroid: [vector, {len(v)} dims]")
    elif k == "article_ids":
        print(f"  article_ids: [{len(v)} ids]")
    else:
        print(f"  {k}: {v}")

# subject vocabulary spread (are singleton subjects normalized?)
from collections import Counter
subj = Counter(s.get("subject", "?") for s in stories)
print(f"\n--- distinct subjects: {len(subj)} ---")
for name, n in subj.most_common(12):
    print(f"  {n:>3}  {name}")
