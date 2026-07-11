#!/usr/bin/env python3
"""
score_news_llm.py — Path B (LLM-as-teacher) scorer. Implements Stage 1a Addendum #2.

Scores news-stream eval set v1 (30 articles) with Claude and reports:
  - ROLLUP:   Spearman rho of LLM score vs human 7-point   (gate 0.60)
  - DRILL-DOWN: 3-class accuracy at +/-0.33                (gate 70%)
  - QUALITATIVE: 5 sample reason/evidence outputs for hand review
directly comparable to the Path A analysis (baseline: rho 0.56, 3-class 63.3%).

Uses NATIVE structured outputs (output_config.format, GA on the Claude API) so the
scoring atom comes back as guaranteed schema-valid JSON — no fragile text parsing.
reason + evidence_spans are mandatory schema fields (the corpus-teaching signal).

Run from voycce-sentiment-api/ with ANTHROPIC_API_KEY set and `pip install anthropic`.
Imports evaluate_v1 for text/aspect/label parity with the eval. Claude does not run
git — Roy commits.
"""
import argparse, hashlib, json, math, os, random, time, unicodedata
from collections import defaultdict
from pathlib import Path

import evaluate_v1 as base  # extract_lead, human_score_to_3class, LABELS_DIR, LABELS

# ---- versioning contract (Addendum #2 sec.2/5) ----
MODEL_VERSION  = os.environ.get("VOYCCE_SCORER_MODEL", "claude-sonnet-5")
PROMPT_VERSION = "snp-v1"
RUBRIC_VERSION = "rubric-7pt-v1"

# Lean scoring: no thinking scratchpad (JSON schema is constrained anyway).
# Set VOYCCE_SCORER_THINKING=1 to re-enable if scores look shallow.
THINKING_DISABLED = os.environ.get("VOYCCE_SCORER_THINKING", "0") != "1"

CACHE_PATH = Path(__file__).resolve().parent / "score_cache_llm.json"
BATCH_SIZE = 10
MAX_TOKENS = 8192  # headroom: Sonnet 5 counts thinking tokens + ~30% denser tokenizer
POS_B, NEG_B = 0.33, -0.33

# ---- structured-output schema: only the model-produced fields (metadata added in code) ----
ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "article_id":     {"type": "string"},
        "score":          {"type": "number"},
        "confidence":     {"type": "number"},
        "reason":         {"type": "string"},
        "evidence_spans": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["article_id", "score", "confidence", "reason", "evidence_spans"],
    "additionalProperties": False,
}
BATCH_SCHEMA = {
    "type": "object",
    "properties": {"results": {"type": "array", "items": ITEM_SCHEMA}},
    "required": ["results"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = f"""You are an expert news-sentiment analyst building a labeled corpus.

For each article you are given a TOPIC (the "aspect") and the article text. Score the \
sentiment the article expresses TOWARD THAT TOPIC — not the article's overall mood — from \
the perspective of someone tracking that topic.

Apply real-world knowledge of what is good or bad, not surface word valence:
- "producer prices rose 6.5%" is NEGATIVE for the economy even though "rose" is neutral.
- "hottest year on record" is NEGATIVE (climate), not a positive achievement.
- "hair-loss breakthrough" / "full hair growth restored" is POSITIVE for sufferers.

Rubric ({RUBRIC_VERSION}), a continuous scale you may use freely between anchors:
  -1.0 very negative | -0.66 negative | -0.33 mildly negative | 0.0 neutral |
  +0.33 mildly positive | +0.66 positive | +1.0 very positive

For each article return:
- score: float in [-1, 1] toward the topic.
- confidence: float in [0, 1], your certainty in the score.
- reason: 1-2 sentences explaining the score, referencing the topic.
- evidence_spans: short substrings copied VERBATIM from the provided text that drive the \
score. Copy exactly (for later highlighting); use [] only when the article is genuinely \
neutral toward the topic.

Echo each article_id exactly. Return one result per article."""


# ---------------------------------------------------------------- data
def collapse(s):
    return "Positive" if s >= POS_B else "Negative" if s <= NEG_B else "Neutral"

def load_articles(limit=None):
    files = sorted(base.LABELS_DIR.glob("article-*.json"))
    out = []
    for f in files:
        art = json.loads(f.read_text())
        primary = next((l for l in art.get("labels", []) if l.get("topicRole") == "primary"),
                       art["labels"][0] if art.get("labels") else None)
        if not primary:
            continue
        out.append({
            "article_id": art["articleId"],
            "aspect": primary["canonicalTopicTitle"],
            "text": art["headline"] + ". " + base.extract_lead(art["fullText"]),
            "human_score": float(primary["score"]),
            "human_class": collapse(float(primary["score"])),
        })
    return out[:limit] if limit else out

def norm(t):
    return " ".join(unicodedata.normalize("NFC", t).split()).strip()

def cache_key(text, aspect):
    h = hashlib.sha256(norm(text).encode("utf-8")).hexdigest()
    return f"{h}|{aspect}|{PROMPT_VERSION}|{MODEL_VERSION}"


# ---------------------------------------------------------------- API
def call_claude(client, articles):
    """One structured-output call for a batch. Returns (results_by_id, latency_ms, usage)."""
    payload = "\n\n".join(
        f"[{i+1}] article_id: {a['article_id']}\naspect: {a['aspect']}\ntext: \"\"\"{a['text']}\"\"\""
        for i, a in enumerate(articles)
    )
    user = f"Score the following {len(articles)} article(s):\n\n{payload}"
    t0 = time.perf_counter()
    # Note: Sonnet 5 rejects non-default `temperature` (400). Determinism now comes from
    # structured outputs + cache. Thinking disabled — a JSON schema needs no scratchpad.
    kwargs = dict(
        model=MODEL_VERSION, max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": BATCH_SCHEMA}},
    )
    if THINKING_DISABLED:
        kwargs["thinking"] = {"type": "disabled"}
    resp = client.messages.create(**kwargs)
    latency = int((time.perf_counter() - t0) * 1000)
    stop = getattr(resp, "stop_reason", None)
    if stop in ("refusal", "max_tokens"):
        raise RuntimeError(f"stop_reason={stop} (batch size {len(articles)})")
    text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
    results = {r["article_id"]: r for r in json.loads(text)["results"]}
    usage = getattr(resp, "usage", None)
    return results, latency, usage


def atom_from(result, article, latency_ms):
    return {
        "article_id": article["article_id"],
        "aspect": article["aspect"],
        "score": max(-1.0, min(1.0, float(result["score"]))),   # clamp (schema can't bound)
        "confidence": max(0.0, min(1.0, float(result["confidence"]))),
        "reason": result["reason"],
        "evidence_spans": list(result.get("evidence_spans", [])),
        "model_version": MODEL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "scorer_latency_ms": latency_ms,
    }


def score_articles(client, articles, cache, refresh, batch_size):
    todo = [a for a in articles if refresh or cache_key(a["text"], a["aspect"]) not in cache]
    print(f"  cache: {len(articles)-len(todo)} hit / {len(todo)} to score", flush=True)
    tok_in = tok_out = 0
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        try:
            results, latency, usage = call_claude(client, batch)
        except (RuntimeError, json.JSONDecodeError, KeyError) as e:
            print(f"    batch {i//batch_size} failed ({e}); retrying items singly", flush=True)
            results, latency, usage = {}, 0, None
            for a in batch:  # partial-failure isolation (Addendum #2 sec.6)
                r1, l1, u1 = call_claude(client, [a])
                results.update(r1)
                if u1: tok_in += getattr(u1, "input_tokens", 0); tok_out += getattr(u1, "output_tokens", 0)
        if usage:
            tok_in += getattr(usage, "input_tokens", 0); tok_out += getattr(usage, "output_tokens", 0)
        per_item = latency // max(1, len(batch))
        for a in batch:
            r = results.get(a["article_id"])
            if r is None:  # still missing after retry
                r1, l1, _ = call_claude(client, [a]); r = r1.get(a["article_id"]); per_item = l1
            cache[cache_key(a["text"], a["aspect"])] = atom_from(r, a, per_item)
        CACHE_PATH.write_text(json.dumps(cache, indent=2))
        print(f"    scored {min(i+batch_size, len(todo))}/{len(todo)}", flush=True)
    return [cache[cache_key(a["text"], a["aspect"])] for a in articles], tok_in, tok_out


# ---------------------------------------------------------------- stats
def spearman(x, y):
    try:
        from scipy.stats import spearmanr
        r = spearmanr(x, y); return float(r.statistic), float(r.pvalue), "scipy"
    except Exception:
        def rank(a):
            o = sorted(range(len(a)), key=lambda i: a[i]); rk = [0.0]*len(a); i = 0
            while i < len(a):
                j = i
                while j+1 < len(a) and a[o[j+1]] == a[o[i]]: j += 1
                for k in range(i, j+1): rk[o[k]] = (i+j)/2.0 + 1.0
                i = j+1
            return rk
        rx, ry = rank(x), rank(y); n = len(x)
        mx, my = sum(rx)/n, sum(ry)/n
        num = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
        dn = math.sqrt(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))
        rho = num/dn if dn else 0.0
        return rho, float("nan"), "manual(no-scipy)"


def report(articles, atoms, tok_in, tok_out):
    by_id = {a["article_id"]: a for a in articles}
    rows = [{**by_id[at["article_id"]], **at} for at in atoms]
    hs = [r["human_score"] for r in rows]; ms = [r["score"] for r in rows]
    rho, p, how = spearman(ms, hs)

    print("\n" + "="*70 + "\n(A) ROLLUP — Spearman rho (LLM score vs human 7-point)\n" + "="*70)
    pv = f"p={p:.4g}" if p == p else "p=n/a"
    print(f"  n={len(rows)}  rho={rho:+.3f}  {pv}  [{how}]   gate 0.60 -> "
          f"{'PASS' if rho>=0.60 else 'FAIL' if rho<0.40 else 'MIDDLE'}   (Path A was 0.56)")

    LB = ["Negative", "Neutral", "Positive"]
    correct = sum(collapse(r["score"]) == r["human_class"] for r in rows)
    acc = correct/len(rows)
    print("\n" + "="*70 + "\n(B) DRILL-DOWN — 3-class accuracy at +/-0.33\n" + "="*70)
    print(f"  {correct}/{len(rows)} = {acc*100:.1f}%   gate 70% -> {'PASS' if acc>=0.70 else 'FAIL'}"
          f"   (Path A was 63.3%)")
    cm = {h: {q: 0 for q in LB} for h in LB}
    for r in rows: cm[r["human_class"]][collapse(r["score"])] += 1
    print("            " + "".join(f"{q:>10}" for q in LB))
    for h in LB: print(f"  {h:<10}" + "".join(f"{cm[h][q]:>10}" for q in LB))
    print("  per-class recall: " + "  ".join(f"{h}={cm[h][h]}/{sum(cm[h].values())}" for h in LB))

    print("\n" + "="*70 + "\n(C) QUALITATIVE — 5 sample reason/evidence outputs\n" + "="*70)
    for r in random.Random(42).sample(rows, min(5, len(rows))):
        ok = "OK" if collapse(r["score"]) == r["human_class"] else "MISS"
        print(f"\n  {r['article_id']}  human={r['human_class']}({r['human_score']:+.2f}) "
              f"llm={collapse(r['score'])}({r['score']:+.2f}) conf={r['confidence']:.2f}  [{ok}]")
        print(f"    aspect: {r['aspect']}")
        print(f"    reason: {r['reason']}")
        print(f"    evidence: {r['evidence_spans']}")

    print("\n" + "="*70 + "\nVERDICT\n" + "="*70)
    both = rho >= 0.60 and acc >= 0.70
    print(f"  rollup rho={rho:+.3f} (0.60) | drill-down {acc*100:.1f}% (70%) -> "
          + ("BOTH PASS — Path B validated, proceed to Phase E" if both
             else "review: gate(s) not cleared — inspect reasons + confidence before deciding"))
    if tok_in or tok_out:
        print(f"  tokens: in={tok_in} out={tok_out}  (30-article dev run; Batch API = 50% off at corpus scale)")
    print("  Reminder: n=30, no inter-annotator agreement yet (Phase 2) — read rho/acc as directional.")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="ignore cache, re-score all")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--limit", type=int, default=None, help="score only first N (smoke test)")
    args = ap.parse_args()

    from anthropic import Anthropic
    client = Anthropic()  # reads ANTHROPIC_API_KEY

    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    articles = load_articles(args.limit)
    print(f"Scoring {len(articles)} articles with {MODEL_VERSION} "
          f"(prompt {PROMPT_VERSION}, rubric {RUBRIC_VERSION})")
    atoms, tok_in, tok_out = score_articles(client, articles, cache, args.refresh, args.batch_size)
    report(articles, atoms, tok_in, tok_out)


if __name__ == "__main__":
    main()
