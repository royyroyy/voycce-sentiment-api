# voycce-sentiment-api

The v2 canonical sentiment scoring infrastructure for VOYCCE.

## Status

**Stage 1a scaffolding (2026-06-23).** Stub deployment only — `POST /score-news` returns fixed values regardless of input. Model integration lands in Stage 1a model selection (next session).

## What this is

A Python FastAPI service deployed on Vercel that scores text against a canonical topic for targeted stance. Built fresh to v2 standards per the Phase 2 Spec v2 (see `docs/architecture/stable-topic-sentiment-phase2.md` in the [voycce-mvp](https://github.com/royyroyy/voycce-mvp) repo).

## Why this exists separately from voycce-x-api

The Phase 2 spec commits to `(input, stream_type, canonical_topic, model_version)` as the scoring atom and requires deterministic, reproducible outputs per Invariant 10. The leading targeted news-domain stance scoring model (NewsMTSC / NewsSentiment, Hamborg & Donnay 2021) uses a custom GRU-TSC architecture that is Python-only — see Prep 3 / Stage 1a deployment shape addendum.

This service is intentionally separate from \`voycce-x-api\` (the pre-v2 social-buzz scraper). The deployment **shape** is the same (Vercel Python + FastAPI + cron-able); the **standards** are different — this service is built fresh to v2 Invariants 1-11. \`voycce-x-api\` will be refactored to match this reference implementation in Stage 1e.

## API contract

### POST /score-news

Request body:

\`\`\`json
{
  "article_text": "string",
  "headline": "string",
  "canonical_topic_title": "string",
  "canonical_topic_id": "string",
  "model_version": "string"
}
\`\`\`

Response:

\`\`\`json
{
  "score": 0.0,
  "confidence": 1.0,
  "model_version": "stub-v0",
  "scorer_latency_ms": 1
}
\`\`\`

### GET /health

Returns \`{"status": "ok"}\`.

## Local development

\`\`\`bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.index:app --reload --port 8000
\`\`\`

Then in another terminal:

\`\`\`bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/score-news \\
  -H "Content-Type: application/json" \\
  -d '{"article_text":"test","headline":"test","canonical_topic_title":"test","canonical_topic_id":"test","model_version":"stub-v0"}'
\`\`\`

## Deployment

Deployed on Vercel. Auto-deploys from main on push.
