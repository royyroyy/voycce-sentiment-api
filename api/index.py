"""
voycce-sentiment-api — v2 canonical sentiment scoring infrastructure.

Phase 2 Stage 1a scaffolding (2026-06-23). The /score-news endpoint
currently returns a stub response; real model scoring lands in
Stage 1a model integration (next session).

Architectural context: docs/architecture/stable-topic-sentiment-phase2.md
in the voycce-mvp repo.
"""

import time
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="voycce-sentiment-api",
    description="v2 canonical sentiment scoring infrastructure for VOYCCE topics",
    version="0.1.0-stub",
)


class ScoreNewsRequest(BaseModel):
    article_text: str
    headline: str
    canonical_topic_title: str
    canonical_topic_id: str
    model_version: str


class ScoreNewsResponse(BaseModel):
    score: float
    confidence: float
    model_version: str
    scorer_latency_ms: int


@app.get("/")
def root():
    return {
        "service": "voycce-sentiment-api",
        "status": "stub",
        "version": "0.1.0-stub",
        "phase": "Stage 1a scaffolding — model integration pending",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score-news", response_model=ScoreNewsResponse)
def score_news(request: ScoreNewsRequest):
    """
    Stub endpoint. Returns fixed values regardless of input.
    Real scoring lands in Stage 1a model integration.
    Contract shape locked here so Stage 1c (NewsService integration)
    can begin in parallel.
    """
    start = time.perf_counter()
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return ScoreNewsResponse(
        score=0.0,
        confidence=1.0,
        model_version="stub-v0",
        scorer_latency_ms=elapsed_ms,
    )
