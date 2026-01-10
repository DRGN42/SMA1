import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

from scripts.scraper import HorDeScraper

from .prompts import build_prompt
from .render import render_video
from .state import load_state, mark_state
from .tts import synthesize_segments
from .utils import outputs_dir, read_json, write_json, ensure_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("runner")

app = FastAPI()


class ScrapeRequest(BaseModel):
    url: str = "https://hor.de/gedichte/gedicht.php"
    language: str = "de"
    avoid_processed: bool = True
    max_attempts: int = 5


class AnalyzeRequest(BaseModel):
    url: str
    title: str
    author: str
    text: str
    language: str = "de"
    url_hash: str
    model: Optional[str] = None


class TTSRequest(BaseModel):
    url: str
    title: str
    author: str
    text: str
    language: str = "de"
    url_hash: str


class RenderRequest(BaseModel):
    url_hash: str
    audio_path: str
    segments_path: str
    images_manifest_path: str
    poem_path: Optional[str] = None


class StateRequest(BaseModel):
    url_hash: str
    status: str


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/scrape")
async def scrape(req: ScrapeRequest) -> Dict[str, Any]:
    scraper = HorDeScraper()
    attempts = 0
    while attempts < req.max_attempts:
        poem = scraper.fetch_random_poem()
        if not poem:
            attempts += 1
            continue

        poem["language"] = req.language
        url_hash = poem["url_hash"]
        if req.avoid_processed:
            state = load_state(url_hash)
            if state and state.get("status") == "done":
                attempts += 1
                continue

        poem_path = outputs_dir() / "logs" / f"poem_{url_hash}.json"
        write_json(poem_path, poem)
        mark_state(url_hash, "scraped")
        return {"url_hash": url_hash, "poem_path": str(poem_path), "poem": poem}

    raise HTTPException(status_code=404, detail="No new poem found")


@app.post("/analyze")
async def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    prompt = build_prompt(req.dict())
    base_url = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    model = req.model or os.environ.get("LMSTUDIO_MODEL", "gpt-oss-20b")

    client = OpenAI(base_url=base_url, api_key="lm-studio")
    try:
        logger.info("Analyzing with model=%s base_url=%s", model, base_url)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
            temperature=0.4,
            max_tokens=2000,
        )
        raw_text = completion.choices[0].message.content
        analysis = _extract_json(raw_text)
    except Exception as exc:
        logger.exception("Analyze failed")
        raise HTTPException(status_code=500, detail=f"Analyze failed: {exc}") from exc

    analysis_path = outputs_dir() / "logs" / f"analysis_{req.url_hash}.json"
    write_json(analysis_path, analysis)
    mark_state(req.url_hash, "analyzed")
    return {"analysis_path": str(analysis_path), **analysis}


@app.post("/tts")
async def tts(req: TTSRequest) -> Dict[str, Any]:
    output_dir = ensure_dir(outputs_dir() / "audio" / req.url_hash)
    try:
        payload = synthesize_segments(req.dict(), output_dir)
    except Exception as exc:
        logger.exception("TTS failed")
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc
    mark_state(req.url_hash, "tts_done")
    return payload


@app.post("/render")
async def render(req: RenderRequest) -> Dict[str, Any]:
    result = render_video(
        url_hash=req.url_hash,
        audio_path=Path(req.audio_path),
        segments_path=Path(req.segments_path),
        images_manifest_path=Path(req.images_manifest_path),
        output_path=outputs_dir() / "videos" / f"{req.url_hash}.mp4",
    )
    mark_state(req.url_hash, "video_done")
    return result


@app.post("/state/mark")
async def state_mark(req: StateRequest) -> Dict[str, Any]:
    return mark_state(req.url_hash, req.status)


@app.get("/state/check/{url_hash}")
async def state_check(url_hash: str) -> Dict[str, Any]:
    state = load_state(url_hash)
    if not state:
        raise HTTPException(status_code=404, detail="not found")
    return state


def _extract_json(raw_text: str) -> Dict[str, Any]:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1:
            raise HTTPException(
                status_code=500,
                detail=f"LLM returned no JSON. Raw: {raw_text[:500]}",
            )
        payload = raw_text[start : end + 1]
        return json.loads(payload)
