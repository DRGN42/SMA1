import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from diffusers import DiffusionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flux")

app = FastAPI()

PIPELINE = None


def _load_pipeline() -> DiffusionPipeline:
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE

    model_id = os.environ.get("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-schnell")
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    logger.info("Loading Flux model %s", model_id)

    pipeline = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        token=token,
    )
    pipeline.to("cuda")
    PIPELINE = pipeline
    return pipeline


class GenerateRequest(BaseModel):
    url_hash: str
    analysis_path: str
    width: int = 768
    height: int = 1344
    steps: int = 12


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/generate_from_analysis")
async def generate(req: GenerateRequest) -> Dict[str, str]:
    analysis_path = Path(req.analysis_path)
    if not analysis_path.exists():
        raise HTTPException(status_code=404, detail="analysis not found")

    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    visuals = analysis.get("visual", {})
    line_prompts: List[str] = visuals.get("line_prompts", [])
    global_prompt = visuals.get("global_prompt", "")
    style_prompt = visuals.get("style_prompt", "")
    negative_prompt = visuals.get("negative_prompt", "")

    pipeline = _load_pipeline()

    output_dir = Path("/opt/poetrybot/outputs/images") / req.url_hash
    output_dir.mkdir(parents=True, exist_ok=True)

    images_meta = []
    for idx, line_prompt in enumerate(line_prompts):
        prompt = " ".join([global_prompt, line_prompt, style_prompt]).strip()
        generator = torch.Generator(device="cuda").manual_seed(1000 + idx)
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=req.width,
            height=req.height,
            num_inference_steps=req.steps,
            generator=generator,
        )
        image = result.images[0]
        output_path = output_dir / f"{idx:03d}.png"
        image.save(output_path)
        images_meta.append(
            {
                "index": idx,
                "prompt": line_prompt,
                "path": str(output_path),
            }
        )

    manifest = {"url_hash": req.url_hash, "images": images_meta}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"images_manifest_path": str(manifest_path)}
