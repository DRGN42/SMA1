import json
import os
from pathlib import Path
from typing import Dict, List

from pydub import AudioSegment
from TTS.api import TTS

from .utils import ensure_dir


def _load_tts() -> TTS:
    model_name = os.environ.get("COQUI_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")
    use_cuda = os.environ.get("COQUI_USE_CUDA", "false").lower() == "true"
    return TTS(model_name=model_name, progress_bar=False, gpu=use_cuda)


def _segment_lines(text: str) -> List[str]:
    return [line.strip() for line in text.split("\n") if line.strip()]


def synthesize_segments(poem: Dict, output_dir: Path) -> Dict:
    speaker_wav = os.environ.get("COQUI_SPEAKER_WAVS")
    language = os.environ.get("COQUI_LANGUAGE", "de")

    lines = _segment_lines(poem["text"])
    ensure_dir(output_dir)

    tts = _load_tts()
    segment_paths: List[Path] = []

    for idx, line in enumerate(lines):
        segment_path = output_dir / f"segment_{idx:03d}.wav"
        tts.tts_to_file(
            text=line,
            file_path=str(segment_path),
            speaker_wav=speaker_wav,
            language=language,
        )
        segment_paths.append(segment_path)

    merged = AudioSegment.empty()
    segments_meta = []
    cursor = 0.0

    for idx, segment_path in enumerate(segment_paths):
        audio = AudioSegment.from_wav(segment_path)
        duration = len(audio) / 1000.0
        start = cursor
        end = cursor + duration
        segments_meta.append(
            {
                "index": idx,
                "text": lines[idx],
                "start": start,
                "end": end,
                "duration": duration,
                "audio_path": str(segment_path),
            }
        )
        merged += audio
        cursor = end

    main_audio_path = output_dir.parent / f"{poem['url_hash']}.wav"
    merged.export(main_audio_path, format="wav")

    segments_json_path = output_dir.parent / f"{poem['url_hash']}_segments.json"
    with segments_json_path.open("w", encoding="utf-8") as handle:
        json.dump(segments_meta, handle, indent=2, ensure_ascii=False)

    return {
        "audio_path": str(main_audio_path),
        "segments_path": str(segments_json_path),
        "segments_count": len(segments_meta),
    }
