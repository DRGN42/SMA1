import json
import subprocess
from pathlib import Path
from typing import Dict, List

from .utils import ensure_dir


def _build_srt(segments: List[Dict], srt_path: Path) -> None:
    lines = []
    for idx, segment in enumerate(segments, start=1):
        start = _format_time(segment["start"])
        end = _format_time(segment["end"])
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(segment["text"])
        lines.append("")
    srt_path.write_text("\n".join(lines), encoding="utf-8")


def _format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


def render_video(
    url_hash: str,
    audio_path: Path,
    segments_path: Path,
    images_manifest_path: Path,
    output_path: Path,
    fps: int = 30,
    size: str = "1080x1920",
) -> Dict:
    ensure_dir(output_path.parent)

    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    manifest = json.loads(images_manifest_path.read_text(encoding="utf-8"))

    if len(segments) != len(manifest["images"]):
        raise ValueError("Segments count does not match image count")

    concat_file = output_path.parent / f"{url_hash}_concat.txt"
    lines = []
    for idx, segment in enumerate(segments):
        image_path = manifest["images"][idx]["path"]
        lines.append(f"file '{image_path}'")
        lines.append(f"duration {segment['duration']}")
    lines.append(f"file '{manifest['images'][-1]['path']}'")
    concat_file.write_text("\n".join(lines), encoding="utf-8")

    srt_path = output_path.parent / f"{url_hash}.srt"
    _build_srt(segments, srt_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-i",
        str(audio_path),
        "-vf",
        f"scale={size}:force_original_aspect_ratio=decrease,pad={size}:(ow-iw)/2:(oh-ih)/2,subtitles={srt_path}",
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(output_path),
    ]

    subprocess.run(cmd, check=True)

    return {
        "video_path": str(output_path),
        "duration": segments[-1]["end"] if segments else 0,
        "segments": len(segments),
    }
