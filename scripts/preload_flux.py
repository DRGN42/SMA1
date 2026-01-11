import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    model_id = os.environ.get("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-schnell")
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    poetrybot_root = Path(os.environ.get("POETRYBOT_ROOT", "/opt/poetrybot"))
    mount_root = os.environ.get("POETRYBOT_MOUNT")
    root = Path(mount_root) if mount_root else poetrybot_root
    if mount_root and not root.exists():
        root = poetrybot_root
    cache_dir = Path(os.environ.get("HF_HOME", root / "models"))

    cache_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=model_id,
        token=token,
        cache_dir=str(cache_dir),
        local_files_only=False,
        resume_download=True,
    )


if __name__ == "__main__":
    main()
