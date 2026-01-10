FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/runner/requirements.txt /app/requirements.txt
RUN python3.11 -m pip install --no-cache-dir -r /app/requirements.txt

COPY services/runner /app/services/runner
COPY scripts /app/scripts

ENV PYTHONPATH=/app
ENV COQUI_TOS_AGREED=1

EXPOSE 8000

CMD ["python3.11", "-m", "uvicorn", "services.runner.app:app", "--host", "0.0.0.0", "--port", "8000"]
