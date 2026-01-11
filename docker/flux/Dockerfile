FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/flux/requirements.txt /app/requirements.txt
RUN python3.11 -m pip install --no-cache-dir -r /app/requirements.txt

COPY services/flux /app/services/flux

EXPOSE 9000

CMD ["python3.11", "-m", "uvicorn", "services.flux.app:app", "--host", "0.0.0.0", "--port", "9000"]
