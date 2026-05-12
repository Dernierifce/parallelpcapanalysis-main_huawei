FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG INSTALL_GPU_PACKAGES=1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=4096

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        tshark \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
COPY requirements-gpu.txt ./requirements-gpu.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r requirements.txt \
    && if [ "$INSTALL_GPU_PACKAGES" = "1" ]; then python3 -m pip install -r requirements-gpu.txt; fi

COPY . ./

EXPOSE 8501

CMD ["streamlit", "run", "/workspace/app_gpu_dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
