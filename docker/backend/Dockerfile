FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATABASE_URL=sqlite+aiosqlite:////data/fedcompass.db \
    REAL_DATA_ROOT=/data/real_data

ARG DEBIAN_MIRROR=deb.debian.org
RUN sed -i "s|http://deb.debian.org|https://${DEBIAN_MIRROR}|g" /etc/apt/sources.list.d/debian.sources \
    && apt-get update -o Acquire::Retries=3 \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libjpeg62-turbo \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/fedcompass

COPY 0623v1李-后端/requirements.txt ./requirements.txt
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PYTORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        python -m pip install \
            --index-url "${PYTORCH_CPU_INDEX_URL}" \
            --extra-index-url "${PIP_INDEX_URL}" \
            "torch==${TORCH_VERSION}+cpu" \
            "torchvision==${TORCHVISION_VERSION}+cpu"; \
    else \
        python -m pip install \
            --index-url "${PIP_INDEX_URL}" \
            "torch==${TORCH_VERSION}" \
            "torchvision==${TORCHVISION_VERSION}"; \
    fi \
    && python -m pip install --index-url "${PIP_INDEX_URL}" -r requirements.txt

COPY 0623v1李-后端/app ./app

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin fedcompass \
    && mkdir -p /data \
    && chown -R fedcompass:fedcompass /opt/fedcompass /data

USER fedcompass

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
