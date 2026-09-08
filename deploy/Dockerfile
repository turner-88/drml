# syntax=docker/dockerfile:1

# DRML runs on a 2 vCPU / 2 GB VPS, so the image bakes in both the ONNX model
# and the onnxruntime shared library: the box must never download either at
# boot, and cold-start memory spikes are what kill a small instance.

# ---- Stage 1: fetch the onnxruntime shared library -------------------------
FROM debian:bookworm-slim AS ort

ARG ORT_VERSION=1.20.1
ARG TARGETARCH

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl tar \
 && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    case "${TARGETARCH}" in \
      amd64) ORT_ARCH=x64 ;; \
      arm64) ORT_ARCH=aarch64 ;; \
      *) echo "unsupported arch: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    curl -fsSL -o /tmp/ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}.tgz"; \
    mkdir -p /opt/ort; \
    tar -xzf /tmp/ort.tgz -C /opt/ort --strip-components=1; \
    cp /opt/ort/lib/libonnxruntime.so* /usr/local/lib/

# ---- Stage 2: build the Go binary ------------------------------------------
FROM golang:1.25-bookworm AS build

WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY . .

# cgo is required: onnxruntime_go binds the ONNX Runtime C API.
ENV CGO_ENABLED=1
RUN go build -trimpath -ldflags="-s -w" -o /out/drml ./cmd/server \
 && go build -trimpath -ldflags="-s -w" -o /out/seed  ./cmd/seed

# ---- Stage 3: runtime -------------------------------------------------------
FROM debian:bookworm-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates tzdata \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 --create-home --home-dir /app drml

COPY --from=ort /usr/local/lib/libonnxruntime.so* /usr/local/lib/
RUN ldconfig

WORKDIR /app

COPY --from=build /out/drml /out/seed /usr/local/bin/
# Model artifacts are part of the image so startup needs no network.
COPY --chown=drml:drml model/vit_dr_int8.onnx model/labels.json model/preprocess.json ./model/

RUN mkdir -p /app/static/uploads && chown -R drml:drml /app

USER drml

ENV ORT_LIB_PATH=/usr/local/lib/libonnxruntime.so \
    MODEL_PATH=/app/model/vit_dr_int8.onnx \
    MODEL_LABELS_PATH=/app/model/labels.json \
    MODEL_PREPROCESS_PATH=/app/model/preprocess.json \
    STORAGE_LOCAL_DIR=/app/static/uploads \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8080 \
    ORT_INTRA_OP_THREADS=2 \
    ORT_INTER_OP_THREADS=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD ["/bin/sh", "-c", "exec 3<>/dev/tcp/127.0.0.1/8080 && printf 'GET /healthz HTTP/1.0\\r\\n\\r\\n' >&3 && head -1 <&3 | grep -q 200"]

ENTRYPOINT ["drml"]
