#!/bin/bash
set -euo pipefail

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
LOG_FILE="${LOG_FILE:-/root/autodl-tmp/ai-backend.log}"
WAIT_SECONDS="${WAIT_SECONDS:-240}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-2}"

export PROJECTS_ROOT="${PROJECTS_ROOT:-/root/autodl-tmp/projects}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/root/autodl-tmp/Ultralytics}"

mkdir -p "$PROJECTS_ROOT" "$YOLO_CONFIG_DIR"

CODE_DIR=""
if [ -f "/root/autodl-tmp/main.py" ]; then
  CODE_DIR="/root/autodl-tmp"
elif [ -f "/root/autodl-tmp/aiVisionIterate/ai-image-recognition-backend/main.py" ]; then
  CODE_DIR="/root/autodl-tmp/aiVisionIterate/ai-image-recognition-backend"
else
  echo "未找到后端 main.py。请确认代码目录存在：/root/autodl-tmp 或 /root/autodl-tmp/aiVisionIterate/ai-image-recognition-backend" >&2
  exit 1
fi

PY="/root/miniconda3/bin/python"
if [ ! -x "$PY" ]; then
  echo "未找到 Python：$PY" >&2
  exit 1
fi

if command -v pgrep >/dev/null 2>&1; then
  pids="$(pgrep -f "uvicorn main:app" || true)"
  if [ -n "${pids:-}" ]; then
    kill -9 $pids >/dev/null 2>&1 || true
  fi
else
  pkill -f "uvicorn main:app" >/dev/null 2>&1 || true
fi

cd "$CODE_DIR"
nohup "$PY" -m uvicorn main:app --host "$HOST" --port "$PORT" --workers 1 --loop uvloop --http httptools --backlog 2048 >>"$LOG_FILE" 2>&1 &

if command -v curl >/dev/null 2>&1; then
  end_ts=$(( $(date +%s) + WAIT_SECONDS ))
  code=""
  while [ "$(date +%s)" -lt "$end_ts" ]; do
    code="$(curl -sS -m 3 -o /dev/null -w "%{http_code}" "http://$HOST:$PORT/api/v1/openapi.json" || true)"
    if [ "$code" = "200" ]; then
      break
    fi
    sleep "$WAIT_INTERVAL_SECONDS"
  done
  echo "openapi_status=${code:-000}"
else
  echo "已启动后端（未安装 curl，跳过 HTTP 检查）。"
fi

echo "CODE_DIR=$CODE_DIR"
echo "LOG_FILE=$LOG_FILE"
