#!/usr/bin/env bash
# Run Jaeger without Docker, from the official standalone release.
#
# Docker is the normal path (see docker-compose.observability.yml). This is the
# fallback for a machine with no container runtime: the same Jaeger v2 all-in-one
# binary the image wraps, on the same ports, verified against the checksum the
# project publishes alongside the release.
#
#   scripts/jaeger_local.sh            # download if needed, then run in foreground
#   scripts/jaeger_local.sh --start    # run in the background, write a logfile
#   scripts/jaeger_local.sh --stop
#   scripts/jaeger_local.sh --status
#
# Ports: 16686 UI + query API, 4317 OTLP/gRPC, 4318 OTLP/HTTP.
# Storage is in-memory: instant start, nothing persisted between runs.

set -euo pipefail

VERSION="${JAEGER_VERSION:-2.20.0}"
PREFIX="${JAEGER_PREFIX:-$HOME/.local/jaeger}"
UI_PORT="${JAEGER_UI_PORT:-16686}"
LOG="${JAEGER_LOG:-$PREFIX/jaeger.log}"
PIDFILE="$PREFIX/jaeger.pid"

case "$(uname -s)" in
  Darwin) OS=darwin ;;
  Linux)  OS=linux ;;
  *) echo "unsupported OS $(uname -s); use the Docker compose file instead" >&2; exit 2 ;;
esac
case "$(uname -m)" in
  arm64|aarch64) ARCH=arm64 ;;
  x86_64|amd64)  ARCH=amd64 ;;
  *) echo "unsupported arch $(uname -m); use the Docker compose file instead" >&2; exit 2 ;;
esac

NAME="jaeger-${VERSION}-${OS}-${ARCH}"
BASE="https://github.com/jaegertracing/jaeger/releases/download/v${VERSION}"
BIN="$PREFIX/$NAME/jaeger"

fetch() {
  [ -x "$BIN" ] && return 0
  mkdir -p "$PREFIX"
  echo "→ downloading $NAME from the official release"
  curl -fsSL -o "$PREFIX/$NAME.tar.gz" "$BASE/$NAME.tar.gz"
  curl -fsSL -o "$PREFIX/$NAME.sha256sum.txt" "$BASE/$NAME.sha256sum.txt"
  tar xzf "$PREFIX/$NAME.tar.gz" -C "$PREFIX"
  # The published checksums are per-file inside the archive, so verify after
  # extraction. A binary that does not match its published hash is not run.
  ( cd "$PREFIX" && shasum -a 256 -c "$NAME.sha256sum.txt" ) \
    || { echo "checksum verification FAILED — refusing to run $BIN" >&2; exit 1; }
  echo "→ checksum verified"
}

status() {
  if curl -fsS --max-time 3 "http://localhost:$UI_PORT/api/services" >/dev/null 2>&1; then
    echo "jaeger is up: UI http://localhost:$UI_PORT · OTLP 4317/gRPC 4318/HTTP"
    curl -fsS "http://localhost:$UI_PORT/api/services"; echo
  else
    echo "jaeger is not answering on port $UI_PORT"
    return 1
  fi
}

case "${1:-run}" in
  --stop)
    if [ -f "$PIDFILE" ]; then
      pid="$(cat "$PIDFILE")"
      kill "$pid" 2>/dev/null || true
      for _ in $(seq 1 10); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      if kill -0 "$pid" 2>/dev/null; then kill -9 "$pid" 2>/dev/null || true; sleep 1; fi
      rm -f "$PIDFILE"
    fi
    if curl -fsS --max-time 3 "http://localhost:$UI_PORT/api/services" >/dev/null 2>&1; then
      echo "still serving on port $UI_PORT - not stopped" >&2
      command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$UI_PORT" -sTCP:LISTEN >&2 || true
      exit 1
    fi
    echo "stopped" ;;
  --status) status ;;
  --start)
    fetch
    ( cd "$PREFIX/$NAME" && exec nohup ./jaeger >"$LOG" 2>&1 ) &
    echo $! >"$PIDFILE"
    for _ in $(seq 1 30); do sleep 1; status >/dev/null 2>&1 && break; done
    status || { echo "--- last 20 log lines"; tail -20 "$LOG"; exit 1; } ;;
  *)
    fetch
    echo "→ UI http://localhost:$UI_PORT · OTLP 4317/gRPC 4318/HTTP · ctrl-c to stop"
    cd "$PREFIX/$NAME" && exec ./jaeger ;;
esac
