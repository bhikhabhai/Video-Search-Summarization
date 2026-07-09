#!/usr/bin/env bash
#
# Mini-VSS build & run helper.
#
#   ./build.sh              build (using cache) and start the stack
#   ./build.sh --no-cache   force a clean image rebuild, then start
#
set -euo pipefail

NO_CACHE=0
for arg in "$@"; do
  case "$arg" in
    --no-cache) NO_CACHE=1 ;;
    -h|--help)
      echo "Usage: ./build.sh [--no-cache]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: ./build.sh [--no-cache]" >&2
      exit 2
      ;;
  esac
done

# ── Prerequisite checks ───────────────────────────────────────
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: 'docker' is not installed or not on PATH." >&2
  echo "Install Docker: https://docs.docker.com/get-docker/" >&2
  exit 1
fi

# Support both Compose v2 ('docker compose') and v1 ('docker-compose').
if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE="docker-compose"
else
  echo "ERROR: Docker Compose is not installed." >&2
  echo "Install the Compose plugin: https://docs.docker.com/compose/install/" >&2
  exit 1
fi

echo "Using: ${COMPOSE}"

# ── Build & run ───────────────────────────────────────────────
if [ "${NO_CACHE}" -eq 1 ]; then
  echo "Building images with --no-cache (clean rebuild)..."
  ${COMPOSE} build --no-cache
  ${COMPOSE} up -d
else
  echo "Building images (using cache) and starting..."
  ${COMPOSE} up -d --build
fi

# ── Success message ───────────────────────────────────────────
echo ""
echo "Mini-VSS is up."
echo "  API : http://localhost:8000        (interactive docs at /docs)"
echo "  UI  : http://localhost:3000"
echo ""
echo "Follow logs with:  ${COMPOSE} logs -f"
echo "Stop with:         ${COMPOSE} down"
