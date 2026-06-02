#!/usr/bin/env python3
"""Entry point for the RAG document processing worker.

Usage:
    python scripts/run_worker.py          # runs forever
    python scripts/run_worker.py --once    # processes once and exits

Environment variables:
    LARAVEL_API_URL       Base URL of the Laravel application (default: http://localhost)
    RAG_SERVICE_TOKEN     Shared secret for service-to-service auth (required)
    WORKER_POLL_INTERVAL  Seconds between polls (default: 15)
"""
import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

# Ensure the project root is on sys.path so ``from app.worker`` works.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Document Worker")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process pending documents once and exit",
    )
    args = parser.parse_args()

    if not os.getenv("RAG_SERVICE_TOKEN"):
        print(
            "ERROR: RAG_SERVICE_TOKEN environment variable is not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    from app.worker import run_worker_once, run_worker_forever

    if args.once:
        asyncio.run(run_worker_once())
    else:
        asyncio.run(run_worker_forever())


if __name__ == "__main__":
    main()
