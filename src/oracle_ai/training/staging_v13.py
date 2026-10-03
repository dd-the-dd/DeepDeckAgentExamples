from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from deepdeck_agent import AgentRunner, PlaySpeed, ServerTarget

from deepdeck_examples.configuration import deep_learning_config
from deepdeck_examples.v13_agent import load_v13_agent


def _write_result(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending.replace(path)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    token = os.getenv("DEEPDECK_API_KEY", "").strip()
    if not token:
        raise RuntimeError("DEEPDECK_API_KEY is required for staging evaluation")
    platform = args.platform_url.rstrip("/")
    target = ServerTarget.deepdeckleague(
        platform_url=f"{platform}/v1",
        account_token=token,
    )
    agent = load_v13_agent(args.checkpoint, device=args.device)
    runner = AgentRunner(
        agent=agent,
        config=deep_learning_config("v13"),
        target=target,
        speed=PlaySpeed.SECOND_1,
    )
    connection = asyncio.create_task(runner.serve())
    match_id = ""
    started = time.time()
    try:
        await runner.wait_until_connected(timeout=45)
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{platform}/v1/matchmaking/exhibitions/random",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "format": args.format,
                    "agentVersion": args.version,
                    "agentGeneration": "v13",
                    "artifactDigest": args.version,
                },
            )
            response.raise_for_status()
            payload = response.json().get("data", {})
            match_id = str(payload.get("matchId", ""))
            if not match_id:
                raise RuntimeError("staging exhibition response omitted matchId")
            while time.time() - started < args.timeout_seconds:
                match_response = await client.get(f"{platform}/v1/matches/{match_id}")
                match_response.raise_for_status()
                summary = match_response.json().get("data", {}).get("summary", {})
                status = str(summary.get("status", "unknown"))
                if status in {"complete", "failed", "cancelled"}:
                    return {
                        "status": status,
                        "matchId": match_id,
                        "version": args.version,
                        "format": args.format,
                        "elapsedSeconds": time.time() - started,
                        "summary": summary,
                    }
                await asyncio.sleep(5)
        raise TimeoutError(f"staging match {match_id} exceeded {args.timeout_seconds}s")
    finally:
        connection.cancel()
        await asyncio.gather(connection, return_exceptions=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one V13 random staging exhibition")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--platform-url", required=True)
    parser.add_argument("--format", default="legacy")
    parser.add_argument("--version", required=True)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    args = parser.parse_args()
    try:
        result = asyncio.run(run(args))
    except Exception as error:
        result = {
            "status": "error",
            "version": args.version,
            "error": str(error),
            "timestampUnixMs": int(time.time() * 1000),
        }
    _write_result(args.result, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["status"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
