"""Live Modal read benchmark for issue #5088 validation.

This benchmark compares the current SDK Modal read path on this branch with the
pre-change shell transport while exercising a concrete workspace-inspection
workload. It intentionally excludes sandbox startup from the read timing.

Run from the repository root after Modal authentication is configured:

    uv run python examples/sandbox/extensions/modal_read_benchmark.py

The benchmark requires live Modal access. It does not require an OpenAI API key.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import platform
import shlex
import statistics
import subprocess
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

from agents.extensions.sandbox import ModalSandboxClient, ModalSandboxClientOptions
from agents.sandbox.session.base_sandbox_session import BaseSandboxSession
from agents.sandbox.workspace_paths import sandbox_path_str
from examples.sandbox.misc.example_support import text_manifest


WORKLOAD_FILES: dict[str, str] = {
    "README.md": (
        "# Modal Demo Workspace\n\n"
        "This workspace exists to validate the Modal sandbox backend with a "
        "repeatable workspace-inspection workload.\n"
    ),
    "incident.md": (
        "# Incident\n\n"
        "- Customer: Fabrikam Retail.\n"
        "- Issue: delayed reporting rollout.\n"
        "- Primary blocker: incomplete security questionnaire.\n"
    ),
    "plan.md": (
        "# Plan\n\n"
        "1. Close the questionnaire.\n"
        "2. Reconfirm the rollout date with the customer.\n"
        "3. Record the handoff and next owner.\n"
    ),
}


def _git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("values must not be empty")
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


async def _native_read(session: BaseSandboxSession, path: Path) -> bytes:
    handle = await session.read(path)
    try:
        payload = handle.read()
    finally:
        handle.close()
    if not isinstance(payload, bytes):
        raise TypeError(f"expected bytes from native session.read(), got {type(payload)!r}")
    return payload


async def _shell_read(session: BaseSandboxSession, path: Path) -> bytes:
    # Reproduce the pre-change Modal transport while retaining SDK path validation.
    resolved = await session._validate_path_access(path)  # noqa: SLF001
    command = ["sh", "-lc", f"cat -- {shlex.quote(sandbox_path_str(resolved))}"]
    result = await session.exec(*command, shell=False)
    if not result.ok():
        raise RuntimeError(
            f"shell baseline read failed for {path}: "
            f"exit={result.exit_code} stderr={result.stderr.decode('utf-8', 'replace')}"
        )
    return result.stdout


async def _verify_correctness(session: BaseSandboxSession, paths: Sequence[Path]) -> None:
    for path in paths:
        expected = WORKLOAD_FILES[path.as_posix()].encode("utf-8")
        native = await _native_read(session, path)
        shell = await _shell_read(session, path)
        if native != expected:
            raise RuntimeError(f"native read mismatch for {path}")
        if shell != expected:
            raise RuntimeError(f"shell baseline mismatch for {path}")
        if native != shell:
            raise RuntimeError(f"native/shell mismatch for {path}")


async def _measure_workload(
    reader: Callable[[BaseSandboxSession, Path], Awaitable[bytes]],
    session: BaseSandboxSession,
    paths: Sequence[Path],
    *,
    cycles: int,
) -> float:
    started = time.perf_counter()
    for _ in range(cycles):
        for path in paths:
            await reader(session, path)
    return time.perf_counter() - started


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "median_s": statistics.median(values),
        "p95_s": _percentile(values, 0.95),
        "min_s": min(values),
        "max_s": max(values),
    }


async def run_benchmark(
    *,
    app_name: str,
    observations: int,
    cycles: int,
    warmup_cycles: int,
) -> dict[str, object]:
    manifest = text_manifest(WORKLOAD_FILES)
    client = ModalSandboxClient()
    options = ModalSandboxClientOptions(app_name=app_name)

    create_started = time.perf_counter()
    session = await client.create(manifest=manifest, options=options)
    await session.start()
    startup_s = time.perf_counter() - create_started

    paths = [Path(name) for name in WORKLOAD_FILES]
    try:
        await _verify_correctness(session, paths)

        for _ in range(warmup_cycles):
            for path in paths:
                await _shell_read(session, path)
                await _native_read(session, path)

        samples: dict[str, list[float]] = {"shell": [], "native": []}
        for observation in range(observations):
            order = ("shell", "native") if observation % 2 == 0 else ("native", "shell")
            for method in order:
                reader = _shell_read if method == "shell" else _native_read
                elapsed = await _measure_workload(
                    reader,
                    session,
                    paths,
                    cycles=cycles,
                )
                samples[method].append(elapsed)
    finally:
        await session.aclose()

    shell = _summary(samples["shell"])
    native = _summary(samples["native"])
    shell_median = shell["median_s"]
    native_median = native["median_s"]
    delta_s = native_median - shell_median
    delta_pct = (delta_s / shell_median * 100.0) if shell_median else 0.0
    reads_per_observation = len(paths) * cycles

    return {
        "benchmark": "modal-supported-workspace-inspection",
        "git_head": _git_head(),
        "modal_version": importlib.metadata.version("modal"),
        "python_version": platform.python_version(),
        "startup_s_excluded_from_read_timings": startup_s,
        "observations_per_method": observations,
        "cycles_per_observation": cycles,
        "files_per_cycle": len(paths),
        "reads_per_observation": reads_per_observation,
        "bytes_per_cycle": sum(len(value.encode("utf-8")) for value in WORKLOAD_FILES.values()),
        "correctness_parity_verified": True,
        "shell": shell,
        "native": native,
        "native_minus_shell_median_s": delta_s,
        "native_minus_shell_median_pct": delta_pct,
        "samples_s": samples,
    }


def _print_markdown(result: dict[str, object]) -> None:
    shell = result["shell"]
    native = result["native"]
    assert isinstance(shell, dict)
    assert isinstance(native, dict)
    print("# Modal native-read workload benchmark")
    print()
    print(f"- Git head: `{result['git_head']}`")
    print(f"- Modal: `{result['modal_version']}`")
    print(f"- Python: `{result['python_version']}`")
    print(f"- Observations per method: {result['observations_per_method']}")
    print(f"- Reads per observation: {result['reads_per_observation']}")
    print(
        "- Startup excluded from read timings: "
        f"{float(result['startup_s_excluded_from_read_timings']):.3f}s"
    )
    print("- Correctness parity: verified before timing")
    print()
    print("| Method | Median workload | p95 | Min | Max |")
    print("| --- | ---: | ---: | ---: | ---: |")
    print(
        "| Shell baseline | "
        f"{float(shell['median_s']):.4f}s | {float(shell['p95_s']):.4f}s | "
        f"{float(shell['min_s']):.4f}s | {float(shell['max_s']):.4f}s |"
    )
    print(
        "| Native read | "
        f"{float(native['median_s']):.4f}s | {float(native['p95_s']):.4f}s | "
        f"{float(native['min_s']):.4f}s | {float(native['max_s']):.4f}s |"
    )
    print()
    print(
        "Native minus shell median: "
        f"{float(result['native_minus_shell_median_s']):+.4f}s "
        f"({float(result['native_minus_shell_median_pct']):+.1f}%)."
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--app-name",
        default="openai-agents-python-modal-read-benchmark",
        help="Modal app name to create or reuse.",
    )
    parser.add_argument(
        "--observations",
        type=int,
        default=8,
        help="Alternating timed observations per method.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=10,
        help="Workspace-inspection cycles per timed observation.",
    )
    parser.add_argument(
        "--warmup-cycles",
        type=int,
        default=2,
        help="Untimed warmup cycles per method.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for the machine-readable result.",
    )
    args = parser.parse_args()
    if args.observations < 2 or args.cycles < 1 or args.warmup_cycles < 0:
        parser.error("observations >= 2, cycles >= 1, and warmup-cycles >= 0 are required")

    result = await run_benchmark(
        app_name=args.app_name,
        observations=args.observations,
        cycles=args.cycles,
        warmup_cycles=args.warmup_cycles,
    )
    _print_markdown(result)
    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    asyncio.run(main())
