"""SWERexEnv integration test — real AGS sandbox.

Verifies the full lifecycle: build → bash/read/write/edit → cleanup.

Usage:
    python tests/env/test_swerex_env_integration.py
    python tests/env/test_swerex_env_integration.py --config configs/env/swerex.yaml
"""

import argparse
import asyncio
import logging
import sys
import traceback

from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("test_swerex_integration")

# ANSI helpers
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(msg: str):
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str):
    print(f"  {RED}✗{RESET} {msg}")


def section(msg: str):
    print(f"\n{BOLD}▶ {msg}{RESET}")


async def main(config_path: str):
    # ── Load config ──────────────────────────────────────────────────
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    log.info("Loaded config from %s", config_path)

    from utu.env.swerex_env import SWERexEnv

    env = SWERexEnv(cfg.get("config", cfg))

    passed, failed = 0, 0

    def check(condition: bool, label: str, detail: str = ""):
        nonlocal passed, failed
        if condition:
            ok(label)
            passed += 1
        else:
            fail(f"{label}  {detail}")
            failed += 1

    # ── 1. Build ─────────────────────────────────────────────────────
    section("build()")
    try:
        await env.build()
        ok("build succeeded")
        passed += 1
    except Exception as e:
        fail(f"build failed: {e}")
        traceback.print_exc()
        return 1

    try:
        # ── 2. get_tools ─────────────────────────────────────────────
        section("get_tools()")
        tools = await env.get_tools()
        check(len(tools) == 4, f"got {len(tools)} tools (expected 4)")
        tool_names = {t.name for t in tools}
        check(tool_names == {"bash", "read_file", "write_file", "edit_file"}, f"tool names: {tool_names}")

        # ── 3. bash — basic commands ─────────────────────────────────
        section("bash: basic commands")

        result = await env._run_bash("echo hello-swerex")
        check("hello-swerex" in result, f"echo: {result.strip()!r}")

        result = await env._run_bash("whoami")
        check(len(result.strip()) > 0, f"whoami: {result.strip()!r}")

        result = await env._run_bash("python3 --version")
        check("Python" in result, f"python3 --version: {result.strip()!r}")

        # ── 4. bash — stateful session (cd persists) ─────────────────
        section("bash: stateful session")

        await env._run_bash("mkdir -p /tmp/swerex_test")
        await env._run_bash("cd /tmp/swerex_test")
        result = await env._run_bash("pwd")
        check("/tmp/swerex_test" in result, f"pwd after cd: {result.strip()!r}")

        # ── 5. write_file ────────────────────────────────────────────
        section("write_file")

        test_content = "line1\nline2\nline3\n"
        result = await env._write_file("/tmp/swerex_test/test.txt", test_content)
        check("Successfully" in result, f"write_file: {result.strip()!r}")

        # ── 6. read_file ─────────────────────────────────────────────
        section("read_file")

        result = await env._read_file("/tmp/swerex_test/test.txt")
        check(result == test_content, f"read_file: {result!r}")

        # read non-existent file
        result = await env._read_file("/tmp/swerex_test/nonexistent.txt")
        check("Error" in result, f"read missing file: {result.strip()!r}")

        # ── 7. edit_file ─────────────────────────────────────────────
        section("edit_file")

        result = await env._edit_file("/tmp/swerex_test/test.txt", "line2", "LINE_TWO")
        check("Successfully" in result, f"edit_file: {result.strip()!r}")

        result = await env._read_file("/tmp/swerex_test/test.txt")
        check("LINE_TWO" in result and "line2" not in result, f"verify edit: {result!r}")

        # edit non-matching string
        result = await env._edit_file("/tmp/swerex_test/test.txt", "XXXXXX", "YYYYYY")
        check("not found" in result, f"edit missing str: {result.strip()!r}")

        # ── 8. bash — run a Python snippet ───────────────────────────
        section("bash: python snippet")

        result = await env._run_bash("python3 -c \"print(sum(range(101)))\"")
        check("5050" in result, f"python sum(0..100): {result.strip()!r}")

        # ── 9. bash — exit code handling ─────────────────────────────
        section("bash: exit code handling")

        result = await env._run_bash("exit 42")
        check("exit code: 42" in result, f"exit 42: {result.strip()!r}")

        # ── 10. cleanup ──────────────────────────────────────────────
        section("cleanup & re-cleanup")
        await env._run_bash("rm -rf /tmp/swerex_test")

    finally:
        # Always cleanup
        try:
            await env.cleanup()
            ok("cleanup succeeded")
            passed += 1
        except Exception as e:
            fail(f"cleanup failed: {e}")
            failed += 1

        # Double cleanup should be safe
        await env.cleanup()
        ok("double cleanup safe")
        passed += 1

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    total = passed + failed
    color = GREEN if failed == 0 else RED
    print(f"{color}{BOLD}{passed}/{total} checks passed{RESET}")
    if failed > 0:
        print(f"{RED}{failed} checks FAILED{RESET}")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SWERexEnv integration test")
    parser.add_argument("--config", default="configs/env/swerex.yaml", help="Path to swerex env config YAML")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.config)))
