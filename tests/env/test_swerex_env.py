"""Tests for SWERexEnv with mocked SWE-ReX runtime."""

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

from utu.env.swerex_env import SWERexEnv, _parse_json, _truncate

# ---------------------------------------------------------------------------
# Mock swerex module so imports inside SWERexEnv methods work without swerex.
# ---------------------------------------------------------------------------

def _install_swerex_mock():
    """Install a mock swerex package into sys.modules."""
    # Only install if not already present.
    if "swerex" in sys.modules:
        return

    swerex = types.ModuleType("swerex")
    swerex_runtime = types.ModuleType("swerex.runtime")
    swerex_runtime_abstract = types.ModuleType("swerex.runtime.abstract")
    swerex_deployment = types.ModuleType("swerex.deployment")
    swerex_deployment_ags = types.ModuleType("swerex.deployment.ags")
    swerex_deployment_remote = types.ModuleType("swerex.deployment.remote")

    # Request/action classes — simple callables that store kwargs as attributes.
    class _SimpleModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    swerex_runtime_abstract.CreateBashSessionRequest = _SimpleModel
    swerex_runtime_abstract.BashAction = _SimpleModel
    swerex_runtime_abstract.ReadFileRequest = _SimpleModel
    swerex_runtime_abstract.WriteFileRequest = _SimpleModel

    swerex_deployment_ags.TencentAGSDeployment = MagicMock
    swerex_deployment_remote.RemoteDeployment = MagicMock

    # Wire up the module hierarchy.
    swerex.runtime = swerex_runtime
    swerex.deployment = swerex_deployment
    swerex_runtime.abstract = swerex_runtime_abstract
    swerex_deployment.ags = swerex_deployment_ags
    swerex_deployment.remote = swerex_deployment_remote

    sys.modules["swerex"] = swerex
    sys.modules["swerex.runtime"] = swerex_runtime
    sys.modules["swerex.runtime.abstract"] = swerex_runtime_abstract
    sys.modules["swerex.deployment"] = swerex_deployment
    sys.modules["swerex.deployment.ags"] = swerex_deployment_ags
    sys.modules["swerex.deployment.remote"] = swerex_deployment_remote


_install_swerex_mock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env(**overrides) -> SWERexEnv:
    config = {"deployment_type": "remote", "host": "localhost", "port": 8000, "workspace": "/repo"}
    config.update(overrides)
    return SWERexEnv(config)


def _mock_runtime():
    """Create a mock runtime with standard methods."""
    rt = AsyncMock()

    # create_session returns an object with .output
    session_resp = MagicMock()
    session_resp.output = "session created"
    rt.create_session.return_value = session_resp

    # run_in_session returns a BashObservation-like object
    obs = MagicMock()
    obs.output = "command output"
    obs.exit_code = 0
    rt.run_in_session.return_value = obs

    # read_file returns a ReadFileResponse-like object
    read_resp = MagicMock()
    read_resp.content = "file content"
    rt.read_file.return_value = read_resp

    # write_file returns normally
    rt.write_file.return_value = MagicMock()

    return rt


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_parse_json_valid(self):
        assert _parse_json('{"command": "ls"}') == {"command": "ls"}

    def test_parse_json_empty(self):
        assert _parse_json("") == {}

    def test_parse_json_invalid(self):
        assert _parse_json("not json") == {}

    def test_truncate_short(self):
        assert _truncate("hello") == "hello"

    def test_truncate_long(self):
        long_text = "x" * 60_000
        result = _truncate(long_text)
        assert len(result) < 60_000
        assert "truncated" in result


# ---------------------------------------------------------------------------
# SWERexEnv tests
# ---------------------------------------------------------------------------


class TestSWERexEnvInit:
    def test_default_config(self):
        env = SWERexEnv()
        assert env._session_name == "default"
        assert env._workspace == "/"
        assert env._bash_timeout == 120

    def test_custom_config(self):
        env = _make_env(session_name="test", workspace="/app", bash_timeout=60)
        assert env._session_name == "test"
        assert env._workspace == "/app"
        assert env._bash_timeout == 60


class TestSWERexEnvBuild:
    @patch("utu.env.swerex_env.SWERexEnv._build_remote", new_callable=AsyncMock)
    async def test_build_remote(self, mock_build_remote):
        env = _make_env(deployment_type="remote")
        mock_runtime = _mock_runtime()

        async def side_effect():
            env._deployment = MagicMock()
            env._deployment.runtime = mock_runtime

        mock_build_remote.side_effect = side_effect
        await env.build()

        mock_build_remote.assert_called_once()
        assert env._runtime is not None
        mock_runtime.create_session.assert_called_once()
        # Should have run cd /repo (workspace != "/")
        mock_runtime.run_in_session.assert_called()

    @patch("utu.env.swerex_env.SWERexEnv._build_ags", new_callable=AsyncMock)
    async def test_build_ags(self, mock_build_ags):
        env = _make_env(deployment_type="ags", workspace="/")
        mock_runtime = _mock_runtime()

        async def side_effect():
            env._deployment = MagicMock()
            env._deployment.runtime = mock_runtime

        mock_build_ags.side_effect = side_effect
        await env.build()

        mock_build_ags.assert_called_once()
        # workspace == "/", so only the pager-disable command runs (no cd)
        assert mock_runtime.run_in_session.call_count == 1

    @patch("utu.env.swerex_env.SWERexEnv._build_remote", new_callable=AsyncMock)
    async def test_build_with_post_startup_commands(self, mock_build_remote):
        env = _make_env(workspace="/", post_startup_commands=["export FOO=bar", "echo hi"])
        mock_runtime = _mock_runtime()

        async def side_effect():
            env._deployment = MagicMock()
            env._deployment.runtime = mock_runtime

        mock_build_remote.side_effect = side_effect
        await env.build()

        # 1 pager-disable + 2 post-startup commands
        assert mock_runtime.run_in_session.call_count == 3


class TestSWERexEnvCleanup:
    async def test_cleanup(self):
        env = _make_env()
        mock_deployment = AsyncMock()
        env._deployment = mock_deployment
        env._runtime = MagicMock()
        env._tools_cache = [MagicMock()]

        await env.cleanup()

        mock_deployment.stop.assert_called_once()
        assert env._deployment is None
        assert env._runtime is None
        assert env._tools_cache is None

    async def test_cleanup_no_deployment(self):
        env = _make_env()
        # Should not raise
        await env.cleanup()


class TestSWERexEnvTools:
    async def test_get_tools_returns_four(self):
        env = _make_env()
        env._runtime = _mock_runtime()
        tools = await env.get_tools()
        assert len(tools) == 4
        names = {t.name for t in tools}
        assert names == {"bash", "read_file", "write_file", "edit_file"}

    async def test_get_tools_caches(self):
        env = _make_env()
        env._runtime = _mock_runtime()
        tools1 = await env.get_tools()
        tools2 = await env.get_tools()
        assert tools1 is tools2


class TestSWERexEnvRunBash:
    async def test_run_bash_success(self):
        env = _make_env()
        env._runtime = _mock_runtime()
        result = await env._run_bash("echo hello")
        assert result == "command output"

    async def test_run_bash_nonzero_exit(self):
        env = _make_env()
        rt = _mock_runtime()
        obs = MagicMock()
        obs.output = "error output"
        obs.exit_code = 1
        rt.run_in_session.return_value = obs
        env._runtime = rt

        result = await env._run_bash("false")
        assert "exit code: 1" in result

    async def test_run_bash_no_runtime(self):
        env = _make_env()
        result = await env._run_bash("echo hi")
        assert "Error" in result

    async def test_run_bash_exception(self):
        env = _make_env()
        rt = _mock_runtime()
        rt.run_in_session.side_effect = RuntimeError("connection lost")
        env._runtime = rt

        result = await env._run_bash("ls")
        assert "Error" in result
        assert "connection lost" in result


class TestSWERexEnvReadFile:
    async def test_read_file_success(self):
        env = _make_env()
        env._runtime = _mock_runtime()
        result = await env._read_file("/tmp/test.txt")
        assert result == "file content"

    async def test_read_file_error(self):
        env = _make_env()
        rt = _mock_runtime()
        rt.read_file.side_effect = FileNotFoundError("/tmp/missing.txt")
        env._runtime = rt

        result = await env._read_file("/tmp/missing.txt")
        assert "Error" in result


class TestSWERexEnvWriteFile:
    async def test_write_file_success(self):
        env = _make_env()
        env._runtime = _mock_runtime()
        result = await env._write_file("/tmp/test.txt", "hello")
        assert "Successfully" in result

    async def test_write_file_error(self):
        env = _make_env()
        rt = _mock_runtime()
        rt.write_file.side_effect = PermissionError("read-only")
        env._runtime = rt

        result = await env._write_file("/tmp/test.txt", "hello")
        assert "Error" in result


class TestSWERexEnvEditFile:
    async def test_edit_file_success(self):
        env = _make_env()
        rt = _mock_runtime()
        read_resp = MagicMock()
        read_resp.content = "hello world"
        rt.read_file.return_value = read_resp
        env._runtime = rt

        result = await env._edit_file("/tmp/test.txt", "hello", "goodbye")
        assert "Successfully" in result

        # Verify write was called with replaced content
        write_call = rt.write_file.call_args[0][0]
        assert write_call.content == "goodbye world"

    async def test_edit_file_not_found(self):
        env = _make_env()
        rt = _mock_runtime()
        read_resp = MagicMock()
        read_resp.content = "hello world"
        rt.read_file.return_value = read_resp
        env._runtime = rt

        result = await env._edit_file("/tmp/test.txt", "missing", "replacement")
        assert "not found" in result

    async def test_edit_file_multiple_matches(self):
        env = _make_env()
        rt = _mock_runtime()
        read_resp = MagicMock()
        read_resp.content = "hello hello hello"
        rt.read_file.return_value = read_resp
        env._runtime = rt

        result = await env._edit_file("/tmp/test.txt", "hello", "bye")
        assert "3 times" in result


if __name__ == '__main__':
    import pytest
    # test current file
    pytest.main([__file__])
