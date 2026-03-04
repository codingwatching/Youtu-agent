# ruff: noqa
from .utils import EnvUtils, setup_logging
from .tracing import setup_tracing

EnvUtils.assert_env(["UTU_LLM_TYPE", "UTU_LLM_MODEL"])
setup_logging(EnvUtils.get_env("UTU_LOG_LEVEL", "WARNING"))
setup_tracing()
