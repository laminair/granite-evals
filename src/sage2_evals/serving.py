"""Start and stop a local vLLM OpenAI-compatible server for one step.

Same shape as granite.build's run-bfcl.sh: the step owns its server, waits for
``/v1/models`` to answer, and tears the whole process group down on exit, so
a crashed eval never leaves GPUs held on the node.
"""

from __future__ import annotations

import logging
import os
import shlex
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

log = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    model: str
    served_model_name: str
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    # Granite 4.2 resolves both to its own parsers (qwen3_coder / nemotron_3);
    # see the model card's "Serving with vLLM" section.
    tool_call_parser: str = "auto"
    reasoning_parser: str = "auto"
    extra_args: list[str] = field(default_factory=list)
    port: int = 0
    """0 picks a free port."""
    startup_timeout_s: int = 1800


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class VLLMServer:
    def __init__(self, config: ServerConfig, log_path: Path):
        self.config = config
        self.log_path = log_path
        self.port = config.port or _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self._proc: subprocess.Popen | None = None

    def command(self) -> list[str]:
        c = self.config
        cmd = [
            "vllm", "serve", c.model,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--served-model-name", c.served_model_name,
            "--tensor-parallel-size", str(c.tensor_parallel_size),
            "--gpu-memory-utilization", str(c.gpu_memory_utilization),
            "--trust-remote-code",
        ]  # fmt: skip
        if c.max_model_len:
            cmd += ["--max-model-len", str(c.max_model_len)]
        if c.tool_call_parser:
            cmd += ["--enable-auto-tool-choice", "--tool-call-parser", c.tool_call_parser]
        if c.reasoning_parser:
            cmd += ["--reasoning-parser", c.reasoning_parser]
        return cmd + c.extra_args

    def __enter__(self) -> VLLMServer:
        cmd = self.command()
        log.info("starting vLLM: %s (log: %s)", shlex.join(cmd), self.log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        logfile = self.log_path.open("ab")
        self._proc = subprocess.Popen(
            cmd, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True
        )
        self._wait_ready()
        return self

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.config.startup_timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited with code {self._proc.returncode} before becoming ready; "
                    f"see {self.log_path}"
                )
            try:
                if httpx.get(f"{self.base_url}/models", timeout=5).status_code == 200:
                    log.info("vLLM ready at %s", self.base_url)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(5)
        self.__exit__(None, None, None)
        raise TimeoutError(f"vLLM not ready after {self.config.startup_timeout_s}s; see {self.log_path}")

    def __exit__(self, *exc) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        pgid = os.getpgid(self._proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            self._proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            self._proc.wait()
