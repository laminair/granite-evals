"""enroot backend.

Images are imported once into a squashfs cache (``SAGE2_ENROOT_CACHE``, a
shared filesystem on BlueVela so every job reuses them) and each sandbox is a
throwaway ``enroot create`` of that squashfs. Commands run with
``enroot start --root --rw``: SWE-bench images expect to be root in /testbed.

Imports use the rootless helpers in :mod:`sage2_evals.sandbox.ovlfs`, because
enroot's own need capabilities BlueVela's compute nodes don't grant. Set
``SAGE2_ENROOT_ROOTLESS=0`` to use enroot's.
"""

from __future__ import annotations

import fcntl
import functools
import logging
import os
import re
import shlex
import subprocess
from pathlib import Path

from sage2_evals.sandbox import Sandbox, ovlfs

log = logging.getLogger(__name__)

ENROOT = os.environ.get("SAGE2_ENROOT", "enroot")


def enroot_uri(image: str) -> str:
    """``docker.io/org/img:tag`` -> ``docker://org/img:tag``; other registries
    use enroot's ``registry#path`` form."""
    first, _, rest = image.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        if first in ("docker.io", "index.docker.io", "registry-1.docker.io"):
            return f"docker://{rest}"
        return f"docker://{first}#{rest}"
    return f"docker://{image}"


def cache_dir() -> Path:
    d = Path(os.environ.get("SAGE2_ENROOT_CACHE", Path.home() / ".cache" / "sage2" / "enroot"))
    d.mkdir(parents=True, exist_ok=True)
    return d


@functools.cache
def _import_env() -> dict[str, str] | None:
    if os.environ.get("SAGE2_ENROOT_ROOTLESS", "1") == "0":
        return None
    return {**os.environ, "PATH": f"{ovlfs.helper_dir()}{os.pathsep}{os.environ.get('PATH', '')}"}


def ensure_squashfs(image: str) -> Path:
    """Import ``image`` into the cache unless present. Safe under concurrency:
    one importer per image, the rest wait on the lock."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", image)
    sqsh = cache_dir() / f"{safe}.sqsh"
    if sqsh.exists():
        return sqsh
    with open(sqsh.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not sqsh.exists():
            tmp = sqsh.with_suffix(".sqsh.partial")
            tmp.unlink(missing_ok=True)
            log.info("enroot import %s", image)
            subprocess.run([ENROOT, "import", "-o", str(tmp), enroot_uri(image)], check=True, env=_import_env())
            tmp.rename(sqsh)
    return sqsh


class EnrootSandbox(Sandbox):
    def start(self) -> None:
        sqsh = ensure_squashfs(self.image)
        subprocess.run(
            [ENROOT, "create", "--name", self.name, str(sqsh)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def _exec_argv(self, command: str, cwd: str, env: dict[str, str]) -> list[str]:
        argv = [ENROOT, "start", "--root", "--rw"]
        for key, value in env.items():
            argv += ["--env", f"{key}={value}"]
        return argv + [self.name, "bash", "-c", f"cd {shlex.quote(cwd)} && {command}"]

    def close(self) -> None:
        subprocess.run(
            [ENROOT, "remove", "--force", self.name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
