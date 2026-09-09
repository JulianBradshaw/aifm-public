"""Remote-execution helpers for the covert pipeline.

Model rollouts run on the `aifm-runpod` box (per project convention: python3.13).
These wrap rsync/ssh so the orchestrator can build a dataset remotely and pull it
back, with one function call. Pure stdlib (subprocess).
"""
from __future__ import annotations
import os
import subprocess
from pathlib import Path

# Environment-specific (not secret — auth is via the SSH alias + key in ~/.ssh/config).
# Override via env vars if the box/path/interpreter changes; defaults match the
# current aifm-runpod setup.
HOST = os.environ.get("COVERT_REMOTE_HOST", "aifm-runpod")
REMOTE_ROOT = os.environ.get("COVERT_REMOTE_ROOT", "~/covert-pause-exp")  # repo copy on the box
PY = os.environ.get("COVERT_REMOTE_PY", "python3.13")
# noise to strip from remote stdout (ssh banner + solver chatter)
_NOISE = ("quantum", "vulnerable", "server may", "openssh", "post-quantum", "lsoda",
          "in above", "it will not", "such that", "step size", "continue anyway",
          "corrector", "abs(h)", "INFO:", "WARNING:", "DEBUG:")


def _clean(s: str) -> str:
    return "\n".join(ln for ln in s.splitlines() if not any(n in ln for n in _NOISE))


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def push(local: str, remote_rel: str) -> None:
    """rsync a local file/dir to REMOTE_ROOT/remote_rel."""
    dest = f"{HOST}:{REMOTE_ROOT}/{remote_rel}"
    r = run(["rsync", "-az", local, dest])
    if r.returncode:
        raise RuntimeError(f"push {local} -> {dest} failed:\n{_clean(r.stderr)}")


def push_tree(local_dir: str, remote_rel: str, excludes=()) -> None:
    """rsync a whole directory tree (creating remote subdirs). Trailing-slash form."""
    args = ["rsync", "-az"]
    for e in excludes:
        args += ["--exclude", e]
    args += [local_dir.rstrip("/") + "/", f"{HOST}:{REMOTE_ROOT}/{remote_rel.rstrip('/')}/"]
    r = run(args)
    if r.returncode:
        raise RuntimeError(f"push_tree {local_dir} failed:\n{_clean(r.stderr)}")


def pull(remote_rel: str, local: str) -> None:
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    r = run(["rsync", "-az", f"{HOST}:{REMOTE_ROOT}/{remote_rel}", local])
    if r.returncode:
        raise RuntimeError(f"pull {remote_rel} -> {local} failed:\n{_clean(r.stderr)}")


def ssh(remote_cmd: str, env: str = "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1") -> str:
    """Run a shell command in REMOTE_ROOT; return cleaned stdout. Raises on nonzero."""
    full = f"cd {REMOTE_ROOT} && {env} {remote_cmd}"
    r = run(["ssh", HOST, full])
    out = _clean((r.stdout or "") + (r.stderr or ""))
    if r.returncode:
        raise RuntimeError(f"ssh failed ({r.returncode}):\n{out[-2000:]}")
    return out
