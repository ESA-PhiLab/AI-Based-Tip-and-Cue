#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import re
from pathlib import Path
import sys

def export_tb_plots(repo_root: Path, run_dir: Path, out_subdir: str = "tb_exports") -> None:
    """export_tb_plots(repo_root, run_dir, out_subdir='tb_exports') -> None: Export TB scalars to PNG/CSV (best-effort)."""
    summary_dir = run_dir / "summary"
    logdir = summary_dir if summary_dir.exists() else run_dir
    outdir = run_dir / out_subdir
    try:
        cmd = [sys.executable, "tools/export_tb_curves.py", "--logdir", str(logdir), "--outdir", str(outdir)]
        rc = subprocess.call(cmd, cwd=str(repo_root))
        if rc != 0:
            print(f"[tb_export] WARNING: export_tb_curves.py exited with {rc} for {run_dir}", flush=True)
    except Exception as e:
        print(f"[tb_export] WARNING: failed exporting TB plots for {run_dir}: {e}", flush=True)

def cleanup_numbered_checkpoints(out_dir: Path) -> None:
    """cleanup_numbered_checkpoints(out_dir) -> None: Delete only checkpointXXXX.pth files in out_dir and out_dir/checkpoints."""
    pattern = re.compile(r"^checkpoint\d+\.pth$")

    for folder in [out_dir, out_dir / "checkpoints"]:
        if folder.exists():
            for p in folder.glob("checkpoint*.pth"):
                if pattern.match(p.name):
                    p.unlink()

def pick_checkpoint(out_dir: Path) -> Path:
    """pick_checkpoint(out_dir) -> Path: Pick checkpoint with highest validation AP50:95 from out_dir/log.txt; print candidates + decision."""
    import re

    ckpt_dir = out_dir / "checkpoints"
    log_path = out_dir / "log.txt"

    candidates: list[Path] = []
    if out_dir.exists():
        candidates.extend(out_dir.glob("*.pth"))
    if ckpt_dir.exists():
        candidates.extend(ckpt_dir.glob("*.pth"))
    candidates = list({p.resolve() for p in candidates})

    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {out_dir}")

    print("\n[pick_checkpoint] Candidate checkpoints found:")
    for c in sorted(candidates, key=lambda p: str(p)):
        try:
            print(f"  - {c}  (mtime={c.stat().st_mtime})")
        except Exception:
            print(f"  - {c}")

    # Metric-based selection using log.txt (preferred)
    if log_path.exists():
        best_epoch: int | None = None
        best_ap: float = -1.0

        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue

            stats = obj.get("test_coco_eval_bbox")
            if not isinstance(stats, (list, tuple)) or len(stats) < 1:
                continue

            try:
                ap = float(stats[1])  # AP50
                epoch = int(obj.get("epoch"))
            except Exception:
                continue

            if ap > best_ap:
                best_ap = ap
                best_epoch = epoch

        if best_epoch is not None:
            print(f"[pick_checkpoint] Best epoch from log.txt: {best_epoch}")
            print(f"[pick_checkpoint] Best AP50:95 from log.txt: {best_ap:.6f}")

            stop_epoch = None
            cfg_path = out_dir / "config" / "train_config.yml"
            if cfg_path.exists():
                txt = cfg_path.read_text(encoding="utf-8", errors="ignore")
                m = re.search(r"stop_epoch:\s*(\d+)", txt)
                if m:
                    try:
                        stop_epoch = int(m.group(1))
                    except Exception:
                        stop_epoch = None

            if stop_epoch is not None:
                print(f"[pick_checkpoint] stop_epoch inferred from config: {stop_epoch}")
            else:
                print("[pick_checkpoint] stop_epoch not found; will prefer best_stg1 if epoch appears stage-1, else fall back to best*.")

            # Prefer explicit stage files if present
            preferred: list[Path] = []
            if stop_epoch is not None and best_epoch >= stop_epoch:
                preferred = [p for p in candidates if p.name == "best_stg2.pth"]
                if not preferred:
                    preferred = [p for p in candidates if p.name == "best_stg2_local.pth"]
            elif stop_epoch is not None and best_epoch < stop_epoch:
                preferred = [p for p in candidates if p.name == "best_stg1.pth"]

            if preferred:
                chosen = preferred[0]
                print(
                    f"[pick_checkpoint] Selected (metric-based): {chosen} "
                    f"| best_epoch={best_epoch} | best_AP50:95={best_ap:.6f}"
                )
                return chosen

            # If stage files missing, fall back to any best* file by mtime
            best_files = [p for p in candidates if p.name.lower().startswith("best")]
            if best_files:
                chosen = sorted(best_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
                print(f"[pick_checkpoint] Selected (fallback best* by mtime): {chosen}")
                return chosen

    # Fallbacks if log.txt missing/unusable
    last = [p for p in candidates if p.name.lower() == "last.pth"]
    if last:
        chosen = last[0]
        print(f"[pick_checkpoint] Selected (last.pth): {chosen}")
        return chosen

    def _ckpt_step(p: Path) -> int:
        m = re.search(r"checkpoint(\d+)\.pth$", p.name)
        return int(m.group(1)) if m else -1

    numbered = [p for p in candidates if re.search(r"checkpoint(\d+)\.pth$", p.name)]
    if numbered:
        numbered.sort(key=_ckpt_step)
        chosen = numbered[-1]
        print(f"[pick_checkpoint] Selected (highest numbered checkpoint): {chosen}")
        return chosen

    chosen = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
    print(f"[pick_checkpoint] Selected (final fallback by mtime): {chosen}")
    return chosen

def read_json(path: str) -> dict[str, Any]:
    """read_json(path) -> dict[str,Any]: Read JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    """write_json(path, obj) -> None: Write JSON with mkdir."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: str, text: str) -> None:
    """write_text(path, text) -> None: Write text with mkdir."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def resolve_path(repo_root: Path, p: str) -> str:
    """resolve_path(repo_root, p) -> str: Resolve absolute or repo-relative path."""
    pp = Path(p)
    return str(pp.resolve()) if pp.is_absolute() else str((repo_root / pp).resolve())



def ensure_env_cuda_visible_devices(gpus: str) -> None:
    """ensure_env_cuda_visible_devices(gpus)->None: Set CUDA_VISIBLE_DEVICES only if not already set."""
    if str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip():
        return
    gpus = str(gpus).strip()
    if not gpus:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def run_and_tee(cmd: list[str], env: dict[str, str] | None, cwd: str | None, log_path: str) -> int:
    """run_and_tee(cmd, env, cwd, log_path) -> int: Stream stdout/stderr to console and append to log."""
    from datetime import datetime
    from pathlib import Path
    import subprocess

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 120 + "\n")
        f.write(f"[run_and_tee] {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"[run_and_tee] CMD: {' '.join(cmd)}\n")
        f.write("=" * 120 + "\n")
        f.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=cwd,
            text=True,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
            f.flush()
            print(line, end="")

        return int(proc.wait())