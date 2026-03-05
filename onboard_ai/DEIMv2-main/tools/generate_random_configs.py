#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, List


LR_CHOICES = [1e-3, 1e-4, 1e-5]
BATCH_CHOICES = [8, 16, 32]
IOU_CHOICES = [0.0, 0.4, 0.8]
MOSAIC_PROB_CHOICES = [0.0, 0.4, 0.8]
RES_CHOICES = [256, 320]

N_GENERATE_DEFAULT = 30  # how many configs to generate when you just press Run in PyCharm
DEFAULT_BATCH = 32

DEFAULT_CLEAN_REL = [
    "results",
    "tb_exports",
    "collected_plots",
    "new_plots",
]


def repo_root() -> Path: return Path(__file__).resolve().parent.parent


def lr_to_tag(lr: float) -> str:
    if lr == 1e-3:
        return "lr3"
    if lr == 1e-4:
        return "lr4"
    if lr == 1e-5:
        return "lr5"
    return f"lr{lr:.0e}"


def pct_tag(x: float) -> int: return int(round(x * 100))


def balanced_pool(values: List[Any], n: int, rng: random.Random) -> List[Any]:
    reps = (n + len(values) - 1) // len(values)
    pool = (values * reps)[:n]
    rng.shuffle(pool)
    return pool


def fmt_float(x: float) -> str:
    s = f"{x:.12f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _is_within(child: Path, parent: Path) -> bool:
    """_is_within(child, parent) -> bool: True if child resolves under parent."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _wipe_dir_contents(d: Path, repo: Path) -> None:
    """_wipe_dir_contents(d, repo) -> None: Delete all contents under directory d (safe within repo)."""
    d = d.resolve()
    if not d.exists() or not d.is_dir():
        return
    if not _is_within(d, repo):
        raise RuntimeError(f"Refusing to clean outside repo: {d}")
    if d == repo.resolve():
        raise RuntimeError(f"Refusing to clean repo root itself: {d}")

    def _onerror(func, path, exc_info):
        try:
            p = Path(path)
            try:
                p.chmod(0o700)
            except Exception:
                pass
            func(path)
        except Exception:
            pass

    for p in d.iterdir():
        if p.is_symlink():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p, onerror=_onerror)
        else:
            try:
                p.chmod(0o600)
            except Exception:
                pass
            p.unlink()


def _remove_pycache(repo: Path) -> int:
    """_remove_pycache(repo) -> int: Remove __pycache__ folders under repo, skipping virtual envs."""
    skip_dir_names = {".venv", "venv", "env", ".env", "site-packages", "__pypackages__"}
    repo = repo.resolve()

    def _should_skip(p: Path) -> bool:
        parts = set(p.parts)
        return any(name in parts for name in skip_dir_names)

    def _onerror(func, path, exc_info):
        try:
            p = Path(path)
            try:
                p.chmod(0o700)
            except Exception:
                pass
            func(path)
        except Exception:
            pass

    n = 0
    for d in repo.rglob("__pycache__"):
        if not d.is_dir():
            continue
        if not _is_within(d, repo):
            continue
        if _should_skip(d):
            continue
        try:
            shutil.rmtree(d, onerror=_onerror)
            n += 1
        except Exception:
            # don't fail the whole script if Windows blocks something
            continue
    return n

def clean_repo_generated(repo: Path, rel_dirs: Iterable[str], extra: List[str]) -> None:
    """clean_repo_generated(repo, rel_dirs, extra) -> None: Wipe common generated artifact dirs inside repo."""
    cleaned = []
    for rel in list(rel_dirs) + list(extra):
        p = (repo / rel).resolve()
        if p.exists() and p.is_dir():
            _wipe_dir_contents(p, repo)
            cleaned.append(str(p))
        elif p.exists() and p.is_file():
            if not _is_within(p, repo):
                raise RuntimeError(f"Refusing to delete outside repo: {p}")
            p.unlink()
            cleaned.append(str(p))

    n_pycache = _remove_pycache(repo)
    if cleaned or n_pycache:
        print("[clean] wiped:")
        for x in cleaned:
            print(f"  - {x}")
        if n_pycache:
            print(f"  - removed __pycache__ dirs: {n_pycache}")
    else:
        print("[clean] nothing to wipe (targets not found).")


def clear_previous_configs(out_dir: Path, keep_name: str) -> int:
    """clear_previous_configs(out_dir, keep_name) -> int: Delete generated yml files in out_dir, keep keep_name."""
    if not out_dir.exists():
        return 0

    removed = 0
    gen_pat = re.compile(r"^\d+_lr.*_b\d+_iou\d+_mp\d+_res\d+\.yml$")
    alt_pat = re.compile(r"^lr.*_b\d+_iou\d+_mp\d+_res\d+\.yml$")

    for f in out_dir.iterdir():
        if not f.is_file():
            continue
        if f.name == keep_name:
            continue
        if gen_pat.match(f.name) or alt_pat.match(f.name):
            f.unlink()
            removed += 1
    return removed


def replace_optimizer_lrs(text: str, lr_i: float) -> str:
    """replace_optimizer_lrs(text, lr_i) -> str: Set optimizer.lr=lr_i and first two param-group lrs to lr_i/2."""
    lr_backbone = lr_i / 2.0

    opt_pat = r'(?ms)^optimizer:\s*\n.*?(?=^[A-Za-z_][A-Za-z0-9_]*:\s|\Z)'
    m = re.search(opt_pat, text)
    if not m:
        raise RuntimeError("Could not locate optimizer: block.")
    block = m.group(0)

    lr_pg_pat = r'(?m)^(\s{6,}lr:\s*)([0-9.+-eE]+)\s*$'
    pg_hits = list(re.finditer(lr_pg_pat, block))
    if len(pg_hits) < 2:
        raise RuntimeError("Could not find two param-group lr lines to replace.")

    def _pg_repl(match: re.Match, counter: List[int]) -> str:
        counter[0] += 1
        if counter[0] <= 2:
            return match.group(1) + fmt_float(lr_backbone)
        return match.group(0)

    counter = [0]
    block = re.sub(lr_pg_pat, lambda mm: _pg_repl(mm, counter), block)

    base_lr_pat = r'(?m)^(\s{2}lr:\s*)([0-9.+-eE]+)\s*$'
    base_hits = list(re.finditer(base_lr_pat, block))
    if not base_hits:
        raise RuntimeError("Could not find optimizer base lr line (2-space indent).")

    last = base_hits[-1]
    block = block[:last.start()] + re.sub(base_lr_pat, r"\g<1>" + fmt_float(lr_i), block[last.start():], count=1)

    return text[:m.start()] + block + text[m.end():]


def replace_train_total_batch_size(text: str, batch: int) -> str:
    """replace_train_total_batch_size(text, batch) -> str: Set train_dataloader.total_batch_size=batch."""
    pat = r'(?ms)^(train_dataloader:\s*\n[\s\S]*?)(?=^[A-Za-z_][A-Za-z0-9_]*:\s|\Z)'
    m = re.search(pat, text)
    if not m:
        raise RuntimeError("Could not locate train_dataloader: block.")
    block = m.group(0)

    tb_pat = r'(?m)^(\s{2}total_batch_size:\s*)(\d+)\s*$'
    if not re.search(tb_pat, block):
        raise RuntimeError("Could not find train_dataloader.total_batch_size to replace.")
    block = re.sub(tb_pat, r"\g<1>" + str(int(batch)), block, count=1)

    return text[:m.start()] + block + text[m.end():]


def replace_ema_and_warmup(text: str, f: float) -> str:
    """replace_ema_and_warmup(text, f) -> str: decay=1-(1-decay0)*f, warmups=round(1000/f), warmup_duration=round(500/f)."""
    ema_warmups = int(round(1000.0 / f))
    warmup_duration = int(round(500.0 / f))

    ema_pat = r'(?ms)^ema:\s*(?:#.*)?\n.*?(?=^[A-Za-z_][A-Za-z0-9_]*:\s|\Z)'
    m = re.search(ema_pat, text)
    if not m:
        raise RuntimeError("Could not locate ema: block.")
    ema_block = m.group(0)

    decay_pat = r'(?m)^(\s{2,}decay:\s*)([0-9.+-eE]+)\s*$'
    decay_m = re.search(decay_pat, ema_block)
    if not decay_m:
        raise RuntimeError("Could not find ema.decay line to replace.")
    base_decay = float(decay_m.group(2))

    new_decay = 1.0 - (1.0 - base_decay) * f
    if new_decay <= 0.0 or new_decay >= 1.0:
        raise RuntimeError(f"Computed invalid ema.decay={new_decay} from base_decay={base_decay} and f={f}.")

    ema_block = re.sub(decay_pat, r"\g<1>" + fmt_float(new_decay), ema_block, count=1)

    warmups_pat = r'(?m)^(\s{2,}warmups:\s*)(\d+)\s*$'
    if not re.search(warmups_pat, ema_block):
        raise RuntimeError("Could not find ema.warmups line to replace.")
    ema_block = re.sub(warmups_pat, r"\g<1>" + str(ema_warmups), ema_block, count=1)

    text = text[:m.start()] + ema_block + text[m.end():]

    warmup_pat = r'(?ms)^lr_warmup_scheduler:\s*\n.*?(?=^[A-Za-z_][A-Za-z0-9_]*:\s|\Z)'
    m2 = re.search(warmup_pat, text)
    if not m2:
        raise RuntimeError("Could not locate lr_warmup_scheduler: block.")
    warm_block = m2.group(0)

    dur_pat = r'(?m)^(\s{2,}warmup_duration:\s*)(\d+)\s*$'
    if not re.search(dur_pat, warm_block):
        raise RuntimeError("Could not find lr_warmup_scheduler.warmup_duration line to replace.")
    warm_block = re.sub(dur_pat, r"\g<1>" + str(warmup_duration), warm_block, count=1)

    return text[:m2.start()] + warm_block + text[m2.end():]


def replace_iou_crop_p(text: str, p: float) -> str:
    """replace_iou_crop_p(text, p) -> str: Replace RandomIoUCrop inline p value."""
    pat = r'(\btype:\s*RandomIoUCrop\b[\s\S]*?\bp:\s*)([0-9.+-eE]+)(\s*(?:,|\}))'
    if not re.search(pat, text):
        raise RuntimeError("Could not find RandomIoUCrop op with a 'p:' field to replace.")
    return re.sub(pat, r"\g<1>" + fmt_float(p) + r"\g<3>", text, count=1)


def replace_mosaic_probability(text: str, mp: float) -> str:
    """replace_mosaic_probability(text, mp) -> str: Replace Mosaic inline probability."""
    pat = r'(\btype:\s*Mosaic\b[\s\S]*?\bprobability:\s*)([0-9.+-eE]+)(\s*,)'
    if not re.search(pat, text):
        raise RuntimeError("Could not find Mosaic op with a 'probability:' field to replace.")
    return re.sub(pat, r"\g<1>" + fmt_float(mp) + r"\g<3>", text, count=1)


def replace_resolution_everywhere(text: str, res: int) -> str:
    """replace_resolution_everywhere(text, res) -> str: Replace all [256,256], Mosaic output_size, collate_fn.base_size."""
    text = re.sub(r'\[\s*256\s*,\s*256\s*\]', f'[{res}, {res}]', text)

    out_size = int(res // 2)
    pat_mosaic = r'(\{type:\s*Mosaic,\s*output_size:\s*)(\d+)(,)'
    if not re.search(pat_mosaic, text):
        raise RuntimeError("Could not find Mosaic output_size to replace.")
    text = re.sub(pat_mosaic, r"\g<1>" + str(out_size) + r"\g<3>", text, count=1)

    pat_base = r'(?m)^(\s*base_size:\s*)256\s*$'
    if not re.search(pat_base, text):
        raise RuntimeError("Could not find collate_fn.base_size: 256 to replace.")
    text = re.sub(pat_base, r"\g<1>" + str(res), text, count=1)

    return text


def remove_bad_dataset_keys_only(text: str) -> str:
    """remove_bad_dataset_keys_only(text) -> str: Remove dataset-level keys that crash CocoDetection (indent >=4)."""
    text = re.sub(r'(?m)^\s{4,}total_batch_size:\s*\d+\s*\n', "", text)
    text = re.sub(r'(?m)^\s{4,}mosaic_prob:\s*[0-9.+-eE]+\s*\n', "", text)
    return text


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate balanced random hyperparameter configs (layout-preserving).")
    ap.add_argument("--n", type=int, default=N_GENERATE_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="configs/hyperparameter_search")
    ap.add_argument("--base_config", type=str, default="configs/hyperparameter_search/S_DEFAULT.yml",
                    help="Path to the default config that will be copied/edited (this file is NOT deleted).")

    ap.add_argument("--no_clean_repo", action="store_true",
                    help="Disable cleaning generated repo artifacts (results/tb_exports/etc).")
    ap.add_argument("--clean_extra", nargs="*", default=[],
                    help="Extra repo-relative paths to wipe (dirs wiped by contents, files deleted). Example: logs runs")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))

    repo = repo_root()
    base_path = (repo / args.base_config).resolve()
    out_dir = (repo / args.out_dir).resolve()

    if not base_path.exists():
        raise SystemExit(f"[gen] base_config not found: {base_path}")

    if not args.no_clean_repo:
        clean_repo_generated(repo, DEFAULT_CLEAN_REL, list(args.clean_extra))

    out_dir.mkdir(parents=True, exist_ok=True)
    removed = clear_previous_configs(out_dir, keep_name=base_path.name)
    if removed:
        print(f"[clean] removed generated configs in {out_dir}: {removed}")
    else:
        print(f"[clean] no generated configs removed in {out_dir}")

    base_text = base_path.read_text(encoding="utf-8")
    manifest_names: list[str] = []

    lr_pool = balanced_pool(LR_CHOICES, int(args.n), rng)
    bs_pool = balanced_pool(BATCH_CHOICES, int(args.n), rng)
    iou_pool = balanced_pool(IOU_CHOICES, int(args.n), rng)
    mp_pool = balanced_pool(MOSAIC_PROB_CHOICES, int(args.n), rng)
    res_pool = balanced_pool(RES_CHOICES, int(args.n), rng)

    counts_lr: dict[float, int] = {}
    counts_bs: dict[int, int] = {}
    counts_iou: dict[float, int] = {}
    counts_mp: dict[float, int] = {}
    counts_res: dict[int, int] = {}

    for i in range(int(args.n)):
        lr_choice = float(lr_pool[i])
        batch = int(bs_pool[i])
        iou = float(iou_pool[i])
        mp = float(mp_pool[i])
        res = int(res_pool[i])

        counts_lr[lr_choice] = counts_lr.get(lr_choice, 0) + 1
        counts_bs[batch] = counts_bs.get(batch, 0) + 1
        counts_iou[iou] = counts_iou.get(iou, 0) + 1
        counts_mp[mp] = counts_mp.get(mp, 0) + 1
        counts_res[res] = counts_res.get(res, 0) + 1

        f = float(batch) / float(DEFAULT_BATCH)
        lr_i = lr_choice * f

        t = base_text
        t = replace_optimizer_lrs(t, lr_i=lr_i)
        t = replace_train_total_batch_size(t, batch=batch)
        t = replace_ema_and_warmup(t, f=f)
        t = replace_iou_crop_p(t, p=iou)
        t = replace_mosaic_probability(t, mp=mp)
        t = replace_resolution_everywhere(t, res=res)
        t = remove_bad_dataset_keys_only(t)

        name = f"{i + 1:02d}_{lr_to_tag(lr_choice)}_b{batch}_iou{pct_tag(iou)}_mp{pct_tag(mp)}_res{res}.yml"
        out_path = out_dir / name
        out_path.write_text(t, encoding="utf-8")
        manifest_names.append(Path(name).stem)

        print(f"[gen] wrote {out_path} | f={fmt_float(f)} lr_choice={fmt_float(lr_choice)} lr_i={fmt_float(lr_i)}")

    def _print_counts(title: str, d: dict[Any, int], key_fn=None) -> None:
        print(f"\n[summary] {title}:")
        items = list(d.items())
        if key_fn is not None:
            items.sort(key=lambda kv: key_fn(kv[0]))
        else:
            items.sort(key=lambda kv: str(kv[0]))
        for k, v in items:
            print(f"  - {k}: {v}")
        print(f"  total: {sum(d.values())}")

    _print_counts("LR choices", counts_lr, key_fn=float)
    _print_counts("Batch sizes", counts_bs, key_fn=int)
    _print_counts("IoU crop p", counts_iou, key_fn=float)
    _print_counts("Mosaic probability", counts_mp, key_fn=float)
    _print_counts("Resolution", counts_res, key_fn=int)
    print(f"\n[summary] n={int(args.n)}")

    manifest_file = out_dir / "generated_configs.txt"
    manifest_file.write_text("\n".join(manifest_names) + "\n", encoding="utf-8")
    print(f"\n[gen] wrote config list -> {manifest_file}")

if __name__ == "__main__":
    main()