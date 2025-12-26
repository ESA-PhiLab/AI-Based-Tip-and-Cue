# obtain_data.py
from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import openpyxl


def _load_excel_cache(xlsx_path: Path) -> tuple[list[str], list[list[Any]]]:
    """Load header + all data rows from first sheet; returns (header, rows)."""
    wb = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
    try:
        ws = wb.active
        it = ws.iter_rows(values_only=True)
        header = [str(h).strip() for h in next(it)]
        rows = [list(r) for r in it if r and any(v is not None for v in r)]
        return header, rows
    finally:
        wb.close()


def pick_random_pose(xlsx_path: Path, pick_pose_seed: int) -> tuple:
    """Pick one random row; returns sat_lat, sat_lon, sat_alt, tgt_lat, tgt_lon, tgt_alt, datetime_utc."""
    random.seed(pick_pose_seed)

    header, rows = _load_excel_cache(xlsx_path)
    if not rows:
        raise ValueError("No data rows found in Excel file.")

    row = random.choice(rows)
    d = dict(zip(header, row))

    # Your sheet uses cue_* for satellite pose, tgt_* for target pose
    result_name = d["result_name"]
    detection_id = d["detection_id"]
    sat_lat = float(d["cue_lat"])
    sat_lon = float(d["cue_lon"])
    sat_alt = float(d["cue_alt"])
    tgt_lat = float(d["tgt_lat"])
    tgt_lon = float(d["tgt_lon"])
    tgt_alt = float(d["tgt_alt"])

    # Keep as string; worker_run.py already parses ISO with Z
    datetime_utc = str(d["t_datetime"])

    return result_name, detection_id, sat_lat, sat_lon, sat_alt, tgt_lat, tgt_lon, tgt_alt, datetime_utc

if __name__ == "__main__":
    xlsx = Path("combined_results.xlsx")

    for i in range(5):
        result_name, detection_id, cue_lat, cue_lon, cue_alt, tgt_lat, tgt_lon, tgt_alt, t_datetime = pick_random_pose(xlsx, pick_pose_seed=i)
        print(result_name, detection_id, cue_lat, cue_lon, cue_alt, tgt_lat, tgt_lon, tgt_alt, t_datetime)
