from __future__ import annotations

from pathlib import Path
from typing import Any

import openpyxl


FIELDS = [
    "result_name",
    "detection_id",
    "cue_lat",
    "cue_lon",
    "cue_alt",
    "tgt_lat",
    "tgt_lon",
    "tgt_alt",
    "t_datetime",
]


def find_repo_root(start: Path) -> Path:
    """Walk up from start to find project root containing '0_results/0_FINAL'; returns root path."""
    p = start.resolve()
    for parent in [p, *p.parents]:
        if (parent / "0_results" / "0_FINAL").is_dir():
            return parent
    raise FileNotFoundError("Could not find repo root containing '0_results/0_FINAL' when walking up from script CWD.")


def norm(s: Any) -> str:
    """Normalize header cell to trimmed string; empty -> ''."""
    return str(s).strip() if s is not None else ""


def iter_img_rows(xlsx_path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Read all rows from sheet ' Img' (or 'Img'); returns (sheet_headers, list of row dicts by header)."""
    wb = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
    try:
        if " Img" in wb.sheetnames:
            ws = wb[" Img"]
        elif "Img" in wb.sheetnames:
            ws = wb["Img"]
        else:
            raise KeyError(f"Sheet ' Img' (or 'Img') not found. Sheets: {wb.sheetnames}")

        rows = ws.iter_rows(values_only=True)
        header_row = next(rows, None)
        if not header_row:
            return ([], [])

        headers = [norm(h) for h in header_row]
        header_map = {h: i for i, h in enumerate(headers) if h}

        out: list[dict[str, Any]] = []
        for r in rows:
            if r is None:
                continue
            if all(v is None or (isinstance(v, str) and v.strip() == "") for v in r):
                continue
            row_dict: dict[str, Any] = {}
            for h, idx in header_map.items():
                row_dict[h] = r[idx] if idx < len(r) else None
            out.append(row_dict)

        return (headers, out)
    finally:
        wb.close()


def choose_results_xlsx(folder: Path) -> Path | None:
    """Pick the results Excel in a folder; returns path or None."""
    candidates = sorted(folder.glob("results_*.xlsx"))
    if candidates:
        return candidates[0]
    any_xlsx = sorted(folder.glob("*.xlsx"))
    return any_xlsx[0] if any_xlsx else None


def write_output_xlsx(out_path: Path, rows: list[list[Any]]) -> None:
    """Write rows (including header row) to a new xlsx file."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "all_results"
    for r in rows:
        ws.append(r)

    for col_idx, name in enumerate(rows[0], start=1):
        ws.cell(row=1, column=col_idx).font = openpyxl.styles.Font(bold=True)

    ws.freeze_panes = "A2"
    wb.save(out_path)
    wb.close()


def main() -> int:
    """Aggregate all rows from ' Img' of each TC_1x1* results xlsx into one output xlsx."""
    cwd = Path.cwd()
    root = find_repo_root(cwd)
    final_dir = root / "0_results" / "0_FINAL"

    tc_dirs = sorted([p for p in final_dir.iterdir() if p.is_dir() and p.name.startswith("TC_1x1")])
    if not tc_dirs:
        print(f"No folders starting with 'TC_1x1' found in: {final_dir}")
        return 1

    output_rows: list[list[Any]] = [FIELDS]
    missing_col_counts: dict[str, int] = {k: 0 for k in FIELDS if k not in ("result_name", "detection_id")}
    total_written = 0
    total_skipped_files = 0

    for d in tc_dirs:
        xlsx = choose_results_xlsx(d)
        if xlsx is None:
            total_skipped_files += 1
            continue

        try:
            _, img_rows = iter_img_rows(xlsx)
        except Exception as e:
            total_skipped_files += 1
            print(f"SKIP {d.name} ({xlsx.name}): {e}")
            continue

        # detection_id: try to use a column if present; otherwise use sequential index per file (1..N).
        for i, row_dict in enumerate(img_rows, start=1):
            det_id = None
            for cand in ("detection_id", "detectionId", "det_id", "id", "ID", "idx", "index"):
                if cand in row_dict and row_dict[cand] is not None:
                    det_id = row_dict[cand]
                    break
            if det_id is None:
                det_id = i

            out_row: list[Any] = []
            out_row.append(d.name)  # result_name

            out_row.append(det_id)  # detection_id

            for k in FIELDS[2:]:
                v = row_dict.get(k, None)
                if k != "t_datetime" and isinstance(v, str) and v.strip() == "":
                    v = None
                if k != "t_datetime" and k in row_dict is False:
                    missing_col_counts[k] += 1
                out_row.append(v)

            output_rows.append(out_row)
            total_written += 1

    out_path = cwd / "combined_results.xlsx"
    write_output_xlsx(out_path, output_rows)

    print(f"Wrote {total_written} rows to: {out_path}")
    if total_skipped_files:
        print(f"Skipped {total_skipped_files} folder(s) with no readable xlsx/' Img' sheet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
