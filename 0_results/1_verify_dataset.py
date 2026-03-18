from pathlib import Path
import pandas as pd


def count_images_and_excel_rows(root_dir: Path) -> None:
    """Loop folders, compare image count vs Excel 'Img' rows, and summarize mismatches."""

    mismatches = []

    for run_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        run_name = run_dir.name

        # --- Count images ---
        image_dir = run_dir / "satellite_images"
        if image_dir.exists():
            image_count = sum(
                1
                for f in image_dir.iterdir()
                if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
            )
        else:
            image_count = None

        # --- Find Excel file ---
        excel_files = list(run_dir.glob("results_*.xlsx"))

        if len(excel_files) == 0:
            excel_rows = None
        else:
            excel_path = excel_files[0]

            try:
                df = pd.read_excel(excel_path, sheet_name="Img")
                df = df.dropna(how="all")
                excel_rows = len(df)
            except Exception:
                excel_rows = None

        # --- Print per case ---
        print(f"{run_name}: images={image_count}, excel_rows={excel_rows}")

        # --- Track mismatch ---
        if image_count != excel_rows:
            mismatches.append((run_name, image_count, excel_rows))

    # --- Summary ---
    print("\n=== SUMMARY ===")

    if len(mismatches) == 0:
        print("All folders match.")
    else:
        print(f"Mismatches found: {len(mismatches)}\n")
        for name, img, rows in mismatches:
            print(f"{name}: images={img}, excel_rows={rows}")


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent / "FINAL_RESULTS"
    count_images_and_excel_rows(ROOT)