"""
Build omop_v54.duckdb from delphi-100k Parquet files.
Run once: uv run python scripts/extract_omop.py
"""

import zipfile
import duckdb
from pathlib import Path

ZIP_PATH = Path(r"C:\Users\mahan\AppData\Local\R\cache\CDMConnector\delphi-100k_5.4.zip")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACT_DIR  = PROJECT_ROOT / "data" / "raw" / "delphi-100k"
OUTPUT_PATH  = PROJECT_ROOT / "data" / "processed" / "omop_v54.duckdb"

REQUIRED_TABLES = [
    "person", "condition_occurrence", "drug_exposure",
    "visit_occurrence", "measurement", "observation",
    "procedure_occurrence", "concept", "vocabulary",
]


def extract_parquet(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/3] Extracting Parquet files from zip...")
    with zipfile.ZipFile(zip_path, "r") as z:
        parquet_files = [n for n in z.namelist() if n.endswith(".parquet")]
        for name in parquet_files:
            z.extract(name, extract_dir)
    print(f"      Extracted {len(parquet_files)} Parquet files → {extract_dir}")


def build_duckdb(extract_dir: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove stale DB if re-running
    if output_path.exists():
        output_path.unlink()

    print(f"\n[2/3] Building DuckDB from Parquet files...")
    con = duckdb.connect(str(output_path))

    parquet_dir = extract_dir / "delphi-100k"
    parquet_files = list(parquet_dir.glob("*.parquet"))

    for pf in sorted(parquet_files):
        table_name = pf.stem
        con.execute(f"""
            CREATE TABLE {table_name} AS
            SELECT * FROM read_parquet('{pf.as_posix()}')
        """)
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"      ✓ {table_name:<35} {count:>10,} rows")

    print(f"\n[3/3] Verifying required OMOP tables...")
    existing = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    missing  = [t for t in REQUIRED_TABLES if t not in existing]

    if missing:
        raise ValueError(f"Missing required OMOP tables: {missing}")

    print(f"      All {len(existing)} tables present ✓")
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"      Output size: {size_mb:.1f} MB")
    con.close()
    print(f"\n✅ Done → {output_path}")


if __name__ == "__main__":
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Zip not found: {ZIP_PATH}")

    extract_parquet(ZIP_PATH, EXTRACT_DIR)
    build_duckdb(EXTRACT_DIR, OUTPUT_PATH)