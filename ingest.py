"""Convert the REES46 monthly event CSVs (~110M rows, ~14 GB) to typed, sorted Parquet with DuckDB."""
import glob
from pathlib import Path

import duckdb

HERE = Path(__file__).parent
DATA = HERE / "data"


def main():
    (DATA / "tmp").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='4GB'; SET threads=4; SET temp_directory='{DATA / 'tmp'}'; SET preserve_insertion_order=false")
    files = sorted(glob.glob(str(DATA / "2019-*.csv")))
    if not files:
        raise SystemExit("No data/2019-*.csv files. Run ./download_data.sh first.")
    for f in files:
        out = DATA / (Path(f).stem + ".parquet")
        print("converting", f)
        con.execute(f"""
        COPY (
          SELECT CAST(strptime(replace(event_time, ' UTC', ''), '%Y-%m-%d %H:%M:%S') AS TIMESTAMP) AS event_time,
                 event_type,
                 CAST(product_id AS BIGINT)  AS product_id,
                 CAST(category_id AS BIGINT) AS category_id,
                 category_code,
                 coalesce(split_part(category_code, '.', 1), 'unknown') AS category,
                 brand,
                 CAST(price AS DOUBLE)       AS price,
                 CAST(user_id AS BIGINT)     AS user_id,
                 user_session
          FROM read_csv('{f}', header=true, all_varchar=true)
        ) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)""")
        print(con.execute(f"SELECT count(*), min(event_time), max(event_time) FROM '{out}'").fetchall())


if __name__ == "__main__":
    main()
