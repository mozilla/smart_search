import shutil
import sqlite3
import pandas as pd
from pathlib import Path

def export_firefox_history(profile_path, output_file, data_dir="./data", limit=1000, exclude_patterns=None, include_fields=None):
    """
    Export Firefox browsing history from places.sqlite to a csv file.
    """

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    tmp_db = data_dir / "places.sqlite"
    shutil.copy(profile_path, tmp_db)

    conn = sqlite3.connect(tmp_db)

    where_clauses = ["title NOTNULL"]
    if exclude_patterns:
        for pattern in exclude_patterns:
            where_clauses.append(f"url NOT LIKE '{pattern}'")
    where_sql = " AND ".join(where_clauses)

    fields_sql = ", ".join(include_fields) if include_fields else "*"

    query = f"""
    SELECT {fields_sql}
    FROM moz_places
    WHERE {where_sql}
    ORDER BY frecency DESC
    LIMIT {limit};
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    output_path = data_dir / output_file
    df.to_csv(output_path, index=False)

    print(f"Exported {len(df)} rows with fields [{fields_sql}] to {output_path}")


if __name__ == "__main__":
    from_path = fr"C:\Users\Frank\AppData\Roaming\Mozilla\Firefox\Profiles\xwzxj9fa.default-nightly\places.sqlite"
    exclude = ["%google.com/search?%", "%bing.com/search?%"]
    fields = ["url", "title", "description", "preview_image_url", "frecency", "last_visit_date"]

    export_firefox_history(from_path, output_file="history_output_file.csv", data_dir="./data", limit=1000, exclude_patterns=exclude, include_fields=fields)


    # information about the places.sqlite
    # - https://en.wikiversity.org/wiki/Firefox/Browsing_history_database
    # - https://firefox-source-docs.mozilla.org/browser/places/architecture-overview.html
    # - https://firefox-source-docs.mozilla.org/browser/places/index.html
    # question:
    # - where does title and description from? the <title> tag and <description> tag in <header>?