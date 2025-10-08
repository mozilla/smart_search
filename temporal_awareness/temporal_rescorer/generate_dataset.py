from datetime import datetime, timedelta
import random
import hashlib
import csv
import os

def generate_url_hash(url):
    hash_object = hashlib.md5(url.encode("utf-8"))
    url_hash = int(hash_object.hexdigest(), 16) % (10 ** 14)
    return url_hash


def generate_history(current_dt, entries_per_day=3, seed=42, savepath=""):
    """
    generate histories for testing, past 10 years, each day has a fixed number of records
    """

    random.seed(seed)

    histories = []

    # generate 3 history for one day for the past over 10 years
    years = 10

    if type(current_dt) == str:
        current_dt = datetime.strptime(current_dt, "%Y-%m-%d")

    c = 0
    for i in range(years*366):
        start = current_dt.replace(hour=0, minute=0, second=0, microsecond=0)-timedelta(days=i)
        for j in range(entries_per_day):
            history_timestamp = start + timedelta(seconds=random.randint(0, 3600*24-1))
            url = f"https://www.firefox.com/{history_timestamp.date()}/{j+1}-{entries_per_day}"
            histories.append({
                "id": c,
                "url": url,
                "title": f"{history_timestamp.date()} - {j + 1}/{entries_per_day}",
                "description": "Get Firefox for Windows, Mac or Linux. Firefox is a free web browser backed by Mozilla, a non-profit dedicated to internet health and privacy.",
                "frecency": 5000,#random.randint(100, 5000),
                "url_hash": generate_url_hash(url),
                "last_visit_date": int(history_timestamp.timestamp()*1000),
                "last_visit_date_str": history_timestamp.isoformat(),
            })
            c += 1

    os.makedirs(savepath, exist_ok=True)
    fieldnames = histories[0].keys()
    with open(f"{savepath}/identical_with_diff_time_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(histories)

    return histories


def generate_golden_from_mapping(mapping, entries_per_day=3, savepath=""):
    rows = []
    for query, w in mapping.items():
        if isinstance(w, tuple):  # (start,end) inclusive
            d = datetime.strptime(w[0], "%Y-%m-%d").date()
            end_d = datetime.strptime(w[1], "%Y-%m-%d").date()
            while d <= end_d:
                base = f"https://www.firefox.com/{d.isoformat()}"
                for i in range(entries_per_day):
                    rows.append({"search_query": "what did I search " + query, "url": f"{base}/{i+1}-{entries_per_day}"})
                d += timedelta(days=1)
        else:  # exact date "YYYY-MM-DD"
            d = datetime.strptime(w, "%Y-%m-%d").date()
            base = f"https://www.firefox.com/{d.isoformat()}"
            for i in range(entries_per_day):
                rows.append({"search_query": "what did I search " + query, "url": f"{base}/{i+1}-{entries_per_day}"})

    rows.sort(key=lambda r: (r["search_query"], r["url"]))
    os.makedirs(savepath, exist_ok=True)
    with open(f"{savepath}/identical_with_diff_time_golden_queries.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["search_query", "url"])
        w.writeheader()
        w.writerows(rows)



if __name__ == "__main__":
    current_dt = "2025-09-29"
    golden_map = {
        # --- single day ---
        "today": "2025-09-29",
        "yesterday": "2025-09-28",
        "yday": "2025-09-28",
        "yestday": "2025-09-28",
        "yesturday": "2025-09-28",
        "yeasterday": "2025-09-28",
        "day before yesterday": "2025-09-27",

        # --- weekdays ---
        "on Monday": "2025-09-22",
        "on Mon": "2025-09-22",
        "on mon": "2025-09-22",
        "monday": "2025-09-22",
        "mon": "2025-09-22",
        "last monday": "2025-09-22",
        "on Sunday": "2025-09-28",
        "on sunday": "2025-09-28",
        "sun": "2025-09-28",

        # --- weekends ---
        "this weekend": ("2025-09-27", "2025-09-28"),
        "last weekend": ("2025-09-27", "2025-09-28"),
        "wknd": ("2025-09-27", "2025-09-28"),

        # --- weeks (Mon–Sun) ---
        "last week": ("2025-09-22", "2025-09-28"),
        "previous week": ("2025-09-22", "2025-09-28"),
        "prev week": ("2025-09-22", "2025-09-28"),
        "last wk": ("2025-09-22", "2025-09-28"),
        "last-week": ("2025-09-22", "2025-09-28"),
        "lastweek": ("2025-09-22", "2025-09-28"),
        "week of 2025-09-22": ("2025-09-22", "2025-09-28"),
        "week of Sep 22, 2025": ("2025-09-22", "2025-09-28"),
        "week of Sept 22, 2025": ("2025-09-22", "2025-09-28"),

        # --- rolling day windows ---
        "past 7 days": ("2025-09-23", "2025-09-29"),
        "last 7 days": ("2025-09-23", "2025-09-29"),
        "last seven days": ("2025-09-23", "2025-09-29"),
        "last7days": ("2025-09-23", "2025-09-29"),
        "last 14 days": ("2025-09-16", "2025-09-29"),
        "past 14 days": ("2025-09-16", "2025-09-29"),
        "last 3 days": ("2025-09-27", "2025-09-29"),
        "last two days": ("2025-09-28", "2025-09-29"),
        "last 30 days": ("2025-08-31", "2025-09-29"),
        "past 30 days": ("2025-08-31", "2025-09-29"),
        "last 30d": ("2025-08-31", "2025-09-29"),
        "past 30d": ("2025-08-31", "2025-09-29"),

        # --- months ---
        "August 2025": ("2025-08-01", "2025-08-31"),
        "Aug 2025": ("2025-08-01", "2025-08-31"),
        "08/2025": ("2025-08-01", "2025-08-31"),
        "June 2024": ("2024-06-01", "2024-06-30"),
        "Jun 2024": ("2024-06-01", "2024-06-30"),
        "06/2024": ("2024-06-01", "2024-06-30"),
        "sept 2024": ("2024-09-01", "2024-09-30"),
        "Sep 2024": ("2024-09-01", "2024-09-30"),
        "September 2024": ("2024-09-01", "2024-09-30"),

        # --- quarters ---
        "Q2 2024": ("2024-04-01", "2024-06-30"),
        "q1 2023": ("2023-01-01", "2023-03-31"),

        # --- seasons ---
        "summer 2024": ("2024-06-01", "2024-08-31"),
        "winter 2023": ("2023-12-01", "2024-02-29"),

        # --- specific dates ---
        "on 2024-02-29": "2024-02-29",
        "Feb 29, 2024": "2024-02-29",
        "29 Feb 2024": "2024-02-29",
        "2/29/2024": "2024-02-29",
        "02/29/2024": "2024-02-29",
        "20240229": "2024-02-29",
        "on 2025/09/28": "2025-09-28",
        "2025.09.28": "2025-09-28",
        "2025 09 28": "2025-09-28",
        "sept 28, 2025": "2025-09-28",
        "sep 28 2025": "2025-09-28",

        # --- ranges ---
        "between 2025-09-01 and 2025-09-15": ("2025-09-01", "2025-09-15"),
        "from 2025-09-01 to 2025-09-15": ("2025-09-01", "2025-09-15"),
        "9/1/2025 - 9/15/2025": ("2025-09-01", "2025-09-15"),
        "Sept 1–15, 2025": ("2025-09-01", "2025-09-15"),
        "Sep 1 to Sep 15, 2025": ("2025-09-01", "2025-09-15"),

        # --- since / until ---
        "since 2025-09-01": ("2025-09-01", "2025-09-29"),
        "since Sep 1, 2025": ("2025-09-01", "2025-09-29"),
        "since 9/1/2025": ("2025-09-01", "2025-09-29"),
        "after 2025-09-01": ("2025-09-02", "2025-09-29"),
        "until 2025-09-15": ("2025-01-01", "2025-09-15"),
        "before 2025-09-01": ("2025-01-01", "2025-08-31"),

        # --- fuzzy/typo month names ---
        "aug 2025": ("2025-08-01", "2025-08-31"),
        "august 2025": ("2025-08-01", "2025-08-31"),
        "august25": ("2025-08-01", "2025-08-31"),
        "aug-2025": ("2025-08-01", "2025-08-31"),

        # --- misc short-hands ---
        "last mo": ("2025-08-01", "2025-08-31"),
        "past mo": ("2025-08-31", "2025-09-29"),  # rolling 30d
        "last yr": ("2024-01-01", "2024-12-31"),
        "2024": ("2024-01-01", "2024-12-31"),

        # --- holiday aliases ---
        "black friday 2024": "2024-11-29",
        "bf 2024": "2024-11-29",
    }

    data = generate_history(current_dt, entries_per_day=3, seed=42, savepath="./data")
    generate_golden_from_mapping(golden_map, entries_per_day=3, savepath="./data")
