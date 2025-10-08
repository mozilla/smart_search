from datetime import datetime, timedelta
from parser import parse_time_window
from ranking import rank_results, format_window_offset
import random


def generate_history(current_dt, entries_per_day=3, seed=42):
    """
    generate histories for testing, past 10 years, each day has a fixed number of records
    """

    random.seed(seed)

    histories = []

    # generate 3 history for one day for the past over 10 years
    years = 10

    if type(current_dt) == str:
        current_dt = datetime.strptime(current_dt, "%Y-%m-%d")

    for i in range(years*366):
        start = current_dt.replace(hour=0, minute=0, second=0, microsecond=0)-timedelta(days=i)
        for j in range(entries_per_day):
            history_timestamp = start + timedelta(seconds=random.randint(0, 3600*24-1))
            histories.append({
                "title": f"title '{history_timestamp.date()} - {j+1}/{entries_per_day}'",
                "url": "https://www.firefox.com/",
                "frecency": random.randint(100, 5000),
                "last_visit_date_str": history_timestamp.isoformat(),
                "last_visit_date": int(history_timestamp.timestamp()*1000),
            })

    return histories


if __name__ == "__main__":
    current = "2025-09-26" # Friday
    entries_per_day = 5
    histories = generate_history(current, entries_per_day=entries_per_day)

    # generate_history(datetime.today())
    # generate_history("2025-09-01")

    # (query, expected # of histories count)
    queries  = [
        ("what news we have last week?", entries_per_day*7),
        ("three days ago show me my news", entries_per_day),
        ("what news we have yesterday?", entries_per_day),
        ("in 2 days show me items", entries_per_day*2),
        ("between 2025-09-01 and 2025-09-05", entries_per_day*5),
        ("what news we have yestreday?", entries_per_day),
        ("what did I read two weeks ago", entries_per_day), # xx ago, same as since
        ("from last month", entries_per_day*(30+26)),
        ("show me news last Monday", entries_per_day),
        ("show me news from last Monday", entries_per_day*12),
        ("show me news on 2025-08-29", entries_per_day),
        ("what news we have tmorow?", entries_per_day*0),
        ("show me news from last Mon", entries_per_day * 1),
        ("show me news for the sone last Monday", entries_per_day * 1),
        ("what did I search on sunday", entries_per_day * 1),
        # ("what did I search last year", entries_per_day * 366),
    ]

    current_dt = datetime.strptime(current, "%Y-%m-%d")
    for query in queries:
        query, sol_days = query
        window = parse_time_window(query, current_dt)
        print(f"\n=== Query: {query} ===\ncurrent date: {current}")
        if window:
            start, end = window
            print(f"Window: {start.isoformat(sep=' ')} -> {end.isoformat(sep=' ')}")
        else:
            print("Window: <none> (fallback to recency only)")

        rankings = rank_results(
            histories, query, current_dt,
            alpha_temporal=1.0,
            lang="en",
            include_outside=True
        )

        is_in_count = 0
        for row in rankings[:sol_days+5]:
            last_visit_date = datetime.fromtimestamp(row["last_visit_date"] / 1000).isoformat(sep=" ")

            diff_days_text = format_window_offset(row["inside_window"], row["window_diff_days"])
            if diff_days_text == "in":
                is_in_count += 1

            print(f"{row['score']:.3f}  {last_visit_date}  {row['title']}  ({row['url']})  [{diff_days_text}]")

        print(f"Result of (within/should within): {is_in_count}/{sol_days}")