"""Builds the analytics database for a fictional test-prep company, BrightPath Learning.

All data is synthetic and generated with a fixed random seed, so every run produces
the same database. Run directly with:  python data_setup.py
"""
import os
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

from config import DB_PATH, DATA_START, DATA_END

SEED = 42

CENTRES = [
    # centre_id, centre_name, city, region, opened_date, capacity, quality factor
    ("BLR-01", "Bengaluru - Jayanagar", "Bengaluru", "South", date(2023, 6, 1), 500, 1.15),
    ("BLR-02", "Bengaluru - Whitefield", "Bengaluru", "South", date(2024, 4, 1), 450, 1.00),
    ("BLR-03", "Bengaluru - Yelahanka", "Bengaluru", "South", date(2025, 6, 1), 400, 0.85),
    ("HYD-01", "Hyderabad - Kukatpally", "Hyderabad", "South", date(2023, 9, 1), 500, 1.10),
    ("HYD-02", "Hyderabad - Madhapur", "Hyderabad", "South", date(2025, 1, 15), 400, 0.95),
    ("CHN-01", "Chennai - Anna Nagar", "Chennai", "South", date(2024, 1, 10), 450, 1.00),
    ("PUN-01", "Pune - Kothrud", "Pune", "West", date(2024, 6, 1), 400, 0.95),
    ("PUN-02", "Pune - Wakad", "Pune", "West", date(2025, 7, 1), 350, 0.80),
    ("DEL-01", "Delhi - Dwarka", "Delhi", "North", date(2024, 3, 1), 500, 1.05),
    ("DEL-02", "Delhi - Rohini", "Delhi", "North", date(2025, 4, 15), 400, 0.90),
    ("KOL-01", "Kolkata - Salt Lake", "Kolkata", "East", date(2024, 8, 1), 400, 0.90),
    ("ONL-01", "Online Live Classes", "Online", "Online", date(2022, 1, 1), 5000, 0.85),
]

COURSES = [
    # course_id, course_name, exam, grade, list fee (INR per year), conversion factor
    ("JEE-11", "JEE 2-Year Programme (Class 11)", "JEE", 11, 125000, 1.00),
    ("JEE-12", "JEE Main + Advanced (Class 12)", "JEE", 12, 110000, 1.05),
    ("NEET-11", "NEET 2-Year Programme (Class 11)", "NEET", 11, 115000, 1.00),
    ("NEET-12", "NEET Full Course (Class 12)", "NEET", 12, 100000, 1.10),
    ("FND-09", "Foundation Class 9", "Foundation", 9, 60000, 0.90),
    ("FND-10", "Foundation Class 10", "Foundation", 10, 65000, 0.95),
]
COURSE_WEIGHTS = [0.22, 0.18, 0.20, 0.15, 0.12, 0.13]
SUBJECTS = {
    "JEE": ["Physics", "Chemistry", "Maths"],
    "NEET": ["Physics", "Chemistry", "Biology"],
    "Foundation": ["Maths", "Science"],
}

# channel: (share of leads, demo attendance probability, conversion probability after attending)
CHANNELS = {
    "Online Ads": (0.34, 0.58, 0.24),
    "Referral": (0.16, 0.86, 0.50),
    "Walk-in": (0.14, 0.90, 0.44),
    "School Event": (0.16, 0.70, 0.30),
    "Organic Search": (0.20, 0.72, 0.36),
}
CITIES = ["Bengaluru", "Hyderabad", "Chennai", "Pune", "Delhi", "Kolkata"]
CITY_WEIGHTS = [0.27, 0.19, 0.12, 0.14, 0.18, 0.10]
MONTH_SEASONALITY = {1: 0.9, 2: 0.9, 3: 1.4, 4: 1.6, 5: 1.6, 6: 1.5, 7: 1.0, 8: 1.0,
                     9: 0.8, 10: 0.8, 11: 0.8, 12: 0.9}
FIRST_NAMES = ["Aarav", "Diya", "Vihaan", "Ananya", "Arjun", "Ishita", "Kabir", "Meera",
               "Rohan", "Saanvi", "Aditya", "Kavya", "Reyansh", "Tara", "Nikhil", "Pooja",
               "Siddharth", "Riya", "Varun", "Sneha"]
LAST_NAMES = ["Sharma", "Reddy", "Iyer", "Patel", "Gupta", "Nair", "Rao", "Das", "Singh",
              "Menon", "Kulkarni", "Banerjee", "Verma", "Pillai", "Joshi"]


def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def _add_months(d: date, n: int) -> date:
    y, m = divmod(d.month - 1 + n, 12)
    return date(d.year + y, m + 1, 1)


def generate_tables() -> dict:
    rng = np.random.default_rng(SEED)
    centre_df = pd.DataFrame(
        [c[:6] for c in CENTRES],
        columns=["centre_id", "centre_name", "city", "region", "opened_date", "capacity"],
    )
    centre_quality = {c[0]: c[6] for c in CENTRES}
    centre_open = {c[0]: c[4] for c in CENTRES}
    course_df = pd.DataFrame(
        [c[:5] for c in COURSES],
        columns=["course_id", "course_name", "exam", "grade", "list_fee_inr"],
    )
    course_info = {c[0]: c for c in COURSES}
    channel_names = list(CHANNELS)
    channel_weights = [CHANNELS[c][0] for c in channel_names]

    leads, demos, students, enrolments, attendance, tests = [], [], [], [], [], []
    lead_n = demo_n = student_n = 0

    day = DATA_START
    while day <= DATA_END:
        n_leads = rng.poisson(10.5 * MONTH_SEASONALITY[day.month])
        for _ in range(n_leads):
            lead_n += 1
            lead_id = f"L{lead_n:06d}"
            channel = rng.choice(channel_names, p=channel_weights)
            city = rng.choice(CITIES, p=CITY_WEIGHTS)
            course_id = rng.choice([c[0] for c in COURSES], p=COURSE_WEIGHTS)
            leads.append((lead_id, day, channel, city, course_id))

            if rng.random() > 0.62:
                continue  # no demo booked
            demo_date = day + timedelta(days=int(rng.integers(0, 11)))
            if demo_date > DATA_END:
                continue
            wants_online = rng.random() < 0.30
            open_here = [c[0] for c in CENTRES if c[2] == city and centre_open[c[0]] <= demo_date]
            centre_id = "ONL-01" if (wants_online or not open_here) else str(rng.choice(open_here))
            _, attend_p, conv_p = CHANNELS[channel]
            attended = rng.random() < min(0.97, attend_p * (0.9 + 0.1 * centre_quality[centre_id]))
            demo_n += 1
            demos.append((f"D{demo_n:06d}", lead_id, centre_id, course_id, demo_date, bool(attended)))

            # Did this lead become a paying student?
            factor = centre_quality[centre_id] * course_info[course_id][5]
            p_convert = conv_p * factor if attended else 0.03
            if rng.random() >= p_convert:
                continue
            if attended and rng.random() < 0.72:
                delay = int(rng.integers(0, 15))      # within the 14-day window
            else:
                delay = int(rng.integers(15, 41))     # converts late
            enrol_date = demo_date + timedelta(days=delay)
            if enrol_date > DATA_END:
                continue

            student_n += 1
            student_id = f"S{student_n:05d}"
            first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
            phone = f"+91 9{rng.integers(100000000, 999999999)}"
            email = f"parent.{first.lower()}.{last.lower()}{student_n}@example.com"
            grade = course_info[course_id][3]
            students.append((student_id, lead_id, f"{first} {last}", phone, email, city, grade, enrol_date))

            list_fee = course_info[course_id][4]
            discount = rng.choice([0.0, 0.05, 0.10, 0.15, 0.20], p=[0.35, 0.25, 0.2, 0.12, 0.08])
            fee_paid = int(round(list_fee * (1 - discount) / 100.0) * 100)
            mode = "online" if centre_id == "ONL-01" else "offline"

            # Outcome: dropped, completed (one-year course) or still active
            dropout_date = None
            status = "active"
            if rng.random() < (0.16 if mode == "online" else 0.11):
                candidate = enrol_date + timedelta(days=int(rng.integers(20, 220)))
                if candidate <= DATA_END:
                    status, dropout_date = "dropped", candidate
            course_end = enrol_date + timedelta(days=365)
            if status == "active" and course_end <= DATA_END:
                status = "completed"
            enrolment_id = f"E{student_n:05d}"
            enrolments.append((enrolment_id, student_id, course_id, centre_id, mode,
                               enrol_date, fee_paid, status, dropout_date))

            # Monthly attendance and tests until the student leaves or data ends
            last_day = min(DATA_END, dropout_date or DATA_END, course_end)
            ability = rng.normal(62, 12)
            base_rate = float(np.clip(rng.beta(8, 2) * centre_quality[centre_id] * 0.95, 0.3, 0.99))
            month = _month_start(enrol_date)
            m_index = 0
            while month <= last_day:
                rate = base_rate
                if dropout_date and (dropout_date - month).days < 75:
                    rate *= 0.6  # attendance falls before a dropout
                scheduled = int(rng.integers(16, 23))
                attended_n = int(rng.binomial(scheduled, min(rate, 0.99)))
                attendance.append((enrolment_id, student_id, month, scheduled, attended_n))
                if m_index >= 1:
                    exam = course_info[course_id][2]
                    for subject in SUBJECTS[exam]:
                        score = ability + 18 * (rate - 0.75) + rng.normal(0, 10)
                        test_day = month + timedelta(days=int(rng.integers(10, 26)))
                        if test_day <= last_day:
                            tests.append((enrolment_id, student_id, test_day, subject,
                                          round(float(np.clip(score, 5, 100)), 1)))
                month = _add_months(month, 1)
                m_index += 1
        day += timedelta(days=1)

    return {
        "centres": centre_df,
        "courses": course_df,
        "leads": pd.DataFrame(leads, columns=["lead_id", "created_date", "channel", "city", "course_id"]),
        "demo_sessions": pd.DataFrame(demos, columns=["demo_id", "lead_id", "centre_id", "course_id",
                                                      "demo_date", "attended"]),
        "students": pd.DataFrame(students, columns=["student_id", "lead_id", "student_name", "phone",
                                                    "parent_email", "city", "grade", "joined_date"]),
        "enrolments": pd.DataFrame(enrolments, columns=["enrolment_id", "student_id", "course_id",
                                                        "centre_id", "mode", "enrol_date", "fee_paid_inr",
                                                        "status", "dropout_date"]),
        "attendance_monthly": pd.DataFrame(attendance, columns=["enrolment_id", "student_id", "month",
                                                                "classes_scheduled", "classes_attended"]),
        "test_scores": pd.DataFrame(tests, columns=["enrolment_id", "student_id", "test_date",
                                                    "subject", "score_pct"]),
    }


DATE_COLUMNS = {"opened_date", "created_date", "demo_date", "joined_date", "enrol_date",
                "dropout_date", "month", "test_date"}


def build_database(path: str = DB_PATH, force: bool = False) -> str:
    """Create the DuckDB file if it does not exist (or if force=True). Returns the path."""
    if os.path.exists(path) and not force:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    tables = generate_tables()
    con = duckdb.connect(path)
    for name, df in tables.items():
        con.register("tmp_df", df)
        casts = ", ".join(
            f"CAST({col} AS DATE) AS {col}" if col in DATE_COLUMNS else col for col in df.columns
        )
        con.execute(f"CREATE TABLE {name} AS SELECT {casts} FROM tmp_df")
        con.unregister("tmp_df")
    con.close()
    return path


if __name__ == "__main__":
    p = build_database(force=True)
    con = duckdb.connect(p, read_only=True)
    for (t,) in con.execute("SELECT table_name FROM information_schema.tables ORDER BY 1").fetchall():
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"{t:20s} {n:>7,} rows")
