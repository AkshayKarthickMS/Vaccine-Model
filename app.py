import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ========== CONFIG ==========
ZERODOSE_PATH = r"C:\Users\AkshayKarthickMS\Desktop\phase-2\zerodose.xlsx"
FACILITY_VISIT_PATH = r"C:\Users\AkshayKarthickMS\Desktop\phase-2\facility_visits.csv"

CHILD_ID_COL = "ID"
VACCINE_COL = "vaccines_administered"

# ========== HELPER FUNCTIONS ==========

def age_to_weeks(y, m, w):
    """Convert years, months, weeks to weeks (approx)."""
    y = y if pd.notnull(y) else 0
    m = m if pd.notnull(m) else 0
    w = w if pd.notnull(w) else 0
    return y * 52 + m * 4 + w


def get_age_weeks(row):
    """
    Prefer current_age_* if present; else fallback to age_*.
    This makes it robust even if some columns are missing/NaN.
    """
    cy = row.get("current_age_years", np.nan)
    cm = row.get("current_age_months", np.nan)
    cw = row.get("current_age_weeks", np.nan)

    ay = row.get("age_years", np.nan)
    am = row.get("age_months", np.nan)
    aw = row.get("age_weeks", np.nan)

    def safe(v):
        return 0 if pd.isna(v) else v

    # Use current age if any of them exists
    if not (pd.isna(cy) and pd.isna(cm) and pd.isna(cw)):
        return age_to_weeks(safe(cy), safe(cm), safe(cw))
    else:
        return age_to_weeks(safe(ay), safe(am), safe(aw))


def parse_vaccine_list(raw):
    """
    Parse vaccine list from:
    - {Penta_3,PCV_3,OPV_3,IPV_2}
    - ["Penta_3","PCV_3",...]
    - "Penta_3,PCV_3,OPV_3"
    """
    if pd.isna(raw):
        return []
    s = str(raw)
    for ch in ['{', '}', '[', ']', '"', "'"]:
        s = s.replace(ch, "")
    tokens = [v.strip() for v in s.split(",") if v.strip()]
    return tokens


def build_vaccine_schedule():
    """
    Minimal vaccine schedule:
    rec = recommended week of life
    max = rough latest acceptable week for simple catch-up
    You can tune max values later with experts.
    """
    data = [
        {"code": "BCG",     "rec": 0,  "max": 52},
        {"code": "OPV_0",   "rec": 0,  "max": 4},

        {"code": "Penta_1", "rec": 6,  "max": 24},
        {"code": "PCV_1",   "rec": 6,  "max": 24},
        {"code": "OPV_1",   "rec": 6,  "max": 24},
        {"code": "Rota_1",  "rec": 6,  "max": 20},
        {"code": "IPV_1",   "rec": 6,  "max": 52},

        {"code": "Penta_2", "rec": 10, "max": 28},
        {"code": "PCV_2",   "rec": 10, "max": 28},
        {"code": "OPV_2",   "rec": 10, "max": 28},

        {"code": "Penta_3", "rec": 14, "max": 32},
        {"code": "PCV_3",   "rec": 14, "max": 32},
        {"code": "OPV_3",   "rec": 14, "max": 32},
        {"code": "IPV_2",   "rec": 14, "max": 52},

        {"code": "MCV_1",   "rec": 39, "max": 156},
        {"code": "MCV_2",   "rec": 65, "max": 208},
    ]
    return pd.DataFrame(data)


def expected_vaccines(age_w, sched):
    """Vaccines that *should* have been given by this age."""
    return sched[sched["rec"] <= age_w]["code"].tolist()


def catchup_vaccines(age_w, missing_set, sched):
    """Which of the missing ones can still be given (age <= max)?"""
    sched = sched.set_index("code")
    give = []
    for v in missing_set:
        if v in sched.index and age_w <= sched.loc[v, "max"]:
            give.append(v)
    return give


def urgency_score(age_w, missing_set, sched):
    """
    Simple urgency metric:
    - more missing vaccines → higher score
    - closer to 'max' week (ageing out) → higher score
    """
    sched = sched.set_index("code")
    score = 0.0
    for v in missing_set:
        if v not in sched.index:
            continue
        rec = sched.loc[v, "rec"]
        max_w = sched.loc[v, "max"]
        window = max(max_w - rec, 1)
        remaining = max(max_w - age_w, 0)
        closeness = 1 - (remaining / window)  # 0 = early, 1 = almost too late
        score += 1 + closeness
    return round(score, 2)


# ========== STREAMLIT APP ==========

st.set_page_config("Zero-Dose Vaccine Resolution Model", layout="wide")
st.title("💉 Zero-Dose Vaccine Resolution Dashboard")
st.caption("Child-level zero-dose catch-up model using zerodose.xlsx; facility_visits.csv used for context.")

# ----- LOAD DATA -----
try:
    zd = pd.read_excel(ZERODOSE_PATH)
    visits = pd.read_csv(FACILITY_VISIT_PATH)
except Exception as e:
    st.error(f"Error loading datasets: {e}")
    st.stop()

# ----- ZERO-DOSE VACCINE LOGIC -----
# NOTE: In zerodose, vaccines_administered often represent "due vaccines / schedule",
# NOT actual received doses. For a true zero-dose child, we treat 'received' as empty.
zd["vaccines_list_zd_raw"] = zd[VACCINE_COL].apply(parse_vaccine_list)

# Compute age in weeks
zd["age_weeks_total"] = zd.apply(get_age_weeks, axis=1)

schedule = build_vaccine_schedule()

records = []
for _, row in zd.iterrows():
    child_id = row.get(CHILD_ID_COL)
    lga = row.get("lga_name", "")
    dist = row.get("Distance to HF", "")
    age_w = row["age_weeks_total"]

    # For zero-dose modeling: assume no schedule vaccines received yet
    received_set = set()
    expected_set = set(expected_vaccines(age_w, schedule))
    missing_set = expected_set  # since received is empty
    catchup_list = catchup_vaccines(age_w, missing_set, schedule)
    urgency = urgency_score(age_w, missing_set, schedule)

    # Zero-dose flag: use dataset's zero_dose column if present
    zd_flag = bool(row.get("zero_dose", True))

    records.append({
        "Child ID": child_id,
        "LGA": lga,
        "Distance to HF": dist,
        "Age (weeks)": round(age_w, 1),
        "Zero-dose (flag)": zd_flag,
        "Missing Vaccine Count": len(missing_set),
        "Missing Vaccines (by age)": ", ".join(sorted(missing_set)),
        "Catch-up Today": ", ".join(sorted(catchup_list)),
        "Urgency Score": urgency,
    })

result_df = pd.DataFrame(records)

# ========== KPI CARDS ==========
col1, col2, col3, col4 = st.columns(4)
col1.metric("Zero-Dose Records (children)", len(result_df))
col2.metric("Median Age (weeks)", float(result_df["Age (weeks)"].median()) if not result_df.empty else 0)
col3.metric("High-Urgency Cases (Score ≥ 6)", int((result_df["Urgency Score"] >= 6).sum()))
col4.metric("Catch-Up Eligible Today", int((result_df["Catch-up Today"] != "").sum()))

# ========== FILTERS ==========
st.sidebar.header("Filters")

urg_min = st.sidebar.slider(
    "Minimum Urgency Score",
    0.0,
    float(max(10.0, result_df["Urgency Score"].max() if not result_df.empty else 10.0)),
    3.0,
)

lga_options = sorted([x for x in result_df["LGA"].dropna().unique().tolist() if x != ""])
selected_lgas = st.sidebar.multiselect("Filter by LGA (optional)", lga_options, default=lga_options)

filtered_df = result_df.copy()
filtered_df = filtered_df[filtered_df["Urgency Score"] >= urg_min]

if selected_lgas:
    filtered_df = filtered_df[filtered_df["LGA"].isin(selected_lgas)]

# ========== VISUAL: URGENCY DISTRIBUTION ==========
st.subheader("📊 Urgency Distribution for Zero-Dose Children")

if not filtered_df.empty:
    fig = px.histogram(
        filtered_df,
        x="Urgency Score",
        nbins=15,
        title="Urgency Score Distribution (Filtered Zero-Dose Children)"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No children match the current filter criteria.")

# ========== PRIORITY TABLE ==========
st.subheader("🚨 Priority Action List for Outreach Teams")

priority_cols = [
    "Child ID",
    "LGA",
    "Distance to HF",
    "Age (weeks)",
    "Missing Vaccine Count",
    "Catch-up Today",
    "Urgency Score",
]

priority_df = filtered_df[priority_cols].sort_values(by="Urgency Score", ascending=False)

st.dataframe(priority_df, use_container_width=True, height=450)

# ========== DOWNLOAD ==========
csv_data = priority_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Priority List (CSV)",
    csv_data,
    "zero_dose_vaccine_actions.csv",
    "text/csv",
)

# ========== CONTEXT: FACILITY VISITS (80k) ==========
st.markdown("---")
st.subheader("🏥 Context: Immunization Visit Volume by LGA (from facility_visits.csv)")

if "track" in visits.columns and "lga_name" in visits.columns:
    imm_visits = visits[visits["track"].str.contains("immunization", na=False)]
    if not imm_visits.empty:
        lga_visits = imm_visits.groupby("lga_name")["id"].count().reset_index()
        lga_visits.columns = ["LGA", "Immunization Visit Count"]

        fig2 = px.bar(
            lga_visits.sort_values(by="Immunization Visit Count", ascending=False),
            x="LGA",
            y="Immunization Visit Count",
            title="Immunization Visits per LGA (Workload & Coverage Context)",
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("This context uses all ~80k visit records, but zero-dose modeling is based on zerodose.xlsx.")
    else:
        st.info("No immunization visits found in facility_visits.csv for context chart.")
else:
    st.info("facility_visits.csv is loaded but missing 'track' or 'lga_name' columns for context chart.")
