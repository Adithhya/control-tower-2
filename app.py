import io
import re
import csv
from datetime import datetime

import pandas as pd
import requests
import streamlit as st


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Control Tower 2.0",
    page_icon="🛰️",
    layout="wide",
)

# =========================
# Premium Light Theme (Apple-ish) + readable contrast
# =========================
st.markdown(
    """
<style>
/* App background */
.stApp {
  background: radial-gradient(1200px 800px at 15% 10%, #ffffff 0%, #f6f7fb 45%, #f1f3f9 100%);
  color: #0b0d12;
}

/* Force Streamlit default text to dark (prevents white-on-white issues) */
html, body, [class*="css"], .stMarkdown, .stText, .stCaption, .stAlert, .stInfo, .stWarning, .stError, .stSuccess {
  color: #0b0d12 !important;
}

/* Headings */
h1, h2, h3, h4 {
  letter-spacing: -0.02em;
  color: #0b0d12 !important;
}

/* Glassy card */
.ct-card {
  background: rgba(255,255,255,0.78);
  border: 1px solid rgba(20,25,35,0.10);
  box-shadow: 0 12px 30px rgba(20,25,35,0.08);
  border-radius: 18px;
  padding: 18px 18px;
  backdrop-filter: blur(10px);
}
.ct-subtle {
  color: rgba(11,13,18,0.65) !important;
}

/* Metric cards */
[data-testid="stMetric"] {
  background: rgba(255,255,255,0.86);
  border: 1px solid rgba(20,25,35,0.10);
  border-radius: 16px;
  padding: 12px 14px;
  box-shadow: 0 12px 26px rgba(20,25,35,0.06);
}
[data-testid="stMetric"] * {
  color: #0b0d12 !important;
}

/* File uploader dropzone */
section[data-testid="stFileUploaderDropzone"] {
  background: rgba(255,255,255,0.90) !important;
  border: 1px dashed rgba(20,25,35,0.22) !important;
  border-radius: 16px !important;
}
section[data-testid="stFileUploaderDropzone"] * {
  color: rgba(11,13,18,0.85) !important;
}

/* Buttons + Download button */
.stButton>button, [data-testid="stDownloadButton"] button {
  border-radius: 14px;
  border: 1px solid rgba(20,25,35,0.14);
  background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(245,246,250,0.98) 100%);
  color: #0b0d12 !important;
  padding: 10px 14px;
  box-shadow: 0 10px 18px rgba(20,25,35,0.10);
}
.stButton>button:hover, [data-testid="stDownloadButton"] button:hover {
  border: 1px solid rgba(20,25,35,0.26);
  transform: translateY(-1px);
}

/* Labels (sliders, uploader, etc.) */
label, .stSlider label, .stTextInput label, .stSelectbox label, .stFileUploader label {
  color: rgba(11,13,18,0.85) !important;
}

/* Slider text (ticks/values) */
[data-testid="stTickBar"] * , .stSlider * {
  color: rgba(11,13,18,0.75) !important;
}

/* DataFrame container */
[data-testid="stDataFrame"] {
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid rgba(20,25,35,0.10);
  box-shadow: 0 12px 22px rgba(20,25,35,0.06);
}

/* Top-right badge */
.ct-badge {
  position: fixed !important;
  top: max(12px, env(safe-area-inset-top)) !important;
  right: 16px !important;
  z-index: 99999 !important;
  display: block !important;
  background: rgba(255,255,255,0.82);
  border: 1px solid rgba(20,25,35,0.12);
  border-radius: 999px;
  padding: 8px 12px;
  font-size: 12px;
  color: rgba(11,13,18,0.80) !important;
  backdrop-filter: blur(10px);
  box-shadow: 0 12px 22px rgba(20,25,35,0.08);
  pointer-events: none;
}

/* Fallback badge rendered inside Streamlit layout */
.ct-inline-badge-wrap {
  display: flex;
  justify-content: flex-end;
  margin-top: 2px;
  margin-bottom: 2px;
}
.ct-inline-badge {
  display: inline-block;
  background: rgba(255,255,255,0.88);
  border: 1px solid rgba(20,25,35,0.16);
  border-radius: 999px;
  padding: 6px 11px;
  font-size: 12px;
  font-weight: 600;
  color: rgba(11,13,18,0.82) !important;
  box-shadow: 0 8px 16px rgba(20,25,35,0.08);
}
</style>

<div class="ct-badge">Trial by Adi</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="ct-inline-badge-wrap">
  <div class="ct-inline-badge">Trial by Adi</div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================
# Helpers (simple & safe)
# =========================
def now_naive_local() -> datetime:
    """
    Returns current local time as tz-naive datetime.
    Why: avoids pandas "tz-aware vs tz-naive" comparison errors.
    """
    return datetime.now()


def normalize_colname(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[\s\-\/]+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm_map = {normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_colname(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def parse_datetime_series(series: pd.Series) -> pd.Series:
    """
    Robust datetime parse (no crashes). Produces datetime64[ns] with NaT for bad values.
    """
    cleaned = series.astype(str).str.strip()
    dt = pd.to_datetime(cleaned, errors="coerce")
    # ensure tz-naive if tz somehow appears
    try:
        if hasattr(dt.dt, "tz_localize"):
            dt = dt.dt.tz_localize(None)
    except Exception:
        pass
    return dt


# =========================
# Robust file loader (fixes your parser error)
# =========================
def sniff_delimiter(sample_text: str) -> str | None:
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except Exception:
        return None


def load_table(uploaded) -> pd.DataFrame:
    """
    Loads CSV/XLSX reliably:
    - UTF-16/tab exports supported
    - utf-8-sig, cp1252, latin1 fallbacks
    - skips bad lines to prevent tokenizing errors
    """
    name = uploaded.name.lower()

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded)

    raw = uploaded.getvalue()

    # BOM check for UTF-16
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        encodings_to_try = ["utf-16", "utf-16le", "utf-16be"]
    else:
        encodings_to_try = ["utf-8-sig", "utf-8", "cp1252", "latin1"]

    last_err = None
    for enc in encodings_to_try:
        try:
            text = raw.decode(enc, errors="strict")
            sample = text[:5000]
            delim = sniff_delimiter(sample)
            if delim is None:
                delim = "\t" if sample.count("\t") > sample.count(",") else ","

            df = pd.read_csv(
                io.StringIO(text),
                sep=delim,
                engine="python",
                on_bad_lines="skip",
            )

            # If it came as 1 column but contains tabs, split it
            if df.shape[1] == 1:
                col0 = df.columns[0]
                if df[col0].astype(str).str.contains("\t").mean() > 0.5:
                    df2 = df[col0].astype(str).str.split("\t", expand=True)
                    first_row = df2.iloc[0].tolist()
                    if all(isinstance(x, str) and len(x.strip()) > 0 for x in first_row):
                        df2.columns = first_row
                        df2 = df2.iloc[1:].reset_index(drop=True)
                    df = df2

            return df

        except Exception as e:
            last_err = e

    raise RuntimeError(f"Could not read file. Last error: {last_err}")


# =========================
# Province normalization (ON -> Ontario)
# =========================
CA_PROVINCES = {
    "ON": "Ontario",
    "QC": "Quebec",
    "BC": "British Columbia",
    "AB": "Alberta",
    "MB": "Manitoba",
    "SK": "Saskatchewan",
    "NS": "Nova Scotia",
    "NB": "New Brunswick",
    "NL": "Newfoundland and Labrador",
    "PE": "Prince Edward Island",
    "NT": "Northwest Territories",
    "YT": "Yukon",
    "NU": "Nunavut",
}


def normalize_region(code_or_name: str) -> str:
    s = str(code_or_name or "").strip()
    if not s:
        return ""
    up = s.upper()
    if up in CA_PROVINCES:
        return CA_PROVINCES[up]
    return s


def normalize_ca_postal(postal: str) -> str:
    """
    Normalize CA postal:
    - L6Y0B2 -> L6Y 0B2
    - keep only letters+digits
    """
    s = str(postal or "").upper().strip()
    s = re.sub(r"[^A-Z0-9]", "", s)
    if len(s) >= 6:
        s = s[:3] + " " + s[3:6]
    return s


# =========================
# Weather: Geocode + Forecast
# Postal-first geocoding using Nominatim (best for CA)
# Fallback to Open-Meteo
# =========================
@st.cache_data(show_spinner=False, ttl=24 * 3600)
def geocode_latlon(city: str, prov_code_or_name: str, postal: str) -> tuple[float | None, float | None, str]:
    city = str(city or "").strip()
    city_no_hyphen = re.sub(r"\s+", " ", city.replace("-", " ")).strip()
    prov_raw = str(prov_code_or_name or "").strip()
    prov_name = normalize_region(prov_raw)
    postal_norm = normalize_ca_postal(postal)
    postal_compact = re.sub(r"[^A-Z0-9]", "", postal_norm)
    fsa = postal_compact[:3] if len(postal_compact) >= 3 else ""

    # 1) Primary CA fallback by postal FSA (very reliable for your MB/ON/NL style data)
    if re.match(r"^[A-Z]\d[A-Z]$", fsa):
        try:
            r = requests.get(f"https://api.zippopotam.us/CA/{fsa}", timeout=20)
            if r.status_code == 200:
                payload = r.json()
                places = payload.get("places") or []
                if places:
                    pick = places[0]
                    if prov_raw:
                        prov_up = prov_raw.upper().strip()
                        for p in places:
                            p_abbr = str(p.get("state abbreviation", "")).upper().strip()
                            p_state = str(p.get("state", "")).strip().lower()
                            if p_abbr == prov_up or (prov_name and p_state == prov_name.lower()):
                                pick = p
                                break

                    lat = float(pick["latitude"])
                    lon = float(pick["longitude"])
                    place = str(pick.get("place name", "")).strip()
                    abbr = str(pick.get("state abbreviation", "")).strip()
                    return lat, lon, f"zippopotam FSA {fsa} -> {place}, {abbr}"
        except Exception:
            pass

    # 2) Open-Meteo fallback; query broadly and then select Canadian match manually.
    om_attempts = []
    if city:
        om_attempts.append(("open-meteo city", city))
    if city_no_hyphen and city_no_hyphen != city:
        om_attempts.append(("open-meteo city no hyphen", city_no_hyphen))
    if city and prov_name:
        om_attempts.append(("open-meteo city+prov", f"{city}, {prov_name}"))

    for mode, q in om_attempts:
        try:
            url = "https://geocoding-api.open-meteo.com/v1/search"
            params = {"name": q, "count": 20, "language": "en", "format": "json"}
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            j = r.json()
            results = j.get("results") or []
            if not results:
                continue

            ca_hits = [x for x in results if str(x.get("country_code", "")).upper() == "CA"]
            if not ca_hits:
                continue

            pick = ca_hits[0]
            if prov_name:
                filtered = [x for x in ca_hits if prov_name.lower() in str(x.get("admin1", "")).lower()]
                if filtered:
                    pick = filtered[0]

            name = str(pick.get("name", "")).strip()
            admin1 = str(pick.get("admin1", "")).strip()
            return float(pick["latitude"]), float(pick["longitude"]), f"{mode}: {q} -> {name}, {admin1}"
        except Exception:
            continue

    return None, None, f"No geocode match | city='{city}' state='{prov_raw}' postal='{postal_norm}'"


@st.cache_data(show_spinner=False, ttl=30 * 60)
def fetch_hourly_forecast(lat: float, lon: float) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "precipitation,snowfall,wind_gusts_10m",
        "forecast_days": 7,
        "timezone": "auto",
    }
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    j = r.json()

    hourly = j.get("hourly") or {}
    t = hourly.get("time") or []
    if not t:
        return pd.DataFrame()

    return pd.DataFrame({
        "time": pd.to_datetime(hourly["time"], errors="coerce"),
        "precip_mm": hourly.get("precipitation", [0] * len(t)),
        "snow_cm": hourly.get("snowfall", [0] * len(t)),
        "wind_gust_kmh": hourly.get("wind_gusts_10m", [0] * len(t)),
    })


def weather_risk_for_window(
    city: str,
    state: str,
    postal: str,
    window_start: datetime,
    window_end: datetime,
    precip_thr: float,
    snow_thr: float,
    gust_thr: float,
) -> tuple[bool, str]:
    lat, lon, debug = geocode_latlon(city, state, postal)
    if lat is None or lon is None:
        return False, debug

    fc = fetch_hourly_forecast(lat, lon)
    if fc.empty:
        return False, "No forecast"

    ws = pd.to_datetime(window_start, errors="coerce")
    we = pd.to_datetime(window_end, errors="coerce")

    w = fc[(fc["time"] >= ws) & (fc["time"] <= we)].copy()
    if w.empty:
        return False, "No hours in window"

    reasons = []
    if (pd.to_numeric(w["precip_mm"], errors="coerce").fillna(0) >= precip_thr).any():
        reasons.append(f"Precip ≥ {precip_thr} mm/hr")
    if (pd.to_numeric(w["snow_cm"], errors="coerce").fillna(0) >= snow_thr).any():
        reasons.append(f"Snow ≥ {snow_thr} cm/hr")
    if (pd.to_numeric(w["wind_gust_kmh"], errors="coerce").fillna(0) >= gust_thr).any():
        reasons.append(f"Gust ≥ {gust_thr} km/h")

    if reasons:
        return True, "; ".join(reasons)
    return False, f"Below thresholds | {debug}"


# =========================
# Header
# =========================
st.title("🛰️ Control Tower 2.0")
st.markdown(
    """
<div class="ct-card">
  <div style="font-size:16px; font-weight:700;">
    AI-driven Transportation Control Tower 2.0 for AutoRisk, risk retention, transit analytics, weather-based shipment monitoring
  </div>
  <div class="ct-subtle" style="margin-top:8px;">
    Upload Uber TMS extract → compute late risk (NOW → CRDD) → flag destinations impacted by weather → download output.
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")


# =========================
# 1) Upload
# =========================
st.header("1) Upload Uber TMS file (CSV/XLSX)")
uploaded = st.file_uploader("Choose a CSV/XLSX file", type=["csv", "xlsx", "xls"])
if not uploaded:
    st.info("Upload a file to begin.")
    st.stop()

with st.spinner("Reading file…"):
    df = load_table(uploaded)

st.success(f"File loaded successfully ✅  Rows: {len(df):,}  Columns: {df.shape[1]:,}")

with st.expander("Preview first 25 rows", expanded=False):
    st.dataframe(df.head(25), use_container_width=True)


# =========================
# 2) Auto-detect key columns (your exact headers)
# =========================
st.header("2) Auto-detect key columns")

col_city = find_column(df, ["destination city"])
col_state = find_column(df, ["destination state code"])
col_postal = find_column(df, ["destination postal code"])
col_crdd = find_column(df, ["crdd"])
col_actual_del = find_column(
    df,
    [
        "Actual Delivery Date/Time",
        "ACTUAL DELIVERY DATE/TIME",
        "actual delivery date",
        "actual delivery date/time",
        "actual delivery datetime",
        "actual_delivery_date",
        "actual_delivery_date_time",
        "actual_delivery_datetime",
        "actual delivery",
    ],
)
col_meid = find_column(df, ["meid"])
col_delivery = find_column(df, ["delivery number"])
col_lane = find_column(df, ["lane"])
col_cust = find_column(df, ["destination name view"])
col_mode_type = find_column(df, ["mode_type", "mode type"])

required_missing = [n for n, c in [
    ("destination city", col_city),
    ("destination state code", col_state),
    ("destination postal code", col_postal),
    ("CRDD", col_crdd),
] if c is None]

if required_missing:
    st.error("Missing required columns: " + ", ".join(required_missing))
    st.write("Columns found (first 80):")
    st.write(list(df.columns[:80]))
    st.stop()

st.markdown(
    f"""
<div class="ct-card">
  <div style="font-weight:700;">Detected columns</div>
  <div class="ct-subtle" style="margin-top:6px; line-height:1.6;">
    City: <b>{col_city}</b><br/>
    State/Province: <b>{col_state}</b><br/>
    Postal: <b>{col_postal}</b><br/>
    CRDD: <b>{col_crdd}</b><br/>
    Actual Delivery Date: <b>{col_actual_del or "(not found)"}</b><br/>
    MEID: <b>{col_meid or "(not found)"}</b><br/>
    Delivery Number: <b>{col_delivery or "(not found)"}</b><br/>
    Lane: <b>{col_lane or "(not found)"}</b><br/>
    Mode Type: <b>{col_mode_type or "(not found)"}</b><br/>
    Customer Name: <b>{col_cust or "(not found)"}</b><br/>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")


# =========================
# 3) Weather settings
# =========================
st.header("3) Weather Risk Settings")

st.markdown(
    """
<div class="ct-card ct-subtle">
<b>Quick weather training (what delays loads):</b><br/>
• <b>Precipitation (mm/hr)</b> = rain intensity. Light: 0–2, Moderate: 2–7, Heavy: 7+.<br/>
• <b>Snow (cm/hr)</b> = snowfall rate. 0.5+ slows; 1.0+ disruptive; 2.0+ severe.<br/>
• <b>Wind gust (km/h)</b> = sudden peak wind. 50–60 impacts trailers; 70+ unsafe.
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    lookahead_hours = st.slider("Look ahead (hours)", 6, 120, 72)
with c2:
    precip_thr = st.slider("Precip threshold (mm/hr)", 0.0, 20.0, 3.0, 0.5)
with c3:
    snow_thr = st.slider("Snow threshold (cm/hr)", 0.0, 5.0, 1.0, 0.1)
with c4:
    gust_thr = st.slider("Wind gust threshold (km/h)", 20, 120, 70, 5)

max_unique = st.slider("Max unique destinations to check (controls API calls)", 10, 400, 100, 10)
st.caption("Tip: Start with 50–100. Increase only after it works.")
if st.button("Clear cached geocoding/weather and rerun"):
    st.cache_data.clear()
    st.rerun()
st.write("")


# =========================
# 4) Late / CRDD status
# =========================
st.header("4) Late / CRDD Status (NOW vs CRDD)")

work = df.copy()
work["_CRDD_DT"] = parse_datetime_series(work[col_crdd])

if col_actual_del:
    work["_ACTUAL_DEL_DT"] = parse_datetime_series(work[col_actual_del])
else:
    work["_ACTUAL_DEL_DT"] = pd.NaT

now_ts = pd.Timestamp(now_naive_local())

work["Passed_CRDD"] = work["_CRDD_DT"].notna() & (now_ts > work["_CRDD_DT"])
if col_mode_type and col_mode_type in work.columns:
    mode_is_ltl = work[col_mode_type].astype(str).str.strip().str.upper().eq("LTL")
else:
    mode_is_ltl = pd.Series(False, index=work.index)
work["LTL_Weekend_Risk"] = (
    mode_is_ltl
    & work["_CRDD_DT"].notna()
    & work["_CRDD_DT"].dt.dayofweek.isin([5, 6])
)
# Per your rule: compare DATE only (ignore time-of-day).
# Same calendar day as CRDD is on-time; only later day is late.
work["_CRDD_DATE"] = work["_CRDD_DT"].dt.date
work["_ACTUAL_DEL_DATE"] = work["_ACTUAL_DEL_DT"].dt.date
work["_ON_TIME_TO_CRDD"] = (
    work["_CRDD_DATE"].notna()
    & work["_ACTUAL_DEL_DATE"].notna()
    & (work["_ACTUAL_DEL_DATE"] <= work["_CRDD_DATE"])
)
work["Danger_Missed_CRDD"] = (
    (
        # No POD and CRDD already passed.
        work["Passed_CRDD"] & work["_ACTUAL_DEL_DT"].isna()
    )
    |
    (
        # Delivered after CRDD date.
        work["_CRDD_DATE"].notna()
        & work["_ACTUAL_DEL_DATE"].notna()
        & (work["_ACTUAL_DEL_DATE"] > work["_CRDD_DATE"])
    )
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Loads", f"{len(work):,}")
m2.metric("Passed CRDD (NOW > CRDD)", f"{int(work['Passed_CRDD'].sum()):,}")
m3.metric(
    "Danger_missed_CRDD (No POD after CRDD OR delivered after CRDD date)",
    f"{int(work['Danger_Missed_CRDD'].sum()):,}",
)
m4.metric("LTL - weekend delivery risk", f"{int(work['LTL_Weekend_Risk'].sum()):,}")
st.write("")


# =========================
# 5) Weather risk (NOW -> CRDD) using POSTAL-FIRST geocoding
# =========================
st.header("5) Weather Risk (NOW → CRDD)")

dest = work[[col_city, col_state, col_postal, "_CRDD_DT"]].copy()
dest.columns = ["_city", "_state", "_postal", "_crdd"]

# Clean strings safely
dest["_city"] = dest["_city"].fillna("").astype(str).str.strip()
dest["_state"] = dest["_state"].fillna("").astype(str).str.strip()
dest["_postal"] = dest["_postal"].fillna("").apply(normalize_ca_postal)

dest["_dest_key"] = dest["_city"] + " | " + dest["_state"] + " | " + dest["_postal"]
eligible_load_mask = work["_CRDD_DT"].notna() & (work["_CRDD_DT"] > now_ts)
eligible_loads = int(eligible_load_mask.sum())
st.caption(
    f"Weather-eligible loads (CRDD > NOW): {eligible_loads:,} | "
    f"NOW: {now_ts.strftime('%Y-%m-%d %H:%M:%S')}"
)

dest_eligible = dest[dest["_crdd"].notna() & (dest["_crdd"] > now_ts)].copy()
dest_agg = (
    dest_eligible.groupby("_dest_key", as_index=False)
    .agg({
        "_city": "first",
        "_state": "first",
        "_postal": "first",
        "_crdd": "min",
    })
)

weather_rows = []
if dest_agg.empty:
    st.warning("No weather window to evaluate because all CRDD values are missing or already passed.")
else:
    dest_agg = dest_agg.head(int(max_unique))
    st.info(f"Unique destinations to evaluate: {len(dest_agg):,} (limited by slider)")

    with st.spinner("Calling geocoding + weather (cached)..."):
        for _, row in dest_agg.iterrows():
            city = row["_city"]
            state = row["_state"]
            postal = row["_postal"]
            crdd_dt = row["_crdd"]

            window_start = now_ts.to_pydatetime()
            window_end = min(crdd_dt, now_ts + pd.Timedelta(hours=int(lookahead_hours)))

            risk, reason = weather_risk_for_window(
                city=city,
                state=state,
                postal=postal,
                window_start=window_start,
                window_end=window_end,
                precip_thr=float(precip_thr),
                snow_thr=float(snow_thr),
                gust_thr=float(gust_thr),
            )

            weather_rows.append(
                {
                    "_dest_key": row["_dest_key"],
                    "Weather_Risk": bool(risk),
                    "Weather_Reason": reason,
                    "Window_Start": window_start,
                    "Window_End": window_end,
                }
            )

weather_df = pd.DataFrame(
    weather_rows,
    columns=["_dest_key", "Weather_Risk", "Weather_Reason", "Window_Start", "Window_End"],
)

work["_postal_norm"] = work[col_postal].fillna("").apply(normalize_ca_postal)
work["_dest_key"] = (
    work[col_city].fillna("").astype(str).str.strip()
    + " | "
    + work[col_state].fillna("").astype(str).str.strip()
    + " | "
    + work["_postal_norm"].astype(str)
)
out = work.merge(
    weather_df[["_dest_key", "Weather_Risk", "Weather_Reason", "Window_Start", "Window_End"]],
    on="_dest_key",
    how="left",
)

out["Weather_Risk"] = out["Weather_Risk"].fillna(False).astype(bool)
out["Weather_Reason"] = out["Weather_Reason"].fillna("NA (CRDD <= NOW or no weather evaluation)")

risk_count = int(out["Weather_Risk"].eq(True).sum())
st.metric("Loads flagged Weather Risk", f"{risk_count:,}")

with st.expander("Debug: show 15 geocode failures (if any)", expanded=False):
    fails = out[out["Weather_Reason"].fillna("").str.contains("No geocode match", na=False)]
    st.dataframe(fails[[col_city, col_state, col_postal, "Weather_Reason"]].head(15), use_container_width=True)

with st.expander("Debug: weather reason counts", expanded=False):
    st.dataframe(
        out["Weather_Reason"].value_counts(dropna=False).reset_index(name="Count").rename(
            columns={"index": "Weather_Reason"}
        ),
        use_container_width=True,
    )


# =========================
# 6) Results + color labels
# =========================
st.header("6) Results")

def risk_label(row):
    if bool(row.get("Danger_Missed_CRDD", False)):
        return "DANGER: missed CRDD"
    if bool(row.get("LTL_Weekend_Risk", False)):
        return "LTL - weekend delivery"
    if bool(row.get("_ON_TIME_TO_CRDD", False)):
        return "On-time: met CRDD"
    if bool(row.get("Passed_CRDD", False)):
        return "Late: passed CRDD"
    if bool(row.get("Weather_Risk", False)):
        return "Weather Risk"
    return "OK"

out["Risk_Label"] = out.apply(risk_label, axis=1)

display_cols = []
for c in [col_meid, col_delivery, col_lane, col_mode_type, col_cust, col_city, col_state, col_postal, col_crdd, col_actual_del]:
    if c and c in out.columns:
        display_cols.append(c)

display_cols += [
    "Passed_CRDD",
    "Danger_Missed_CRDD",
    "LTL_Weekend_Risk",
    "Weather_Risk",
    "Weather_Reason",
    "Window_Start",
    "Window_End",
    "Risk_Label",
]
display_df = out[display_cols].copy()

def style_rows(df_):
    def row_style(row):
        label = str(row.get("Risk_Label", ""))
        if "DANGER" in label:
            return ["background-color: rgba(255, 59, 48, 0.18)"] * len(row)
        if "LTL - weekend delivery" in label:
            return ["background-color: rgba(175, 82, 222, 0.16)"] * len(row)
        if "Late" in label:
            return ["background-color: rgba(255, 149, 0, 0.16)"] * len(row)
        if "Weather" in label:
            return ["background-color: rgba(10, 132, 255, 0.14)"] * len(row)
        return ["background-color: rgba(52, 199, 89, 0.10)"] * len(row)
    return df_.style.apply(row_style, axis=1)

st.dataframe(style_rows(display_df.head(500)), use_container_width=True)
st.caption("Showing first 500 rows for performance. Download includes all rows.")
st.write("")


# =========================
# 7) Download output (all original + flags)
# =========================
st.header("7) Download updated file")

final_out = out.copy()
csv_bytes = final_out.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Download updated CSV (all original columns + flags)",
    data=csv_bytes,
    file_name="control_tower_2_output.csv",
    mime="text/csv",
)
