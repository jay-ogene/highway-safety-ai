import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dashboard.charts import (
    incidents_by_type_chart,
    severity_bar_chart,
    ttc_histogram,
    incidents_over_time,
    speed_distribution
)

st.set_page_config(
    page_title="Highway Safety AI",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ───────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 2rem; }
[data-testid="stMetricLabel"] { font-size: 0.85rem; }
.alert-critical {
    background:#FEE2E2; border-left:4px solid #E24B4A;
    padding:8px 12px; border-radius:4px; margin:4px 0;
    color: #7B1111;
}
.alert-high {
    background:#FEF3C7; border-left:4px solid #EF9F27;
    padding:8px 12px; border-radius:4px; margin:4px 0;
    color: #633806;
}
.alert-warning {
    background:#EFF6FF; border-left:4px solid #378ADD;
    padding:8px 12px; border-radius:4px; margin:4px 0;
    color: #0C447C;
}
.alert-behavior {
    background:#F0FDF4; border-left:4px solid #639922;
    padding:8px 12px; border-radius:4px; margin:4px 0;
    color: #27500A;
}
</style>
""", unsafe_allow_html=True)

import json
import tempfile

import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from datetime import datetime

# ── GCP Credentials — MUST run before any BigQuery call ──────────
def _setup_credentials():
    """Write Streamlit secrets to a temp key file."""
    try:
        if "gcp_service_account" in st.secrets:
            key = dict(st.secrets["gcp_service_account"])
            if "private_key" in key:
                key["private_key"] = key["private_key"].replace("\\n", "\n")
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            )
            json.dump(key, tmp)
            tmp.flush()
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
            return True
    except Exception:
        pass

    # Fall back to local key file
    local_key = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "secrets", "gcp-key.json"
    )
    if os.path.exists(local_key):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = local_key
        return True

    return False

_creds_ok = _setup_credentials()

# ── BigQuery client ───────────────────────────────────────────────
@st.cache_resource
def get_bq_client():
    if not _creds_ok:
        return None
    try:
        from storage.bigquery_logger import BigQueryLogger
        return BigQueryLogger()
    except Exception as e:
        st.sidebar.error(f"BigQuery: {str(e)[:80]}")
        return None


@st.cache_data(ttl=30)
def load_summary(sequence_filter: str = "") -> dict:
    bq = get_bq_client()
    if bq is None:
        return {}
    seq = f"WHERE sequence_id = '{sequence_filter}'" \
        if sequence_filter else ""
    sql = f"""
        SELECT
            COUNT(*) as total_incidents,
            COUNTIF(event_type='near_miss') as near_miss_count,
            COUNTIF(event_type='behavior') as behavior_count,
            COUNTIF(severity=4) as critical_count,
            COUNTIF(severity=3) as high_count,
            COUNTIF(severity=2) as warning_count,
            COUNTIF(behavior_type='TAILGATING') as tailgating_count,
            COUNTIF(behavior_type='SUDDEN_BRAKING') as braking_count,
            COUNTIF(behavior_type='WRONG_WAY') as wrong_way_count,
            COUNTIF(behavior_type='STOPPED_VEHICLE') as stopped_count,
            ROUND(AVG(IF(ttc_seconds>0,ttc_seconds,NULL)),3) as avg_ttc,
            ROUND(MIN(IF(ttc_seconds>0,ttc_seconds,NULL)),3) as min_ttc,
            COUNT(DISTINCT sequence_id) as sequences_analyzed,
            COUNT(DISTINCT camera_id) as cameras_active
        FROM `{bq.table_ref}`
        {seq}
    """
    rows = list(bq.client.query(sql).result())
    return dict(rows[0]) if rows else {}


@st.cache_data(ttl=30)
def load_incidents(
    limit: int = 200,
    sequence_filter: str = ""
) -> pd.DataFrame:
    bq = get_bq_client()
    if bq is None:
        return pd.DataFrame()
    seq = f"AND sequence_id = '{sequence_filter}'" \
        if sequence_filter else ""
    sql = f"""
        SELECT
            event_type, alert_level, behavior_type,
            severity, ttc_seconds, distance_meters,
            closing_speed_kmh, vehicle_a_id, vehicle_b_id,
            track_id, frame_idx, sequence_id,
            description, timestamp
        FROM `{bq.table_ref}`
        WHERE 1=1 {seq}
        ORDER BY timestamp DESC
        LIMIT {limit}
    """
    rows = list(bq.client.query(sql).result())
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/traffic-jam.png",
             width=60)
    st.title("Highway Safety AI")
    st.caption("YOLOv8s · 0.833 mAP@0.5 · GCP")
    st.divider()

    sequence_filter = st.selectbox(
        "Camera sequence",
        options=["All sequences", "MVI_20011"],
        index=0
    )
    seq = "" if sequence_filter == "All sequences" \
        else sequence_filter

    n_incidents = st.slider(
        "Incidents to load", 50, 500, 200, 50
    )
    auto_refresh = st.toggle("Auto-refresh (30s)", value=False)
    if st.button("Refresh now"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    st.caption("Data: UA-DETRAC highway cameras")
    st.caption("Project: highway-safety-ai-jude")


# ── Auto refresh ──────────────────────────────────────────────────
if auto_refresh:
    import time
    time.sleep(30)
    st.cache_data.clear()
    st.rerun()


# ── Load Data ─────────────────────────────────────────────────────
with st.spinner("Loading from BigQuery..."):
    summary = load_summary(seq)
    df = load_incidents(n_incidents, seq)


# ── Header ────────────────────────────────────────────────────────
st.title("🛣️ Highway Safety Incident Dashboard")
st.caption(
    "Real-time vehicle incident detection — "
    "YOLOv8s · ByteTrack · TTC Analysis · GCP"
)
st.divider()


# ── KPI Row ───────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)

k1.metric(
    "Total Incidents",
    f"{summary.get('total_incidents', 0):,}"
)
k2.metric(
    "🔴 Critical",
    f"{summary.get('critical_count', 0):,}",
    delta="TTC < 1.5s",
    delta_color="inverse"
)
k3.metric(
    "🟠 High Risk",
    f"{summary.get('high_count', 0):,}",
    delta="TTC < 2.5s",
    delta_color="inverse"
)
k4.metric(
    "Avg TTC",
    f"{summary.get('avg_ttc', 0):.2f}s"
)
k5.metric(
    "Min TTC",
    f"{summary.get('min_ttc', 0):.2f}s"
)
k6.metric(
    "Cameras",
    f"{summary.get('cameras_active', 0)}"
)

st.divider()


# ── Charts Row 1 ──────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Incidents by type")
    st.plotly_chart(
        incidents_by_type_chart(summary),
        use_container_width=True
    )

with col2:
    st.subheader("Severity breakdown")
    st.plotly_chart(
        severity_bar_chart(summary),
        use_container_width=True
    )
    st.subheader("Behavior summary")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Tailgating",
              summary.get("tailgating_count", 0))
    b2.metric("Braking",
              summary.get("braking_count", 0) or
              summary.get("sudden_braking_count", 0))
    b3.metric("Wrong way",
              summary.get("wrong_way_count", 0))
    b4.metric("Stopped",
              summary.get("stopped_count", 0) or
              summary.get("stopped_vehicle_count", 0))


# ── Charts Row 2 ──────────────────────────────────────────────────
if not df.empty:
    col3, col4 = st.columns([1, 1])

    with col3:
        st.subheader("TTC distribution")
        st.plotly_chart(
            ttc_histogram(df),
            use_container_width=True
        )

    with col4:
        st.subheader("Incidents over time (by frame)")
        st.plotly_chart(
            incidents_over_time(df),
            use_container_width=True
        )

    st.subheader("Closing speed distribution")
    st.plotly_chart(
        speed_distribution(df),
        use_container_width=True
    )


# ── Live Feed ─────────────────────────────────────────────────────
st.divider()
st.subheader("Live incident feed")
st.caption(f"Showing {min(len(df), 30)} most recent events")

if df.empty:
    st.info("No incidents found. Run the pipeline to generate data.")
else:
    for _, row in df.head(30).iterrows():
        desc = row.get("description", "")
        etype = row.get("event_type", "")
        severity = row.get("severity", 1)
        ts = row.get("timestamp", "")
        if hasattr(ts, "strftime"):
            ts = ts.strftime("%H:%M:%S")

        if severity == 4:
            css = "alert-critical"
        elif severity == 3:
            css = "alert-high"
        elif etype == "behavior":
            css = "alert-behavior"
        else:
            css = "alert-warning"

        st.markdown(
            f'<div class="{css}">'
            f'<strong>{ts}</strong> &nbsp; {desc}'
            f'</div>',
            unsafe_allow_html=True
        )


# ── Raw Data ──────────────────────────────────────────────────────
st.divider()
with st.expander("Raw incident data"):
    if not df.empty:
        st.dataframe(
            df[[
                "timestamp", "event_type", "alert_level",
                "behavior_type", "severity", "ttc_seconds",
                "distance_meters", "description"
            ]].head(100),
            use_container_width=True,
            hide_index=True
        )