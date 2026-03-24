"""
STRIDE — Structural Time-based Risk Intelligence for Damage Evolution
Dashboard: Zone-centric structural health monitoring for bridges & structures.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features.processor import compute_features
import time
from pathlib import Path

# Base directory — resolves correctly regardless of launch directory
BASE_DIR = Path(__file__).parent

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from ingestion.simulated_loader import load_structural_data
from ingestion.iot_loader import load_iot_data, get_serial_reader, is_iot_connected
from features.processor import compute_features
from model.risk_model import RiskModel
from decision.logic import apply_decision_logic
from database.db_manager import (
    init_db, get_risk_history, get_alert_history,
    get_sensor_history, get_db_stats
)
def get_cloud_data():
    url = "https://your-api-url.com/data"   # friend kudukkura link
    res = requests.get(url)
    data = res.json()
    return pd.DataFrame(data)

# Ensure DB is initialised on every startup
init_db()

# ── Serial port detection ────────────────────────────────────────────────────
try:
    import serial.tools.list_ports
    AVAILABLE_PORTS = [p.device for p in serial.tools.list_ports.comports()]
except ImportError:
    AVAILABLE_PORTS = []

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="STRIDE — Structural Risk Intelligence",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
  }

  /* Dark engineering theme */
  .stApp {
    background: #0a0e1a;
    color: #c8d8e8;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #0d1220;
    border-right: 1px solid #1e3a5f;
  }

  /* Header strip */
  .stride-header {
    background: linear-gradient(135deg, #0d1f3c 0%, #0a2a4a 50%, #051525 100%);
    border: 1px solid #1e4d7b;
    border-radius: 8px;
    padding: 24px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
  }
  .stride-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #00d4ff, #0088cc, #00d4ff);
  }
  .stride-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #00d4ff;
    letter-spacing: 4px;
    margin: 0;
    text-shadow: 0 0 20px rgba(0,212,255,0.4);
  }
  .stride-subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #5a8aaa;
    letter-spacing: 2px;
    margin-top: 4px;
  }
  .stride-tagline {
    font-size: 0.95rem;
    color: #7ab0cc;
    margin-top: 8px;
    font-style: italic;
  }

  /* Zone cards */
  .zone-card {
    background: #0d1828;
    border: 1px solid #1a3050;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
  }
  .zone-card:hover { border-color: #00d4ff44; }
  .zone-card.critical { border-left: 4px solid #ff3b3b; }
  .zone-card.repair   { border-left: 4px solid #ff8c00; }
  .zone-card.monitor  { border-left: 4px solid #00cc66; }

  .zone-name {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #c8d8e8;
    letter-spacing: 1px;
  }
  .zone-desc {
    font-size: 0.78rem;
    color: #4a6a80;
    margin-bottom: 8px;
    font-family: 'Share Tech Mono', monospace;
  }

  /* Urgency badges */
  .badge-critical {
    background: #3d0a0a; color: #ff6b6b;
    border: 1px solid #ff3b3b;
    border-radius: 4px; padding: 2px 10px;
    font-size: 0.75rem; font-weight: 700;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 1px;
  }
  .badge-repair {
    background: #2a1a00; color: #ffaa44;
    border: 1px solid #ff8c00;
    border-radius: 4px; padding: 2px 10px;
    font-size: 0.75rem; font-weight: 700;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 1px;
  }
  .badge-monitor {
    background: #002a18; color: #44ffaa;
    border: 1px solid #00cc66;
    border-radius: 4px; padding: 2px 10px;
    font-size: 0.75rem; font-weight: 700;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 1px;
  }

  /* Metric blocks */
  .metric-row {
    display: flex; gap: 12px; margin-top: 8px; flex-wrap: wrap;
  }
  .metric-box {
    background: #080f1a;
    border: 1px solid #1a3050;
    border-radius: 6px;
    padding: 8px 16px;
    min-width: 120px;
  }
  .metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    color: #3a6a8a;
    letter-spacing: 1px;
    text-transform: uppercase;
  }
  .metric-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #00d4ff;
  }
  .metric-value.warn { color: #ff8c00; }
  .metric-value.crit { color: #ff3b3b; }

  /* Section headers */
  .section-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #00d4ff;
    letter-spacing: 3px;
    text-transform: uppercase;
    border-bottom: 1px solid #1a3050;
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
  }

  /* Live indicator */
  .live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #00ff88;
    box-shadow: 0 0 8px #00ff88;
    animation: pulse 1.5s infinite;
    margin-right: 6px;
    vertical-align: middle;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  /* Explanation box */
  .explanation-box {
    background: #080f1a;
    border-left: 3px solid #00d4ff44;
    border-radius: 0 6px 6px 0;
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 0.88rem;
    color: #7ab0cc;
    font-family: 'Exo 2', sans-serif;
  }

  /* Override Streamlit defaults */
  .stSelectbox label, .stSlider label { color: #7ab0cc !important; }
  .stButton > button {
    background: #0d2a4a;
    color: #00d4ff;
    border: 1px solid #00d4ff44;
    border-radius: 6px;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 1px;
  }
  .stButton > button:hover {
    background: #0d3a6a;
    border-color: #00d4ff;
  }
  div[data-testid="metric-container"] {
    background: #0d1828;
    border: 1px solid #1a3050;
    border-radius: 8px;
    padding: 12px;
  }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib theme ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#080f1a',
    'axes.facecolor':    '#080f1a',
    'axes.edgecolor':    '#1a3050',
    'axes.labelcolor':   '#7ab0cc',
    'text.color':        '#c8d8e8',
    'xtick.color':       '#4a6a80',
    'ytick.color':       '#4a6a80',
    'grid.color':        '#1a3050',
    'grid.alpha':        0.6,
    'legend.facecolor':  '#0d1828',
    'legend.edgecolor':  '#1a3050',
    'legend.labelcolor': '#c8d8e8',
    'font.family':       'monospace',
})

ZONE_COLORS = {
    'North-Abutment':  '#ff6b6b',
    'Main-Span-Center':'#ff8c00',
    'East-Pier-Joint': '#00d4ff',
    'West-Pier-Joint': '#7ab0cc',
    'South-Abutment':  '#44ffaa',
}
DEFAULT_COLOR = '#00d4ff'

# ── Helpers ──────────────────────────────────────────────────────────────────
def urgency_badge(category):
    if category == "Immediate Attention":
        return '<span class="badge-critical">⚠ IMMEDIATE ATTENTION</span>'
    if category == "Repair Soon":
        return '<span class="badge-repair">🔧 REPAIR SOON</span>'
    return '<span class="badge-monitor">✓ MONITOR</span>'

def card_class(category):
    if category == "Immediate Attention": return "critical"
    if category == "Repair Soon":         return "repair"
    return "monitor"

def etc_display(val):
    if val == 0.0:      return "CRITICAL", "crit"
    if val == np.inf:   return "STABLE ∞", ""
    if val > 9000:      return "STABLE ∞", ""
    if val < 48:        return f"{val:.1f} hrs", "warn"
    return f"{val:.0f} hrs", ""

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stride-header">
  <div class="stride-title">▣ STRIDE</div>
  <div class="stride-subtitle">STRUCTURAL TIME-BASED RISK INTELLIGENCE FOR DAMAGE EVOLUTION</div>
  <div class="stride-tagline">
    A crack is dangerous not because it exists — but because of how it <em>evolves</em>.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("### ⚙ DATA SOURCE")
data_source = st.sidebar.selectbox(
    "Mode",
    ["Simulated Data", "Live IoT (Cloud)"]
)

if data_source == "IoT Data (Arduino Serial)":
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Serial Connection**")
    port = st.sidebar.selectbox(
        "Port",
        AVAILABLE_PORTS if AVAILABLE_PORTS else ["No Ports Found"]
    )
    baud = st.sidebar.selectbox("Baud Rate", [9600, 115200, 57600])
    zone_label = st.sidebar.text_input("Zone Label", value="Live-Sensor-Node")

    if "iot_connected" not in st.session_state:
        st.session_state.iot_connected = False

    btn_label = "⏹ Disconnect" if st.session_state.iot_connected else "▶ Connect IoT Device"
    if st.sidebar.button(btn_label):
        reader = get_serial_reader(port, baud, zone_id=zone_label)
        if not st.session_state.iot_connected:
            try:
                reader.connect()
                st.session_state.iot_connected = True
                st.sidebar.success("✅ Connected")
            except Exception as e:
                st.sidebar.error(f"❌ {e}")
        else:
            reader.disconnect()
            st.session_state.iot_connected = False
            st.sidebar.info("🔌 Disconnected")

st.sidebar.markdown("---")
st.sidebar.markdown("**About STRIDE**")
st.sidebar.markdown("""
<div style="font-size:0.78rem; color:#4a6a80; font-family: monospace; line-height:1.6">
  Team: AI Admins<br>
  College: KRCE<br>
  Hackathon: CARE HACK'26<br>
  Problem: AI08 — Equipment Failure<br>
  Domain: AI + IoT
</div>
""", unsafe_allow_html=True)

# ── Data pipeline ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=2)
def get_simulated_data():
    results_path = BASE_DIR / "data" / "simulated" / "risk_analysis_results.csv"
    if not results_path.exists():
        return None
    df = pd.read_csv(results_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def get_iot_data():
    if not is_iot_connected():
        return None
    raw = load_iot_data()
    if raw is None or raw.empty:
        return None
    try:
        processed = compute_features(raw, data_source="iot")
        model = RiskModel()
        if not model.load():
            model.train(processed)
            model.save()
        processed = model.predict(processed)
        processed = apply_decision_logic(processed)
        return processed
    except Exception as e:
        st.error(f"⚠ IoT processing error: {e}")
        return None

# ── Load data ─────────────────────────────────────────────────────────────────
if data_source == "Simulated Data":
    df = get_simulated_data()
else:
    try:
        df = get_cloud_data()
    except:
        st.warning("⚠ Cloud data not available, using simulated data")
        df = get_simulated_data()

# ── Main content ──────────────────────────────────────────────────────────────
if df is not None and not df.empty:
    latest = df.groupby('zone_id').tail(1).sort_values('sri', ascending=False).reset_index(drop=True)

    # ── Top KPI row ──────────────────────────────────────────────────────────
    total_zones     = len(latest)
    critical_zones  = (latest['urgency_category'] == 'Immediate Attention').sum()
    repair_zones    = (latest['urgency_category'] == 'Repair Soon').sum()
    avg_sri         = latest['sri'].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Monitored Zones",     total_zones)
    k2.metric("⚠ Immediate Attention", critical_zones)
    k3.metric("🔧 Repair Soon",       repair_zones)
    k4.metric("Avg Structural Risk",  f"{avg_sri:.3f}")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 Live Analysis", "🗂 Historical Records", "⚠ Alert Log"])

    with tab1:
        st.markdown('<div class="section-header">Zone Risk Overview</div>', unsafe_allow_html=True)

        # ── Zone cards ───────────────────────────────────────────────────────
        for _, row in latest.iterrows():
            etc_val, etc_cls = etc_display(row['etc'])
            cc = card_class(row['urgency_category'])
            desc = row.get('zone_description', '')

            st.markdown(f"""
            <div class="zone-card {cc}">
              <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                  <div class="zone-name">📍 {row['zone_id']}</div>
                  <div class="zone-desc">{desc}</div>
                </div>
                {urgency_badge(row['urgency_category'])}
              </div>
              <div class="metric-row">
                <div class="metric-box">
                  <div class="metric-label">SRI</div>
                  <div class="metric-value {'crit' if row['sri']>=0.8 else 'warn' if row['sri']>=0.5 else ''}">{row['sri']:.3f}</div>
                </div>
                <div class="metric-box">
                  <div class="metric-label">DPR / hr</div>
                  <div class="metric-value">{row['dpr']:.5f}</div>
                </div>
                <div class="metric-box">
                  <div class="metric-label">Time-to-Critical</div>
                  <div class="metric-value {etc_cls}">{etc_val}</div>
                </div>
                <div class="metric-box">
                  <div class="metric-label">Deformation</div>
                  <div class="metric-value">{row['deformation_mm']:.1f} mm</div>
                </div>
                <div class="metric-box">
                  <div class="metric-label">Gap</div>
                  <div class="metric-value">{row['gap_mm']:.2f} mm</div>
                </div>
              </div>
              <div class="explanation-box">ℹ {row['urgency_explanation']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── SRI trend chart — all zones ──────────────────────────────────────
        st.markdown('<div class="section-header">SRI Progression — All Zones</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(14, 5))
        for zone_id in df['zone_id'].unique():
            zdf = df[df['zone_id'] == zone_id].sort_values('timestamp')
            color = ZONE_COLORS.get(zone_id, DEFAULT_COLOR)
            ax.plot(zdf['timestamp'], zdf['sri'], label=zone_id, color=color, linewidth=1.8, alpha=0.9)

        ax.axhline(0.8, color='#ff3b3b', linestyle='--', linewidth=1.2, alpha=0.7, label='Critical (0.8)')
        ax.axhline(0.5, color='#ff8c00', linestyle='--', linewidth=1.0, alpha=0.7, label='Elevated (0.5)')
        ax.fill_between(df['timestamp'].unique(), 0.8, 1.0, alpha=0.04, color='#ff3b3b')
        ax.fill_between(df['timestamp'].unique(), 0.5, 0.8, alpha=0.03, color='#ff8c00')
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Time", fontsize=9)
        ax.set_ylabel("Structural Risk Index (SRI)", fontsize=9)
        ax.set_title("Structural Risk Index Over Time — All Monitored Zones", fontsize=10, color='#c8d8e8')
        ax.legend(loc='upper left', fontsize=8, framealpha=0.8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.divider()

        # ── Individual zone deep-dive ────────────────────────────────────────
        st.markdown('<div class="section-header">Zone Deep-Dive Analysis</div>', unsafe_allow_html=True)

        zone_options = sorted(df['zone_id'].unique())
        selected_zone = st.selectbox("Select Zone", zone_options)

        zdf = df[df['zone_id'] == selected_zone].sort_values('timestamp')
        lr  = zdf.iloc[-1]

        c1, c2, c3, c4 = st.columns(4)
        etc_v, _ = etc_display(lr['etc'])
        c1.metric("Current SRI",       f"{lr['sri']:.3f}")
        c2.metric("Damage Rate (DPR)", f"{lr['dpr']:.5f}/hr")
        c3.metric("Time-to-Critical",  etc_v)
        c4.metric("Urgency",           lr['urgency_category'])

        st.markdown(f'<div class="explanation-box">ℹ {lr["urgency_explanation"]}</div>', unsafe_allow_html=True)

        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        zone_color = ZONE_COLORS.get(selected_zone, DEFAULT_COLOR)

        ax1.plot(zdf['timestamp'], zdf['sri'], color=zone_color, linewidth=2, label='SRI')
        ax1.fill_between(zdf['timestamp'], 0, zdf['sri'], alpha=0.12, color=zone_color)
        ax1.axhline(0.8, color='#ff3b3b', linestyle='--', linewidth=1.2, label='Critical (0.8)')
        ax1.axhline(0.5, color='#ff8c00', linestyle='--', linewidth=1.0, label='Elevated (0.5)')
        ax1.set_ylim(0, 1.05)
        ax1.set_ylabel("SRI", fontsize=9)
        ax1.set_title(f"Structural Risk Analysis — {selected_zone}", fontsize=10, color='#c8d8e8')
        ax1.legend(loc='upper left', fontsize=8)

        ax2.plot(zdf['timestamp'], zdf['deformation_mm'], color='#7ab0cc', linewidth=1.8, label='Deformation (mm)')
        ax2b = ax2.twinx()
        ax2b.plot(zdf['timestamp'], zdf['gap_mm'], color='#ff8c00', linewidth=1.4, linestyle=':', label='Gap (mm)')
        ax2b.tick_params(axis='y', colors='#ff8c00')
        ax2b.yaxis.label.set_color('#ff8c00')
        ax2b.set_ylabel("Gap (mm)", fontsize=9, color='#ff8c00')
        ax2b.spines['right'].set_color('#ff8c0044')
        ax2.set_ylabel("Deformation (mm)", fontsize=9)
        ax2.set_xlabel("Time", fontsize=9)
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # ── DPR bar chart ────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Damage Progression Rate — Zone Comparison</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.82rem; color:#4a6a80; font-family:monospace; margin-bottom:12px;">
          <b style="color:#ff6b6b;">Positive DPR</b> → zone is deteriorating &nbsp;|&nbsp;
          <b style="color:#44ffaa;">Negative DPR</b> → zone is currently stabilizing / improving
        </div>
        """, unsafe_allow_html=True)

        fig3, ax3 = plt.subplots(figsize=(10, 4))
        dpr_data = latest.sort_values('dpr', ascending=True)
        bar_colors = ['#44ffaa' if v <= 0 else '#ff6b6b' for v in dpr_data['dpr']]
        bars = ax3.barh(dpr_data['zone_id'], dpr_data['dpr'], color=bar_colors, alpha=0.85, height=0.55)
        ax3.axvline(0,    color='#c8d8e8', linewidth=0.8, alpha=0.5)
        ax3.axvline(0.02, color='#ff8c00', linestyle='--', linewidth=1.0, alpha=0.7, label='Elevated threshold (0.02)')
        ax3.set_xlabel("Damage Progression Rate (DPR/hr)", fontsize=9)
        ax3.set_title("Zone DPR Comparison — Positive = Deteriorating, Negative = Stabilizing", fontsize=10, color='#c8d8e8')
        ax3.legend(fontsize=8)
        for bar, val in zip(bars, dpr_data['dpr']):
            x_pos = bar.get_width() + (0.0001 if val >= 0 else -0.0001)
            ha = 'left' if val >= 0 else 'right'
            ax3.text(x_pos, bar.get_y() + bar.get_height() / 2,
                     f'{val:+.5f}', va='center', ha=ha,
                     fontsize=8, color='#44ffaa' if val <= 0 else '#ff6b6b')
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    # ── Tab 2: Historical Records ─────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">Historical Risk Records</div>', unsafe_allow_html=True)

        # DB stats
        stats = get_db_stats()
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Readings",    stats.get('total_readings', 0))
        s2.metric("Risk Snapshots",    stats.get('total_risk_records', 0))
        s3.metric("Zones Tracked",     stats.get('zones_tracked', 0))
        s4.metric("Critical Events",   stats.get('total_critical_events', 0))

        st.divider()

        # SRI history from DB
        hist_zone = st.selectbox("Select Zone for History", ["All Zones"] + sorted(df['zone_id'].unique().tolist()), key="hist_zone")
        zone_filter = None if hist_zone == "All Zones" else hist_zone
        history_df = get_risk_history(zone_id=zone_filter, limit=1000)

        if not history_df.empty:
            fig4, ax4 = plt.subplots(figsize=(14, 5))
            for zid in history_df['zone_id'].unique():
                zh = history_df[history_df['zone_id'] == zid]
                color = ZONE_COLORS.get(zid, DEFAULT_COLOR)
                ax4.plot(zh['recorded_at'], zh['sri'], label=zid, color=color, linewidth=1.6, alpha=0.9)
            ax4.axhline(0.8, color='#ff3b3b', linestyle='--', linewidth=1.0, alpha=0.6, label='Critical (0.8)')
            ax4.axhline(0.5, color='#ff8c00', linestyle='--', linewidth=0.8, alpha=0.6, label='Elevated (0.5)')
            ax4.set_ylim(0, 1.05)
            ax4.set_xlabel("Recorded At", fontsize=9)
            ax4.set_ylabel("SRI", fontsize=9)
            ax4.set_title("SRI History from Database", fontsize=10, color='#c8d8e8')
            ax4.legend(fontsize=8)
            fig4.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)

            st.markdown('<div class="section-header">Raw Records</div>', unsafe_allow_html=True)
            display_hist = history_df[['recorded_at', 'zone_id', 'sri', 'dpr', 'urgency_category', 'data_source']].copy()
            display_hist['sri'] = display_hist['sri'].round(4)
            display_hist['dpr'] = display_hist['dpr'].round(6)
            st.dataframe(display_hist, use_container_width=True, hide_index=True)
        else:
            st.info("No historical records yet. Run `python main.py` or connect IoT device to populate the database.")

    # ── Tab 3: Alert Log ──────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">Alert Event Log</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.82rem; color:#4a6a80; font-family:monospace; margin-bottom:16px;">
          All events where a zone triggered Immediate Attention or Repair Soon — persisted to database.
        </div>
        """, unsafe_allow_html=True)

        alerts_df = get_alert_history(limit=200)

        if not alerts_df.empty:
            # Summary counts
            a1, a2 = st.columns(2)
            a1.metric("Immediate Attention Events",
                      (alerts_df['urgency_category'] == 'Immediate Attention').sum())
            a2.metric("Repair Soon Events",
                      (alerts_df['urgency_category'] == 'Repair Soon').sum())

            st.divider()
            for _, row in alerts_df.iterrows():
                cc = "critical" if row['urgency_category'] == "Immediate Attention" else "repair"
                badge = urgency_badge(row['urgency_category'])
                st.markdown(f"""
                <div class="zone-card {cc}" style="margin-bottom:8px;">
                  <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                      <span class="zone-name">📍 {row['zone_id']}</span>
                      <span style="font-family:monospace; font-size:0.75rem; color:#4a6a80; margin-left:12px;">
                        {row['recorded_at'].strftime('%Y-%m-%d %H:%M') if hasattr(row['recorded_at'], 'strftime') else row['recorded_at']}
                      </span>
                    </div>
                    {badge}
                  </div>
                  <div class="metric-row" style="margin-top:8px;">
                    <div class="metric-box"><div class="metric-label">SRI</div>
                      <div class="metric-value crit">{row['sri']:.3f}</div></div>
                    <div class="metric-box"><div class="metric-label">DPR</div>
                      <div class="metric-value">{row['dpr']:.5f}</div></div>
                    <div class="metric-box"><div class="metric-label">Source</div>
                      <div class="metric-value" style="font-size:0.9rem;">{row['data_source']}</div></div>
                  </div>
                  <div class="explanation-box" style="margin-top:8px;">ℹ {row['urgency_explanation']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✓ No alert events recorded yet. All zones currently within safe parameters.")

elif data_source == "Simulated Data":
    st.error("⚠ No results found. Run `python main.py` first to generate analysis.")
    st.code("python main.py", language="bash")

else:
    # IoT disconnected state
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown("""
        <div style="text-align:center; padding:40px 0;">
          <div style="font-size:4rem;">📡</div>
          <div style="font-family:'Rajdhani',sans-serif; font-size:1.4rem; color:#3a6a8a; letter-spacing:3px; margin-top:8px;">
            AWAITING CONNECTION
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="explanation-box" style="margin-top:32px;">
          <strong style="color:#00d4ff;">IoT Device Not Connected</strong><br><br>
          STRIDE maintains data integrity — no fake sensor data is injected.<br><br>
          To connect live Arduino data:<br>
          1. Power on the Arduino Uno<br>
          2. Verify flex sensor, ultrasonic, and MPU6050 connections<br>
          3. Select the correct COM port in the sidebar<br>
          4. Click <strong>▶ Connect IoT Device</strong>
        </div>
        """, unsafe_allow_html=True)

# ── Live data auto-refresh ────────────────────────────────────────────────────
if data_source == "IoT Data (Arduino Serial)" and is_iot_connected():
    time.sleep(1)
    st.rerun()
