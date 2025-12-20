import streamlit as st
import pandas as pd

from reports import (
    generate_trade_by_trade_dashboard_png,
    generate_weekly_dashboard_png,
    generate_monthly_dashboard_png,
    generate_quarterly_dashboard_png,
)

# -----------------------
# Page Config
# -----------------------

st.set_page_config(page_title="VEN Dashboard Generator", layout="centered")

# -----------------------
# Page State
# -----------------------

if "page" not in st.session_state:
    st.session_state.page = "dashboard"

# -----------------------
# Header Styling
# -----------------------

st.markdown(
    """
    <style>
    .ven-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 0.2rem;
        border-bottom: 1px solid #e5e5e5;
        margin-bottom: 1.2rem;
    }
    .ven-title {
        font-size: 1.8rem;
        font-weight: 600;
    }
    .nav-btn button {
        background: none !important;
        border: none !important;
        padding: 0 !important;
        margin-left: 1.2rem !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        color: #444 !important;
        cursor: pointer !important;
    }
    .nav-btn.active button {
        color: #000 !important;
        border-bottom: 2px solid #000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------
# Header Layout
# -----------------------

col_title, col_nav = st.columns([3, 2])

with col_title:
    st.markdown("<div class='ven-title'>VEN Dashboard Generator</div>", unsafe_allow_html=True)

with col_nav:
    nav1, nav2 = st.columns(2)

    with nav1:
        if st.button("Dashboards", key="nav_dash"):
            st.session_state.page = "dashboard"

    with nav2:
        if st.button("How It Works", key="nav_how"):
            st.session_state.page = "howto"

st.markdown("<hr style='margin-top:0.5rem; margin-bottom:1.2rem;'>", unsafe_allow_html=True)

st.caption("Upload your Sierra Chart trades and export performance dashboards as PNGs.")

# -----------------------
# Helpers
# -----------------------

def load_trades(file):
    try:
        return pd.read_csv(file, sep="\t", engine="python")
    except Exception:
        return pd.read_csv(file, engine="python")


def preprocess_for_reports(df, R_value):
    for col in ["Entry DateTime", "Exit DateTime"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(" EP", "", regex=False)
                .str.replace(" BP", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "Entry DateTime" in df.columns:
        df = df[df["Entry DateTime"].notna()].copy()

    if "Exit DateTime" in df.columns and df["Exit DateTime"].notna().any():
        start_date = df["Exit DateTime"].min()
        end_date = df["Exit DateTime"].max()
    else:
        start_date = df["Entry DateTime"].min()
        end_date = df["Entry DateTime"].max()

    pnl_col = "Profit/Loss (C)"
    mfe_col = "FlatToFlat Max Open Profit (C)"
    mae_col = "FlatToFlat Max Open Loss (C)"

    def to_float(series):
        return pd.to_numeric(
            series.astype(str).str.replace(",", "", regex=False).str.strip(),
            errors="coerce"
        ).fillna(0.0)

    df[pnl_col] = to_float(df[pnl_col])
    df[mfe_col] = to_float(df[mfe_col])
    df[mae_col] = to_float(df[mae_col])

    df["Display_PnL_R"] = df[pnl_col] / float(R_value)
    df["Display_MFE_R"] = df[mfe_col] / float(R_value)
    df["Display_MAE_R"] = df[mae_col] / float(R_value)

    df["Display_PnL_USD"] = df[pnl_col]
    df["Display_MFE_USD"] = df[mfe_col]
    df["Display_MAE_USD"] = df[mae_col]

    return df, start_date, end_date


def make_filename(kind, start_date, end_date, R_value):
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    same_day = start_str == end_str

    if kind == "tbt":
        if same_day:
            return f"VEN_TBT_{start_str}_R{int(R_value)}.png"
        else:
            return f"VEN_TBT_{start_str}_to_{end_str}_R{int(R_value)}.png"

    if kind == "weekly":
        return f"VEN_Weekly_{start_str}_to_{end_str}_R{int(R_value)}.png"

    if kind == "monthly":
        month_str = start_date.strftime("%B_%Y")
        return f"VEN_Monthly_{month_str}_R{int(R_value)}.png"

    if kind == "quarterly":
        q = (start_date.month - 1) // 3 + 1
        return f"VEN_Quarterly_Q{q}_{start_date.year}_USD.png"

# ======================================================================
# DASHBOARD PAGE
# ======================================================================

if st.session_state.page == "dashboard":

    col_up, col_r = st.columns(2)

    with col_up:
        st.markdown("### 1️⃣ Upload your trades")
        uploaded_file = st.file_uploader(
            "TradesList.txt from SierraChart\\SavedTradeActivity",
            type=["txt", "csv"]
        )

    with col_r:
        st.markdown("### 2️⃣ Set your R value")
        pad_l, r_mid, pad_r = st.columns([1, 3, 1])
        with r_mid:
            R_value = st.number_input(
                "USD per R",
                min_value=1.0,
                value=500.0,
                step=50.0
            )

    st.caption("R is used for Trade-by-Trade, Weekly and Monthly dashboards. Quarterly is always USD.")

    if uploaded_file:
        try:
            df = load_trades(uploaded_file)
            df, start_date, end_date = preprocess_for_reports(df, R_value)

            st.success(f"{len(df)} trades loaded • {start_date.date()} → {end_date.date()}")

            st.markdown("### 3️⃣ Generate your dashboard")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button("Trade-by-Trade (R)"):
                    df["Display_PnL"] = df["Display_PnL_R"]
                    df["Display_MFE"] = df["Display_MFE_R"]
                    df["Display_MAE"] = df["Display_MAE_R"]
                    png = generate_trade_by_trade_dashboard_png(df, R_value, "R", start_date, end_date)
                    st.download_button(
                        "Download PNG",
                        png,
                        make_filename("tbt", start_date, end_date, R_value),
                        "image/png",
                        type="primary"
                    )
                st.caption("Daily execution. Per-trade P/L with MFE & MAE.")

            with col2:
                if st.button("Weekly (R)"):
                    df["Display_PnL"] = df["Display_PnL_R"]
                    df["Display_MFE"] = df["Display_MFE_R"]
                    df["Display_MAE"] = df["Display_MAE_R"]
                    png = generate_weekly_dashboard_png(df, R_value, "R", start_date, end_date)
                    st.download_button(
                        "Download PNG",
                        png,
                        make_filename("weekly", start_date, end_date, R_value),
                        "image/png",
                        type="primary"
                    )
                st.caption("Weekly review with daily breakdown and stats.")

            with col3:
                if st.button("Monthly (R)"):
                    df["Display_PnL"] = df["Display_PnL_R"]
                    df["Display_MFE"] = df["Display_MFE_R"]
                    df["Display_MAE"] = df["Display_MAE_R"]
                    png = generate_monthly_dashboard_png(df, R_value, "R", start_date, end_date)
                    st.download_button(
                        "Download PNG",
                        png,
                        make_filename("monthly", start_date, end_date, R_value),
                        "image/png",
                        type="primary"
                    )
                st.caption("Full month performance with EV and daily structure.")

            with col4:
                if st.button("Quarterly (USD)"):
                    df["Display_PnL"] = df["Display_PnL_USD"]
                    df["Display_MFE"] = df["Display_MFE_USD"]
                    df["Display_MAE"] = df["Display_MAE_USD"]
                    png = generate_quarterly_dashboard_png(df, R_value, "USD", start_date, end_date)
                    st.download_button(
                        "Download PNG",
                        png,
                        make_filename("quarterly", start_date, end_date, R_value),
                        "image/png",
                        type="primary"
                    )
                st.caption("Big-picture USD view for quarterly performance.")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Upload your TradesList.txt above to begin.")

# ======================================================================
# HOW-TO PAGE
# ======================================================================

if st.session_state.page == "howto":

    st.markdown("## How It Works")

    st.markdown(
        """
        ### 1️⃣ Export from Sierra Chart
        The dashboard date range is defined in Sierra Chart **before you export**:

        - Trade → Trade Activity Log → Trades  
        - Use the **Date Range To Display -  From / To** fields to select the period you want  
        - Then go to File → Save Log As → **TradesList.txt**

         (*Whatever date range you select here is what the VEN dashboards will use.*)

        ### 2️⃣ Upload & Set R
        - Upload the file from **SierraChart\SavedTradeActivity** 
        - Enter your **USD per R** value.

        ### 3️⃣ Generate Dashboards
         Click the dashboard that matches the period you exported:

        - **Trade-by-Trade (R)** – Per-trade P/L with MFE & MAE. For a snapshot of your daily execution.  
        - **Weekly (R)** – For your weekly review with a daily breakdown and stats.   
        - **Monthly (R)** – Full month performance with EV and daily structure.  
        - **Quarterly (USD)** – Big-picture USD view for quarterly performance.  

        Files are never stored.
        """
    )

# -----------------------
# Footer
# -----------------------

st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem; margin-top:1.5rem;'>"
    "No files are stored. All processing happens in memory.<br> Built by "
    "<a href='https://x.com/nevonal' target='_blank' style='color:#666; font-weight:500; text-decoration:none;'>Nev</a>"
    " for Sierra Chart traders."
    "</div>",
    unsafe_allow_html=True
)
