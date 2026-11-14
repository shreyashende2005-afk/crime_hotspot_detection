import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="Crime Hotspot Detection | Los Angeles",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------
# CUSTOM STYLE
# ------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #eef4ff 0%, #f8fbff 48%, #ffffff 100%);
    }
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }
    .hero {
        padding: 1.5rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(115deg, #101d3a 0%, #173d67 58%, #176b87 100%);
        color: white;
        margin-bottom: 1.1rem;
        box-shadow: 0 8px 28px rgba(20, 45, 80, .14);
    }
    .hero h1 { color: white; margin: 0; font-size: 2.15rem; }
    .hero p { color: #dcecff; margin: .45rem 0 0 0; font-size: 1rem; }
    .metric-card {
        background: white;
        border: 1px solid #e3eaf4;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 3px 12px rgba(20, 45, 80, .05);
    }
    .metric-label { color: #64748b; font-size: .85rem; }
    .metric-value { color: #14213d; font-size: 1.65rem; font-weight: 750; }
    .metric-note { color: #64748b; font-size: .75rem; margin-top: .15rem; }
    .section-title { color: #14213d; font-weight: 700; margin: .5rem 0 .7rem 0; }
    .small-note { color: #64748b; font-size: .82rem; }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e3eaf4;
        border-radius: 14px;
        padding: 13px 16px;
        box-shadow: 0 3px 12px rgba(20, 45, 80, .04);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #e8eff9;
        border-radius: 9px 9px 0 0;
        padding: 9px 18px;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #155e75 !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------
DEFAULT_DATA_PATH = "LA_Crime_Data_from_2020_to_2024.csv"
USE_COLUMNS = [
    "DR_NO", "DATE OCC", "TIME OCC", "AREA", "AREA NAME",
    "Part 1-2", "Crm Cd", "Crm Cd Desc", "Vict Age",
    "Weapon Used Cd", "LAT", "LON"
]
FEATURE_COLS = [
    "Unique_Crime_Categories",
    "Weekend_Share",
    "Night_Share",
    "Serious_Crime_Share",
    "Victim_Age_Mean",
    "Weapon_Reported_Share",
    "Mean_Hour"
]
RANDOM_STATE = 42

# ------------------------------------------------------------
# DATA LOADING & PREPARATION
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data_from_path(path, file_mtime):
    data = pd.read_csv(path, usecols=lambda c: c in USE_COLUMNS, low_memory=False)
    return prepare_data(data)

@st.cache_data(show_spinner=False)
def load_data_from_upload(file_bytes):
    from io import BytesIO
    data = pd.read_csv(BytesIO(file_bytes), usecols=lambda c: c in USE_COLUMNS, low_memory=False)
    return prepare_data(data)

def prepare_data(data):
    required = set(USE_COLUMNS)
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("The uploaded CSV is missing required columns: " + ", ".join(sorted(missing)))

    data = data.copy()
    data["DATE OCC"] = pd.to_datetime(data["DATE OCC"], errors="coerce")
    data["TIME OCC"] = pd.to_numeric(data["TIME OCC"], errors="coerce")
    data["LAT"] = pd.to_numeric(data["LAT"], errors="coerce")
    data["LON"] = pd.to_numeric(data["LON"], errors="coerce")

    data = data.dropna(subset=["DATE OCC", "LAT", "LON"]).copy()
    # Keep plausible coordinates in the Los Angeles study region.
    data = data[
        data["LAT"].between(33.0, 35.0) &
        data["LON"].between(-119.5, -117.0)
    ].copy()

    data["AREA NAME"] = data["AREA NAME"].fillna("Unknown").astype(str)
    data["Crm Cd Desc"] = data["Crm Cd Desc"].fillna("Unknown").astype(str)
    data["Vict Age"] = pd.to_numeric(data["Vict Age"], errors="coerce")
    data.loc[~data["Vict Age"].between(0, 100), "Vict Age"] = np.nan

    data["Year"] = data["DATE OCC"].dt.year
    data["Month"] = data["DATE OCC"].dt.month
    data["DayOfWeek"] = data["DATE OCC"].dt.dayofweek
    data["IsWeekend"] = (data["DayOfWeek"] >= 5).astype(int)
    data["Hour"] = (data["TIME OCC"].fillna(0).astype(int) // 100).clip(0, 23)
    data["IsNight"] = (
        data["Hour"].between(20, 23) | data["Hour"].between(0, 5)
    ).astype(int)

    # Rounded coordinates aggregate incidents into approximate geographic cells.
    data["GridLat"] = data["LAT"].round(3)
    data["GridLon"] = data["LON"].round(3)
    return data

@st.cache_data(show_spinner=False)
def make_grid(data):
    grid = data.groupby(["GridLat", "GridLon"]).agg(
        Incident_Count=("DR_NO", "count"),
        Unique_Crime_Categories=("Crm Cd", "nunique"),
        Weekend_Share=("IsWeekend", "mean"),
        Night_Share=("IsNight", "mean"),
        Serious_Crime_Share=("Part 1-2", lambda x: (pd.to_numeric(x, errors="coerce") == 1).mean()),
        Victim_Age_Mean=("Vict Age", "mean"),
        Weapon_Reported_Share=("Weapon Used Cd", lambda x: x.notna().mean()),
        Mean_Hour=("Hour", "mean"),
        Area_Name=("AREA NAME", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    ).reset_index()

    if grid["Victim_Age_Mean"].notna().any():
        grid["Victim_Age_Mean"] = grid["Victim_Age_Mean"].fillna(grid["Victim_Age_Mean"].median())
    else:
        grid["Victim_Age_Mean"] = 0

    threshold = grid["Incident_Count"].quantile(0.90)
    grid["Hotspot"] = (grid["Incident_Count"] >= threshold).astype(int)
    return grid, float(threshold)

@st.cache_resource(show_spinner=False)
def train_hotspot_model(grid):
    X = grid[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = grid["Hotspot"]

    if len(grid) < 10 or y.nunique() < 2 or y.value_counts().min() < 2:
        return None, None, None, None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities) if y_test.nunique() == 2 else np.nan
    }
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0, output_dict=True)
    return model, metrics, cm, report, X, y, probabilities

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>📍 Crime Hotspot Detection</h1>
    <p>Los Angeles · LAPD reported crime data · 2020–2024</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "Explore historical crime patterns, geographic concentrations, and model-based "
    "hotspot indicators through an interactive dashboard."
)
# ------------------------------------------------------------
# DATA SOURCE
# ------------------------------------------------------------
with st.expander("📂 Dataset settings", expanded=not os.path.exists(DEFAULT_DATA_PATH)):
    st.write(
        "Keep the supplied CSV beside this app, or upload it below. "
        "The app processes the data locally in your Python environment."
    )

    uploaded_file = st.file_uploader(
        "Upload LAPD crime CSV",
        type=["csv"]
    )

    st.caption(f"Default file expected: `{DEFAULT_DATA_PATH}`")
try:
    with st.spinner("Loading and preparing LAPD crime data... This may take a little while for the full CSV."):
        if uploaded_file is not None:
            df = load_data_from_upload(uploaded_file.getvalue())
        elif os.path.exists(DEFAULT_DATA_PATH):
            df = load_data_from_path(
                DEFAULT_DATA_PATH,
                os.path.getmtime(DEFAULT_DATA_PATH)
            )
        else:
            st.info("Upload `LA_Crime_Data_from_2020_to_2024.csv` above or place it in the same folder as app.py.")
            st.stop()
except Exception as exc:
    st.error(f"Unable to load the dataset: {exc}")
    st.stop()

if df.empty:
    st.error("No usable records were found after cleaning the dataset.")
    st.stop()

# ------------------------------------------------------------
# FILTERS
# ------------------------------------------------------------
st.markdown("### 🎛️ Explore the data")
filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1.4])

years = sorted(df["Year"].dropna().astype(int).unique().tolist())
areas = ["All Areas"] + sorted(df["AREA NAME"].dropna().unique().tolist())

with filter_col1:
    selected_years = st.multiselect("Year", years, default=years)
with filter_col2:
    selected_area = st.selectbox("LAPD Area", areas)
with filter_col3:
    st.markdown(
        "<div class='small-note'>Filters update the charts, geographic grid, and hotspot model. "
        "Model results describe historical high-volume cells, not future crime events.</div>",
        unsafe_allow_html=True
    )

filtered = df[df["Year"].isin(selected_years)].copy()
if selected_area != "All Areas":
    filtered = filtered[filtered["AREA NAME"] == selected_area]

if filtered.empty:
    st.warning("No records match the selected filters. Please change the year or area.")
    st.stop()

grid, hotspot_threshold = make_grid(filtered)
model, model_metrics, cm, report, X_all, y_all, _ = train_hotspot_model(grid)

# ------------------------------------------------------------
# TOP METRICS
# ------------------------------------------------------------
st.markdown("### 📊 Dashboard Overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Reported incidents", f"{len(filtered):,}")
m2.metric("Geographic grid cells", f"{len(grid):,}")
m3.metric("High-volume cells", f"{int(grid['Hotspot'].sum()):,}")
m4.metric("Hotspot threshold", f"{hotspot_threshold:,.0f} incidents")

st.caption(
    "A high-volume cell is defined as a geographic grid cell at or above the 90th percentile "
    "of incident counts in the currently filtered dataset."
)

# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tab_overview, tab_map, tab_analysis, tab_model, tab_about = st.tabs([
    "🏠 Overview", "🗺️ Hotspot Map", "📈 Crime Analysis", "🤖 ML Model", "ℹ️ About"
])

with tab_overview:
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### Reported incidents by year")
        yearly = filtered.groupby("Year").size().reset_index(name="Incidents")
        fig_year = px.bar(
            yearly, x="Year", y="Incidents", text_auto=".2s",
            color="Incidents", color_continuous_scale="Blues"
        )
        fig_year.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            coloraxis_showscale=False, margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig_year, use_container_width=True)

    with right:
        st.markdown("#### Top crime categories")
        top_crimes = filtered["Crm Cd Desc"].value_counts().head(10).sort_values()
        fig_crime = px.bar(
            x=top_crimes.values, y=top_crimes.index, orientation="h",
            labels={"x": "Incidents", "y": "Crime category"},
            color=top_crimes.values, color_continuous_scale="Teal"
        )
        fig_crime.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            coloraxis_showscale=False, margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig_crime, use_container_width=True)

    left2, right2 = st.columns(2)
    with left2:
        st.markdown("#### Incidents by hour")
        by_hour = filtered.groupby("Hour").size().reindex(range(24), fill_value=0).reset_index()
        by_hour.columns = ["Hour", "Incidents"]
        fig_hour = px.line(by_hour, x="Hour", y="Incidents", markers=True)
        fig_hour.update_traces(line=dict(width=3, color="#167d9a"))
        fig_hour.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(dtick=2), margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig_hour, use_container_width=True)

    with right2:
        st.markdown("#### Incidents by day of week")
        day_labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        by_day = filtered.groupby("DayOfWeek").size().reindex(range(7), fill_value=0).reset_index()
        by_day.columns = ["Day", "Incidents"]
        by_day["Day Name"] = by_day["Day"].map(dict(enumerate(day_labels)))
        fig_day = px.bar(by_day, x="Day Name", y="Incidents", color="Incidents", color_continuous_scale="Purples")
        fig_day.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            coloraxis_showscale=False, margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig_day, use_container_width=True)

    st.markdown("#### LAPD areas with the most reported incidents")
    area_counts = filtered["AREA NAME"].value_counts().head(12).sort_values()
    fig_area = px.bar(
        x=area_counts.values, y=area_counts.index, orientation="h",
        labels={"x": "Reported incidents", "y": "LAPD area"},
        color=area_counts.values, color_continuous_scale="Blues"
    )
    fig_area.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        coloraxis_showscale=False, margin=dict(l=10, r=10, t=20, b=10)
    )
    st.plotly_chart(fig_area, use_container_width=True)

with tab_map:
    st.markdown("#### Aggregated geographic incident concentration")
    st.write(
        "The map uses rounded geographic grid cells rather than individual incident locations. "
        "This reduces location precision and is more appropriate for area-level visualization."
    )

    map_mode = st.radio(
        "Map display",
        ["Incident concentration", "High-volume hotspot cells"],
        horizontal=True
    )
    map_data = grid.copy()
    if map_mode == "High-volume hotspot cells":
        map_data = map_data[map_data["Hotspot"] == 1]

    if map_data.empty:
        st.info("No hotspot cells are available for this selection.")
    else:
        map_data["Radius"] = np.clip(
            100 + map_data["Incident_Count"] * 1.5, 100, 1000
        )
        map_data["Elevation"] = map_data["Incident_Count"] * 2
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_data,
            get_position="[GridLon, GridLat]",
            get_radius="Radius",
            get_fill_color="[220, 65, 65, 150]" if map_mode == "High-volume hotspot cells" else "[25, 120, 170, 125]",
            pickable=True,
            auto_highlight=True
        )
        view = pdk.ViewState(
            latitude=float(map_data["GridLat"].mean()),
            longitude=float(map_data["GridLon"].mean()),
            zoom=9,
            pitch=25
        )
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            tooltip={
                "html": "<b>Area:</b> {Area_Name}<br/>"
                        "<b>Aggregated incidents:</b> {Incident_Count}<br/>"
                        "<b>High-volume cell:</b> {Hotspot}",
                "style": {"backgroundColor": "#172b4d", "color": "white"}
            },
            map_style="light"
        )
        st.pydeck_chart(deck, use_container_width=True)

    st.markdown("#### Highest-volume grid cells")
    display_cols = ["GridLat", "GridLon", "Area_Name", "Incident_Count", "Hotspot"]
    st.dataframe(
        grid.sort_values("Incident_Count", ascending=False)[display_cols].head(25),
        use_container_width=True,
        hide_index=True
    )

with tab_analysis:
    left, right = st.columns(2)
    with left:
        st.markdown("#### Crime categories")
        category_data = filtered["Crm Cd Desc"].value_counts().head(15).reset_index()
        category_data.columns = ["Crime Category", "Incidents"]
        st.dataframe(category_data, use_container_width=True, hide_index=True)

    with right:
        st.markdown("#### Reported incidents by LAPD area")
        area_data = filtered["AREA NAME"].value_counts().reset_index()
        area_data.columns = ["LAPD Area", "Incidents"]
        st.dataframe(area_data, use_container_width=True, hide_index=True)

    st.markdown("#### Weekend vs weekday")
    weekend_data = filtered["IsWeekend"].map({0: "Weekday", 1: "Weekend"}).value_counts().reset_index()
    weekend_data.columns = ["Period", "Incidents"]
    fig_weekend = px.pie(
        weekend_data, names="Period", values="Incidents", hole=0.55,
        color_discrete_sequence=["#1b7894", "#7b6bb1"]
    )
    fig_weekend.update_layout(paper_bgcolor="white", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_weekend, use_container_width=True)

    st.markdown("#### Grid cell incident count distribution")
    fig_hist = px.histogram(
        grid, x="Incident_Count", nbins=35,
        labels={"Incident_Count": "Incidents per grid cell"},
        color_discrete_sequence=["#267c91"]
    )
    fig_hist.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_hist, use_container_width=True)

with tab_model:
    st.markdown("#### Random Forest hotspot classification")
    st.write(
        "The model classifies grid cells as high-volume or regular based on aggregated "
        "crime-pattern features. The target label is created from the 90th percentile of "
        "historical incident counts. This is a retrospective classification exercise, "
        "not a validated forecast of future incidents."
    )

    if model is None:
        st.warning("There are not enough grid cells or both classes to train and evaluate the model for the current filters.")
    else:
        metric_cols = st.columns(5)
        for col, (name, value) in zip(metric_cols, model_metrics.items()):
            col.metric(name, "N/A" if pd.isna(value) else f"{value:.3f}")

        cm_fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Predicted regular", "Predicted hotspot"],
            y=["Actual regular", "Actual hotspot"],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            showscale=False
        ))
        cm_fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted class",
            yaxis_title="Actual class",
            paper_bgcolor="white",
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(cm_fig, use_container_width=True)

        importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
        importance_fig = px.bar(
            x=importance.values, y=importance.index, orientation="h",
            labels={"x": "Feature importance", "y": "Feature"},
            color=importance.values, color_continuous_scale="Teal"
        )
        importance_fig.update_layout(
            title="Feature Importance",
            coloraxis_showscale=False,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        st.plotly_chart(importance_fig, use_container_width=True)

        st.markdown("#### Classification report")
        report_df = pd.DataFrame(report).T
        st.dataframe(report_df.round(3), use_container_width=True)

        # Apply the trained model to all current grid cells for an exploratory score.
        X_current = grid[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
        X_current = X_current.fillna(X_current.median(numeric_only=True))
        grid_scored = grid.copy()
        grid_scored["Predicted_Hotspot"] = model.predict(X_current)
        grid_scored["Hotspot_Probability"] = model.predict_proba(X_current)[:, 1]
        st.markdown("#### Highest model-scored grid cells")
        st.dataframe(
            grid_scored.sort_values("Hotspot_Probability", ascending=False)[
                ["GridLat", "GridLon", "Area_Name", "Incident_Count",
                 "Predicted_Hotspot", "Hotspot_Probability"]
            ].head(20).style.format({"Hotspot_Probability": "{:.3f}"}),
            use_container_width=True
        )

with tab_about:
    st.markdown("#### Project objective")
    st.write(
        "Identify geographic concentrations in historical LAPD crime records and communicate "
        "patterns through interactive charts, aggregated maps, and a machine-learning model."
    )
    st.markdown("#### Methods used")
    st.markdown("""
    - Data cleaning and feature engineering
    - Exploratory data analysis
    - Geographic aggregation using rounded latitude/longitude grid cells
    - 90th-percentile threshold to define high-volume cells
    - Random Forest classification
    - Accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix
    """)
    st.markdown("#### Important limitations")
    st.markdown("""
    - The data represents reported and recorded incidents, not every incident that occurred.
    - The map shows aggregated historical counts and should not be interpreted as a live safety map.
    - A hotspot label describes a geographic cell, not a person or an individual's likelihood of offending.
    - Historical concentrations do not establish causation or guarantee future patterns.
    - Coordinates are rounded before mapping; individual incident locations are not displayed.
    - Model metrics depend on the selected date and area filters and should be interpreted with class imbalance in mind.
    """)
    st.info(
        "Use these outputs for descriptive analysis and responsible community-safety discussion. "
        "Do not use them to profile individuals or make decisions about a person."
    )

# ------------------------------------------------------------
# DOWNLOAD AGGREGATED RESULTS
# ------------------------------------------------------------
st.markdown("---")
st.markdown("### ⬇️ Export aggregated hotspot results")
export_grid = grid.copy()
if model is not None:
    X_export = export_grid[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    X_export = X_export.fillna(X_export.median(numeric_only=True))
    export_grid["Predicted_Hotspot"] = model.predict(X_export)
    export_grid["Hotspot_Probability"] = model.predict_proba(X_export)[:, 1]

csv_bytes = export_grid.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download aggregated hotspot CSV",
    data=csv_bytes,
    file_name="la_crime_hotspot_grid_results.csv",
    mime="text/csv"
)

st.markdown(
    "<p class='small-note'>Crime Hotspot Detection · Los Angeles · "
    "For academic and analytical use. Results are based on the currently selected filters.</p>",
    unsafe_allow_html=True
)
