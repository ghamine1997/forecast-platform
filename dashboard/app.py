"""Store sales forecasting dashboard (Streamlit). Reads only what the pipeline stored."""
import json
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

from src.config import get_engine
from src.metrics import smape

st.set_page_config(page_title="Store Sales Forecast", page_icon="📈", layout="wide")

HORIZONS = {"Next 7 days": 7, "Next 30 days": 30, "Next 90 days": 90, "Next 12 months": 365}
INTERVAL_FILL = "rgba(99, 110, 250, 0.2)"


# ---------------- Data access ----------------

@st.cache_resource
def get_db():
    return get_engine()


@st.cache_data(ttl=600, show_spinner=False)
def query(sql, **params):
    return pd.read_sql(text(sql), get_db(), params=params)


def filters_sql(store, item, extra=None):
    conds, params = [], {}
    if store is not None:
        conds.append("store = :store")
        params["store"] = store
    if item is not None:
        conds.append("item = :item")
        params["item"] = item
    if extra:
        conds.append(extra)
    return ("WHERE " + " AND ".join(conds)) if conds else "", params


def load_history(store, item):
    table = "sales_daily_store" if item is None else "clean_sales"
    where, params = filters_sql(store, item)
    df = query(f"SELECT date, SUM(sales) AS sales FROM {table} {where} "
               f"GROUP BY date ORDER BY date", **params)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_forecast(store, item, horizon):
    if item is None:
        table, trend = "latest_forecast", ", SUM(trend) AS trend"
    else:
        table, trend = "forecasts_item", ""
    where, params = filters_sql(store, item, "horizon_day <= :h")
    params["h"] = horizon
    df = query(f"""SELECT date, SUM(yhat) AS yhat, SUM(yhat_lower) AS yhat_lower,
                          SUM(yhat_upper) AS yhat_upper {trend}
                   FROM {table} {where} GROUP BY date ORDER BY date""", **params)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_model():
    df = query("SELECT * FROM production_model")
    return None if df.empty else df.iloc[0]


# ---------------- Charts ----------------

def style(fig, height):
    fig.update_layout(height=height, hovermode="x unified",
                      margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


def forecast_chart(history, fc, show_trend):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["yhat_upper"], mode="lines",
                             line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["yhat_lower"], mode="lines",
                             line=dict(width=0), fill="tonexty", fillcolor=INTERVAL_FILL,
                             name="80% interval"))
    fig.add_trace(go.Scatter(x=history["date"], y=history["sales"], mode="lines",
                             name="Actual sales"))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["yhat"], mode="lines", name="Forecast"))
    if show_trend and "trend" in fc.columns:
        fig.add_trace(go.Scatter(x=fc["date"], y=fc["trend"], mode="lines",
                                 name="Trend", line=dict(dash="dash")))
    return style(fig, 450)


def monthly_chart(history, fc):
    hist_m = history.set_index("date")["sales"].resample("MS").sum()
    hist_m = hist_m[hist_m.index >= hist_m.index.max() - pd.DateOffset(months=23)]
    fc_m = fc.set_index("date")["yhat"].resample("MS").sum()
    fig = go.Figure([go.Bar(x=hist_m.index, y=hist_m.values, name="Actual"),
                     go.Bar(x=fc_m.index, y=fc_m.values, name="Forecast")])
    return style(fig, 350)


# ---------------- Sidebar filters ----------------

model = load_model()
if model is None:
    st.error("No production model yet. Run the Airflow pipeline first.")
    st.stop()

stores = query("SELECT DISTINCT store FROM sales_daily_store ORDER BY store")["store"].astype(int).tolist()
items = query("SELECT DISTINCT item FROM clean_sales ORDER BY item")["item"].astype(int).tolist()

with st.sidebar:
    st.header("Filters")
    store_choice = st.selectbox("Store", ["All stores"] + stores,
                                format_func=lambda s: s if isinstance(s, str) else f"Store {s}")
    item_choice = st.selectbox("Product", ["All products"] + items,
                               format_func=lambda i: i if isinstance(i, str) else f"Product {i}")
    horizon_label = st.radio("Forecast horizon", list(HORIZONS), index=1)

store = None if isinstance(store_choice, str) else store_choice
item = None if isinstance(item_choice, str) else item_choice
horizon = HORIZONS[horizon_label]

history = load_history(store, item)
first, last = history["date"].min().date(), history["date"].max().date()
default_period = (last - timedelta(days=180), last)

with st.sidebar:
    period = st.date_input("History shown", value=default_period,
                           min_value=first, max_value=last)
    show_trend = st.checkbox("Show trend", value=False, disabled=item is not None)
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

start, end = period if isinstance(period, tuple) and len(period) == 2 else default_period
shown = history[(history["date"].dt.date >= start) & (history["date"].dt.date <= end)]
fc = load_forecast(store, item, horizon)

# ---------------- Header and KPIs ----------------

store_txt = "All stores" if store is None else f"Store {store}"
item_txt = "All products" if item is None else f"Product {item}"
model_txt = f"{model['model_name']} v{int(model['version'])}"

st.title("📈 Store Sales Forecast")
st.caption(f"{store_txt} · {item_txt} · Model: {model_txt} · Data until {last}")

total_fc = fc["yhat"].sum()
last_year = history.set_index("date")["sales"].reindex(fc["date"] - pd.Timedelta(days=364)).sum()
last_12m = history.loc[history["date"] > history["date"].max() - pd.Timedelta(days=365), "sales"].sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Forecast · {horizon_label}", f"{total_fc:,.0f}",
          f"{(total_fc / last_year - 1) * 100:+.1f}% vs last year" if last_year else None)
c2.metric("Same period last year", f"{last_year:,.0f}",
          help="Same dates one year earlier (364 days, so weekdays match)")
c3.metric("Sales, last 12 months", f"{last_12m:,.0f}")
c4.metric("Forecast accuracy", f"{100 - model['test_smape']:.1f}%",
          help="100 minus the test SMAPE, measured on Oct–Dec 2017, data the model had never seen")

tab_fc, tab_acc, tab_model = st.tabs(["📈 Forecast", "🎯 Accuracy", "🧠 Model"])

# ---------------- Tab 1: Forecast ----------------

with tab_fc:
    st.plotly_chart(forecast_chart(shown, fc, show_trend))
    if store is None or item is not None:
        st.caption("When several stores or products are combined, the interval is the sum of "
                   "individual intervals, so it is slightly wider (more cautious) than a true combined interval.")

    st.subheader("Monthly sales evolution")
    st.plotly_chart(monthly_chart(history, fc))

    with st.expander("Forecast table"):
        table = fc.copy()
        table["date"] = table["date"].dt.date
        table = table.round(0)
        st.dataframe(table, hide_index=True)
        st.download_button("Download CSV", table.to_csv(index=False).encode("utf-8"),
                           file_name=f"forecast_{store_txt}_{item_txt}_{horizon}d.csv".replace(" ", "_"),
                           mime="text/csv")

# ---------------- Tab 2: Accuracy ----------------

with tab_acc:
    run_id, name = model["run_id"], model["model_name"]
    st.caption("Predictions made for Oct–Dec 2017 by the candidate model, before it had seen those days.")
    if item is not None:
        st.info("Accuracy is measured at store level. Product forecasts are a share of the store forecast.")

    test = query("""SELECT date, store, AVG(sales) AS sales, AVG(yhat) AS yhat
                    FROM test_predictions WHERE run_id = :r AND model_name = :m
                    GROUP BY date, store""", r=run_id, m=name)
    test["date"] = pd.to_datetime(test["date"])
    selected = test if store is None else test[test["store"] == store]
    daily = selected.groupby("date")[["sales", "yhat"]].sum().reset_index()

    fig = go.Figure([go.Scatter(x=daily["date"], y=daily["sales"], mode="lines", name="Actual"),
                     go.Scatter(x=daily["date"], y=daily["yhat"], mode="lines", name="Predicted")])
    st.plotly_chart(style(fig, 380))

    left, right = st.columns(2)
    with left:
        st.subheader("Error by store (SMAPE %)")
        per_store = (test.groupby("store")
                     .apply(lambda g: smape(g["sales"], g["yhat"]), include_groups=False)
                     .reset_index(name="smape"))
        bar = go.Figure(go.Bar(x=per_store["store"].astype(str), y=per_store["smape"]))
        st.plotly_chart(style(bar, 320))
    with right:
        st.subheader("Model comparison (test set)")
        comparison = query("""SELECT DISTINCT ON (model_name) model_name, smape, rmse, mae,
                                     mape, r2, bias_pct, coverage_80
                              FROM evaluation_results WHERE run_id = :r
                              ORDER BY model_name, evaluated_at DESC""", r=run_id)
        st.dataframe(comparison.round(3), hide_index=True)

        st.subheader("Cross-validation folds")
        folds = query("""SELECT DISTINCT ON (model_name, fold) model_name, fold,
                                val_start::date AS val_start, val_end::date AS val_end, smape, rmse
                         FROM cv_fold_metrics WHERE run_id = :r
                         ORDER BY model_name, fold""", r=run_id)
        st.dataframe(folds.round(3), hide_index=True)

# ---------------- Tab 3: Model ----------------

with tab_model:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Version", f"v{int(model['version'])}")
    m2.metric("Model", model["model_name"])
    m3.metric("Registered", str(pd.Timestamp(model["registered_at"]).date()))
    m4.metric("Dataset version", model["dataset_version"])

    k = st.columns(6)
    k[0].metric("CV SMAPE", f"{model['cv_smape']:.2f}%")
    k[1].metric("Test SMAPE", f"{model['test_smape']:.2f}%")
    k[2].metric("Test RMSE", f"{model['test_rmse']:.1f}")
    k[3].metric("Test MAE", f"{model['test_mae']:.1f}")
    k[4].metric("Test R²", f"{model['test_r2']:.3f}")
    k[5].metric("Interval coverage", f"{model['test_coverage_80']:.1f}%",
                help="Share of actual values inside the 80% interval (target: about 80%)")

    st.write(f"**Trained on:** {pd.Timestamp(model['train_start']).date()} → "
             f"{pd.Timestamp(model['train_end']).date()} · **Pipeline run:** {model['run_id']}")

    st.subheader("Hyperparameters (chosen by Optuna)")
    st.json(json.loads(model["params"]))

    st.subheader("Registry history")
    st.dataframe(query("""SELECT version, model_name, stage, registered_at, dataset_version,
                                 ROUND(cv_smape::numeric, 2) AS cv_smape,
                                 ROUND(test_smape::numeric, 2) AS test_smape
                          FROM model_registry ORDER BY version DESC"""), hide_index=True)

    st.subheader("Latest data quality checks")
    st.dataframe(query("""SELECT check_name, passed, detail, run_at FROM data_quality_log
                          WHERE run_at = (SELECT MAX(run_at) FROM data_quality_log)"""),
                 hide_index=True)

    st.subheader("Latest hyperparameter tuning")
    st.dataframe(query("""SELECT DISTINCT ON (model_name) model_name,
                                 ROUND(best_smape::numeric, 3) AS best_smape,
                                 n_trials, duration_s, run_at
                          FROM hyperparameter_runs ORDER BY model_name, run_at DESC"""),
                 hide_index=True)