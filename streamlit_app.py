"""
E-commerce Retention ML Pipeline — Interactive Streamlit Dashboard
5 tabs: Overview · Model Results · Analytics · Pipeline & API · Predict
"""
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Ecommerce Retention ML",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BUNDLE_PATH = Path("models/demo_bundle.pkl")

@st.cache_resource
def load_bundle():
    if not BUNDLE_PATH.exists():
        st.error("Demo bundle not found. Run `python train_demo.py` first.")
        st.stop()
    return joblib.load(BUNDLE_PATH)

bundle = load_bundle()
ds = bundle["dataset_stats"]
metrics = bundle["metrics"]
feature_cols = bundle["feature_cols"]
feat_stats = bundle["feature_stats"]

# ── Tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🛒 Overview", "📊 Model Results", "📈 Analytics",
    "⚙️ Pipeline & API", "🔮 Predict"
])

# ═══════════════════════════════════════════════════════
# TAB 1 — Overview
# ═══════════════════════════════════════════════════════
with tab1:
    st.title("🛒 Ecommerce Customer Retention ML Pipeline")
    st.markdown(
        "End-to-end ML pipeline for predicting which customers will **repeat-purchase** "
        "and estimating their **expected spend**. Built on **3,500 real transactions** "
        f"({ds['date_min']} to {ds['date_max']})."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Transactions", f"{ds['n_transactions']:,}")
    c2.metric("Total Revenue", f"${ds['total_revenue']:,.0f}")
    c3.metric("Avg Order Value", f"${ds['avg_order_value']:,.0f}")
    c4.metric("Avg Profit", f"${ds['avg_profit']:,.2f}")
    c5.metric("Profit Margin", f"{ds['overall_profit_margin']:.1f}%")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Revenue by Category")
        cat_df = pd.DataFrame(bundle["cat_stats"])
        fig = go.Figure(go.Bar(
            x=cat_df["category"],
            y=cat_df["total_sales"],
            marker_color=["#4FC3F7", "#FFD54F", "#A5D6A7"],
            text=[f"${v:,.0f}" for v in cat_df["total_sales"]],
            textposition="outside",
        ))
        fig.update_layout(
            yaxis_title="Total Sales ($)",
            plot_bgcolor="#1a1a2e", paper_bgcolor="#0e1117", font_color="white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Revenue by Region")
        reg_df = pd.DataFrame(bundle["region_stats"])
        colors_map = {"North": "#EF9A9A", "South": "#CE93D8",
                      "East": "#80DEEA", "West": "#FFCC80"}
        fig2 = go.Figure(go.Bar(
            x=reg_df["region"],
            y=reg_df["total_sales"],
            marker_color=[colors_map.get(r, "#ccc") for r in reg_df["region"]],
            text=[f"${v:,.0f}" for v in reg_df["total_sales"]],
            textposition="outside",
        ))
        fig2.update_layout(
            yaxis_title="Total Sales ($)",
            plot_bgcolor="#1a1a2e", paper_bgcolor="#0e1117", font_color="white"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Monthly Revenue Trend")
    monthly_df = pd.DataFrame(bundle["monthly"]).sort_values("year_month")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=monthly_df["year_month"], y=monthly_df["sales"],
        mode="lines+markers",
        line=dict(color="#4FC3F7", width=2),
        fill="tozeroy", fillcolor="rgba(79,195,247,0.15)",
        marker=dict(size=4),
        name="Monthly Revenue"
    ))
    fig3.update_layout(
        xaxis_title="Month", yaxis_title="Sales ($)",
        plot_bgcolor="#1a1a2e", paper_bgcolor="#0e1117",
        font_color="white",
        xaxis=dict(tickangle=-45, nticks=18),
    )
    st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 2 — Model Results
# ═══════════════════════════════════════════════════════
with tab2:
    st.title("📊 Model Evaluation Results")

    m = metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CV ROC-AUC", f"{m['roc_auc_cv']:.4f}")
    c2.metric("CV PR-AUC", f"{m['pr_auc_cv']:.4f}")
    c3.metric("Test ROC-AUC", f"{m['roc_auc_test']:.4f}")
    c4.metric("RMSE Expected Spend", f"${m['rmse_expected_spend']:,.0f}")

    c5, c6 = st.columns(2)
    c5.metric("Training Customers", f"{m['n_customers']}")
    c6.metric("Positive Rate", f"{m['positive_rate']:.1%}")

    st.info(
        "**Note:** ROC-AUC=1.0 reflects clean separability in this synthetic demo setup "
        "(monthly product×region cohorts; very low positive rate ~3.7%). "
        "In production on real customer-level data, typical values are 0.70–0.85."
    )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Classifier Feature Importances")
        clf_imp = bundle["clf_importance"]
        imp_df = pd.DataFrame.from_dict(clf_imp, orient="index", columns=["importance"])
        imp_df = imp_df.sort_values("importance", ascending=True)
        fig = go.Figure(go.Bar(
            x=imp_df["importance"], y=imp_df.index,
            orientation="h",
            marker_color="#4FC3F7",
        ))
        fig.update_layout(
            xaxis_title="Importance",
            plot_bgcolor="#1a1a2e", paper_bgcolor="#0e1117", font_color="white",
            height=300, margin=dict(l=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Actual vs Expected Spend (Test Set)")
        sc = bundle["scatter_data"]
        fig2 = go.Figure()
        mn, mx = min(sc["actual"] + sc["predicted"]), max(sc["actual"] + sc["predicted"])
        fig2.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx],
                                  mode="lines", line=dict(dash="dash", color="gray"),
                                  name="Perfect fit"))
        fig2.add_trace(go.Scatter(
            x=sc["actual"], y=sc["predicted"],
            mode="markers",
            marker=dict(
                color=sc["prob"], colorscale="Blues", size=10,
                colorbar=dict(title="P(repeat)"), line=dict(color="white", width=1)
            ),
            name="Test obs.",
        ))
        fig2.update_layout(
            xaxis_title="Actual Future Spend ($)", yaxis_title="Predicted Expected Spend ($)",
            plot_bgcolor="#1a1a2e", paper_bgcolor="#0e1117", font_color="white",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Decile Lift Table")
    lift_df = pd.DataFrame(bundle["lift_data"])
    if not lift_df.empty:
        st.dataframe(
            lift_df.rename(columns={
                "decile": "Decile", "avg_prob": "Avg P(repeat)",
                "conversion_rate": "Actual Conversion Rate"
            }).style.format({
                "Avg P(repeat)": "{:.3f}",
                "Actual Conversion Rate": "{:.3f}",
            }),
            use_container_width=True, hide_index=True
        )

# ═══════════════════════════════════════════════════════
# TAB 3 — Analytics
# ═══════════════════════════════════════════════════════
with tab3:
    st.title("📈 Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Products by Revenue")
        prod_df = pd.DataFrame(bundle["product_stats"]).head(10)
        fig = go.Figure(go.Bar(
            x=prod_df["total_sales"],
            y=prod_df["product_name"],
            orientation="h",
            marker_color="#FFD54F",
            text=[f"${v:,.0f}" for v in prod_df["total_sales"]],
            textposition="outside",
        ))
        fig.update_layout(
            xaxis_title="Total Sales ($)", yaxis=dict(autorange="reversed"),
            plot_bgcolor="#1a1a2e", paper_bgcolor="#0e1117", font_color="white", height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Profit Margin by Category")
        pm_df = pd.DataFrame(bundle["profit_margin"])
        fig2 = go.Figure(go.Bar(
            x=pm_df["category"], y=pm_df["avg_margin_pct"],
            marker_color=["#A5D6A7", "#4FC3F7", "#EF9A9A"],
            text=[f"{v:.1f}%" for v in pm_df["avg_margin_pct"]],
            textposition="outside",
        ))
        fig2.update_layout(
            yaxis_title="Avg Profit Margin (%)",
            plot_bgcolor="#1a1a2e", paper_bgcolor="#0e1117", font_color="white"
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Revenue Split by Category (Pie)")
        cat_df = pd.DataFrame(bundle["cat_stats"])
        fig3 = go.Figure(go.Pie(
            labels=cat_df["category"],
            values=cat_df["total_sales"],
            hole=0.45,
            marker=dict(colors=["#4FC3F7", "#FFD54F", "#A5D6A7"],
                        line=dict(color="#0e1117", width=2)),
        ))
        fig3.update_layout(
            paper_bgcolor="#0e1117", font_color="white",
            legend=dict(bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Orders by Region (Pie)")
        reg_df = pd.DataFrame(bundle["region_stats"])
        fig4 = go.Figure(go.Pie(
            labels=reg_df["region"],
            values=reg_df["order_count"],
            hole=0.45,
            marker=dict(colors=["#EF9A9A", "#CE93D8", "#80DEEA", "#FFCC80"],
                        line=dict(color="#0e1117", width=2)),
        ))
        fig4.update_layout(
            paper_bgcolor="#0e1117", font_color="white",
            legend=dict(bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Product Performance Table")
    prod_table = pd.DataFrame(bundle["product_stats"])
    st.dataframe(
        prod_table.rename(columns={
            "product_name": "Product", "total_sales": "Total Sales ($)",
            "total_profit": "Total Profit ($)", "avg_quantity": "Avg Qty",
            "order_count": "Orders"
        }).style.format({
            "Total Sales ($)": "${:,.0f}", "Total Profit ($)": "${:,.0f}",
            "Avg Qty": "{:.1f}", "Orders": "{:,}"
        }),
        use_container_width=True, hide_index=True
    )

# ═══════════════════════════════════════════════════════
# TAB 4 — Pipeline & API
# ═══════════════════════════════════════════════════════
with tab4:
    st.title("⚙️ Pipeline & API")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ML Pipeline Steps")
        steps = [
            ("1. Data Ingestion", "Load `ecommerce_sales_data.csv` · `src/data/make_dataset.py`"),
            ("2. Column Inference", "Auto-detect date/customer/amount/order columns — supports any CSV schema"),
            ("3. Feature Engineering", "RFM features: frequency, monetary, recency, tenure, avg order gap · `src/features/build_features.py`"),
            ("4. Classification", "RandomForestClassifier (balanced) — predict repeat purchase probability"),
            ("5. Regression", "RandomForestRegressor (positive-only) — predict conditional spend"),
            ("6. Expected Spend", "E[Spend] = P(repeat) × Conditional Spend"),
            ("7. Evaluation", "ROC-AUC, PR-AUC (classifier) · RMSE (spend) via StratifiedKFold CV"),
            ("8. Artifact Export", "`artifacts/models/model_bundle.joblib` · `models/demo_bundle.pkl`"),
        ]
        for step, desc in steps:
            st.markdown(f"**{step}** — {desc}")

        st.subheader("Two-Stage Model Architecture")
        st.code(
            "Input: RFM features\n"
            "  │\n"
            "  ├─ Stage 1: RandomForestClassifier\n"
            "  │    → P(repeat purchase)  [0..1]\n"
            "  │\n"
            "  └─ Stage 2: RandomForestRegressor (trained on positive labels only)\n"
            "       → Conditional Spend ($)\n"
            "\n"
            "Output: Expected Spend = P(repeat) × Conditional Spend",
            language="text"
        )

    with col2:
        st.subheader("FastAPI Endpoints")
        endpoints = [
            ("GET", "/health", "Health check"),
            ("POST", "/predict", "Score one customer: RFM JSON → prob + expected_spend"),
            ("POST", "/predict/batch", "Score many customers from a list"),
        ]
        for method, path, desc in endpoints:
            color = "#4FC3F7" if method == "GET" else "#A5D6A7"
            st.markdown(
                f"<span style='background:{color};color:#000;padding:2px 6px;border-radius:3px;"
                f"font-size:0.75em;font-weight:bold'>{method}</span> "
                f"`{path}` — {desc}",
                unsafe_allow_html=True
            )

        st.markdown("**Example request:**")
        st.code(
            'curl -X POST http://localhost:8000/predict \\\n'
            '  -H "Content-Type: application/json" \\\n'
            '  -d \'{\n'
            '    "frequency_orders": 5,\n'
            '    "monetary_sum": 1200.0,\n'
            '    "monetary_mean": 240.0,\n'
            '    "active_days": 12,\n'
            '    "recency_days": 15,\n'
            '    "customer_tenure_days": 90,\n'
            '    "avg_order_gap_days": 18.0\n'
            '  }\'',
            language="bash"
        )

        st.subheader("Project Structure")
        st.code(
            "ecommerce-retention-ml-pipeline/\n"
            "├── src/\n"
            "│   ├── data/         make_dataset.py\n"
            "│   ├── features/     build_features.py  (RFM)\n"
            "│   ├── models/       train.py  evaluate.py\n"
            "│   ├── inference/    predict.py\n"
            "│   ├── api/          main.py  (FastAPI)\n"
            "│   └── utils/        io.py\n"
            "├── tests/            test_features.py\n"
            "├── configs/          base.yaml\n"
            "├── models/           demo_bundle.pkl\n"
            "├── scripts/          run_pipeline.py\n"
            "├── streamlit_app.py\n"
            "├── train_demo.py\n"
            "└── .github/workflows/ci.yml",
            language="text"
        )

# ═══════════════════════════════════════════════════════
# TAB 5 — Predict
# ═══════════════════════════════════════════════════════
with tab5:
    st.title("🔮 Customer Retention Predictor")
    st.markdown(
        "Enter RFM (Recency-Frequency-Monetary) features to predict "
        "**probability of repeat purchase** and **expected revenue**."
    )

    classifier = bundle["classifier"]
    regressor = bundle["regressor"]

    col_form, col_result = st.columns([1, 2])

    with col_form:
        st.subheader("Customer Profile")

        frequency_orders = st.number_input(
            "Orders in History Window",
            min_value=1, max_value=500, value=5,
            help="How many orders placed in the observation window"
        )
        monetary_sum = st.number_input(
            "Total Spend ($)",
            min_value=0.0, max_value=50000.0, value=800.0, step=50.0,
            help="Sum of all order amounts in the history window"
        )
        monetary_mean = st.number_input(
            "Avg Order Value ($)",
            min_value=0.0, max_value=10000.0, value=160.0, step=10.0
        )
        active_days = st.number_input(
            "Active Days",
            min_value=1, max_value=365, value=30,
            help="Number of distinct days with at least one order"
        )
        recency_days = st.number_input(
            "Recency (days since last order)",
            min_value=0, max_value=365, value=20,
            help="Lower = more recent = more likely to return"
        )
        customer_tenure_days = st.number_input(
            "Customer Tenure (days)",
            min_value=0, max_value=1000, value=60,
            help="Days between first and last order in history window"
        )
        avg_order_gap_days = st.number_input(
            "Avg Days Between Orders",
            min_value=0.0, max_value=365.0, value=12.0, step=1.0
        )

        predict_btn = st.button("Predict Retention", type="primary", use_container_width=True)

    with col_result:
        # Build input feature vector
        input_data = pd.DataFrame([{
            "frequency_orders": frequency_orders,
            "monetary_sum": monetary_sum,
            "monetary_mean": monetary_mean,
            "active_days": active_days,
            "recency_days": recency_days,
            "customer_tenure_days": customer_tenure_days,
            "avg_order_gap_days": avg_order_gap_days,
        }])
        # Ensure feature column order matches training
        input_data = input_data[feature_cols]

        prob_proba = classifier.predict_proba(input_data)
        if prob_proba.shape[1] > 1:
            prob_repeat = float(prob_proba[0, 1])
        else:
            prob_repeat = float(prob_proba[0, 0])

        if regressor is not None:
            cond_spend = float(np.clip(regressor.predict(input_data)[0], 0, None))
        else:
            cond_spend = float(bundle["dataset_stats"]["avg_order_value"])

        expected_spend = prob_repeat * cond_spend

        # Gauge
        gauge_color = "#4FC3F7" if prob_repeat < 0.5 else "#A5D6A7" if prob_repeat < 0.8 else "#FFD54F"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_repeat * 100,
            number={"suffix": "%", "font": {"size": 40, "color": gauge_color}},
            title={"text": "P(Repeat Purchase)", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": gauge_color},
                "bgcolor": "#1a1a2e",
                "bordercolor": "white",
                "steps": [
                    {"range": [0, 40], "color": "#2a1a1a"},
                    {"range": [40, 70], "color": "#1a2a1a"},
                    {"range": [70, 100], "color": "#2a2a1a"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.8,
                    "value": 50,
                },
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0e1117", font_color="white", height=280,
            margin=dict(t=60, b=20, l=30, r=30)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Metrics row
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Probability", f"{prob_repeat:.1%}")
        mc2.metric("Conditional Spend", f"${cond_spend:,.0f}")
        mc3.metric("Expected Revenue", f"${expected_spend:,.0f}")

        # Segment classification
        if prob_repeat >= 0.7:
            segment = "🟢 High-Value Retained"
            segment_note = "Strong repeat buyer — candidate for upsell / loyalty program."
        elif prob_repeat >= 0.4:
            segment = "🟡 At Risk"
            segment_note = "Moderate retention — consider win-back campaign."
        else:
            segment = "🔴 Likely to Churn"
            segment_note = "Low re-purchase probability — priority for reactivation."

        st.info(f"**Segment:** {segment}  \n{segment_note}")

        # Context: how does this customer compare to training population?
        st.subheader("Feature Context (vs Training Population)")
        ctx_cols = st.columns(len(feature_cols))
        for i, col in enumerate(feature_cols):
            fstat = feat_stats[col]
            val = input_data.iloc[0][col]
            delta_val = val - fstat["mean"]
            ctx_cols[i % len(ctx_cols)].metric(
                col.replace("_", " ").title(),
                f"{val:.1f}",
                delta=f"{delta_val:+.1f} vs avg",
                delta_color="inverse" if col == "recency_days" else "normal",
            )

        with st.expander("Input Summary"):
            st.json({
                "frequency_orders": frequency_orders,
                "monetary_sum": monetary_sum,
                "monetary_mean": monetary_mean,
                "active_days": active_days,
                "recency_days": recency_days,
                "customer_tenure_days": customer_tenure_days,
                "avg_order_gap_days": avg_order_gap_days,
                "prediction": {
                    "prob_repeat_purchase": round(prob_repeat, 4),
                    "conditional_spend": round(cond_spend, 2),
                    "expected_spend": round(expected_spend, 2),
                    "segment": segment,
                }
            })
