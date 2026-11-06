
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📦",
    layout="wide"
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📦 Vendor Invoice Intelligence Portal")

st.subheader(
    "AI-Driven Freight Cost Prediction & Invoice Risk Flagging"
)

st.markdown("""
This machine learning application helps finance teams:

- Forecast freight costs accurately.
- Identify high-risk vendor invoices.
- Support finance and procurement decisions.
- Reduce manual invoice review efforts.
""")

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Model Selection")

selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    [
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    ]
)

st.sidebar.markdown("""
---

### Business Impact

- 📈 Improved cost forecasting
- 🛡️ Reduced invoice risk
- ⚡ Faster finance operations
""")


# ==================================================
# MODULE 1: FREIGHT COST PREDICTION
# ==================================================

if selected_model == "Freight Cost Prediction":

    st.header("🚚 Freight Cost Prediction")

    st.write("""
    Predict expected freight cost using invoice quantity
    and invoice dollar amount.
    """)

    with st.form("freight_form"):

        col1, col2 = st.columns(2)

        with col1:
            quantity = st.number_input(
                "Quantity",
                min_value=1,
                value=1200
            )

        with col2:
            dollars = st.number_input(
                "Invoice Dollars ($)",
                min_value=1.0,
                value=5000.0
            )

        submit_freight = st.form_submit_button(
            "Predict Freight Cost"
        )

    if submit_freight:

        input_data = {
            "Quantity": [quantity],
            "Dollars": [dollars]
        }

        try:
            prediction = predict_freight_cost(input_data)

            predicted_freight = prediction[
                "Predicted_Freight"
            ].iloc[0]

            st.success("Freight prediction completed successfully.")

            st.metric(
                label="Estimated Freight Cost",
                value=f"${predicted_freight:,.2f}"
            )

        except Exception as e:
            st.error(f"Prediction Error: {e}")


# ==================================================
# MODULE 2: INVOICE RISK FLAGGING
# ==================================================

elif selected_model == "Invoice Manual Approval Flag":

    st.header("🛡️ Invoice Risk Flagging & Manual Review")

    st.write("""
    Predict whether a vendor invoice requires manual approval
    based on invoice, freight and purchase patterns.
    """)

    with st.form("invoice_flag_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            invoice_quantity = st.number_input(
                "Invoice Quantity",
                min_value=1,
                value=100
            )

            invoice_dollars = st.number_input(
                "Invoice Dollars ($)",
                min_value=1.0,
                value=352.95
            )

            freight = st.number_input(
                "Freight Cost",
                min_value=0.0,
                value=1.73
            )

        with col2:

            total_brands = st.number_input(
                "Total Brands",
                min_value=0,
                value=5
            )

            total_item_quantity = st.number_input(
                "Total Item Quantity",
                min_value=1,
                value=162
            )

        with col3:

            days_po_to_invoice = st.number_input(
                "Days PO to Invoice",
                min_value=0,
                value=10
            )

            total_item_dollars = st.number_input(
                "Total Item Dollars ($)",
                min_value=1.0,
                value=2476.0
            )

        submit_flag = st.form_submit_button(
            "Evaluate Invoice Risk"
        )

    if submit_flag:

        input_data = {
            "invoice_quantity": [invoice_quantity],
            "invoice_dollars": [invoice_dollars],
            "Freight": [freight],
            "total_brands": [total_brands],
            "total_item_quantity": [total_item_quantity],
            "days_po_to_invoice": [days_po_to_invoice],
            "total_item_dollars": [total_item_dollars]
        }

        try:
            prediction = predict_invoice_flag(input_data)

            flag_prediction = prediction[
                "Predicted_Flag"
            ].iloc[0]

            st.divider()

            if int(flag_prediction) == 1:

                st.error(
                    "⚠️ Invoice requires Manual Approval "
                    "(High Risk / Anomaly Detected)."
                )

            else:

                st.success(
                    "✅ Invoice is Safe for Auto-Approval."
                )

        except Exception as e:
            st.error(f"Prediction Error: {e}")
            import traceback
            st.code(traceback.format_exc())
