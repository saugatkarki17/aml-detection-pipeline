import streamlit as st
import requests
from datetime import datetime

BASE_URL = "http://localhost:8000"  # link for API

st.set_page_config(page_title="AML Alert System", layout="wide")
st.title("AML Alert System")

view = st.sidebar.radio("Choose View", ["Submit Transaction", "View Alerts"])

# Transaction form
if view == "Submit Transaction":
    st.header("Submit a Transaction")

    with st.form("txn_form"):
        amount_paid = st.number_input("Amount Paid", min_value=0.0, step=1.0)
        amount_received = st.number_input("Amount Received", min_value=0.0, step=1.0)
        txn_count = st.number_input("Transaction Count", min_value=1)
        total_sent = st.number_input("Total Sent", min_value=0.0, step=1.0)
        hour_of_day = st.slider("Hour of Day", 0, 23, 12)
        day_of_week = st.selectbox("Day of Week", list(range(7)), format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
        payment_format = st.selectbox("Payment Format", ["Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"])
        payment_currency = st.selectbox("Payment Currency", [
            "Bitcoin", "Brazil Real", "Canadian Dollar", "Euro", "Mexican Peso", "Ruble", "Rupee",
            "Saudi Riyal", "Shekel", "Swiss Franc", "UK Pound", "US Dollar", "Yen", "Yuan"
        ])
        receiving_currency = st.selectbox("Receiving Currency", [
            "Bitcoin", "Brazil Real", "Canadian Dollar", "Euro", "Mexican Peso", "Ruble", "Rupee",
            "Saudi Riyal", "Shekel", "Swiss Franc", "UK Pound", "US Dollar", "Yen", "Yuan"
        ])

        submitted = st.form_submit_button("Submit")

        if submitted:
            payload = {
                "amount_paid": amount_paid,
                "amount_received": amount_received,
                "txn_count": txn_count,
                "total_sent": total_sent,
                "hour_of_day": hour_of_day,
                "day_of_week": day_of_week,
                "payment_format": payment_format,
                "payment_currency": payment_currency,
                "receiving_currency": receiving_currency
            }

            try:
                res = requests.post(f"{BASE_URL}/predict", json=payload)
                res.raise_for_status()
                result = res.json()
                st.success("Prediction Complete")
                st.json(result)
            except Exception as e:
                st.error(f"API Error: {e}")

# Admin View (Dummy login)
elif view == "View Alerts":
    st.header("Admin Login")

    #hardcoded credentials (for demo only)
    ADMIN_USERNAME = "admin"
    ADMIN_PASSWORD = "secure123"

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login = st.form_submit_button("Login")

            if login:
                if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                    st.success("Login successful")
                    st.session_state.logged_in = True
                else:
                    st.error("Incorrect username or password")
    else:
        st.subheader("Flagged Alerts")

        try:
            res = requests.get(f"{BASE_URL}/alerts")
            res.raise_for_status()
            alerts = res.json()

            if alerts:
                df = []
                for alert in alerts:
                    df.append({
                        "Timestamp": alert["timestamp"],
                        "Paid": alert["amount_paid"],
                        "Received": alert["amount_received"],
                        "Risk": alert["risk_level"],
                        "Probability": alert["probability"],
                        "Format": alert["payment_format"],
                        "Currency Flow": f"{alert['payment_currency']} → {alert['receiving_currency']}"
                    })
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No alerts found.")
        except Exception as e:
            st.error(f"Failed to fetch alerts: {e}")
