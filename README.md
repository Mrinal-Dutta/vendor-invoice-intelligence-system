# Vendor Invoice Intelligence System

**Freight Cost Prediction & Invoice Risk Flagging**

---

## 📌 Table of Contents

- <a href="#project-overview">Project Overview</a>
- <a href="#business-objectives">Business Objectives</a>
- <a href="#data-sources">Data Sources</a>
- <a href="#eda">Exploratory Data Analysis</a>
- <a href="#feature-engineering">Feature Engineering</a>
- <a href="#models-used">Models Used</a>
- <a href="#metrics">Evaluation Metrics</a>
- <a href="#feature-importance">Feature Importance</a>
- <a href="#application">Application</a>
- <a href="#project-structure">Project Structure</a>
- <a href="#installation">Installation</a>
- <a href="#how-to-run-this-project">How to Run This Project</a>
- <a href="#future-improvements">Future Improvements</a>
- <a href="#author--contact">Author & Contact</a>

---

<h2><a class="anchor" id="project-overview"></a>📌 Project Overview</h2>

This project implements an **end-to-end machine learning system** designed to support finance teams by:

1. **Predicting expected freight cost** for vendor invoices.
2. **Flagging high-risk invoices** that require manual review due to abnormal cost, freight, or operational patterns.

---

<h2><a class="anchor" id="business-objectives"></a>🎯 Business Objectives</h2>

- Automate freight cost estimation.
- Identify abnormal vendor invoice transactions.
- Reduce manual invoice review efforts.
- Improve financial accuracy and operational efficiency.
- Help finance teams prioritize high-risk invoices.

---

<h2><a class="anchor" id="data-sources"></a>📂 Data Sources</h2>

The project uses an SQLite database containing:

- purchases
- purchase_prices
- vendor_invoice
- begin_inventory
- end_inventory

Database: `inventory.db`

---

<h2><a class="anchor" id="eda"></a>🔍 Exploratory Data Analysis</h2>

- Data understanding and inspection.
- Missing value analysis.
- Duplicate value detection.
- Statistical analysis.
- Outlier detection.
- Data visualization.
- Feature relationship analysis.

---

<h2><a class="anchor" id="feature-engineering"></a>⚙️ Feature Engineering</h2>

Important features created for model training:

- Invoice Quantity
- Invoice Dollars
- Freight
- Total Item Quantity
- Total Item Dollars
- Total Brands
- Days PO to Invoice
- Average Receiving Delay

---

<h2><a class="anchor" id="models-used"></a>🤖 Models Used</h2>

**Regression Models:**

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

**Classification Models:**

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

---

<h2><a class="anchor" id="metrics"></a>📊 Evaluation Metrics</h2>

**Regression:**
- MAE
- RMSE
- R² Score

**Classification:**
- Accuracy
- Precision
- Recall
- F1 Score

---

<h2><a class="anchor" id="feature-importance"></a>📈 Feature Importance</h2>

Random Forest Feature Importance is used to identify the most influential features in predicting invoice risk.

---

<h2><a class="anchor" id="application"></a>🖥️ Application</h2>

The application provides:

- Freight Cost Prediction.
- Vendor Invoice Risk Flagging.
- Prediction results for new input data.
- Support for manual invoice review decisions.

---

<h2><a class="anchor" id="project-structure"></a>📁 Project Structure</h2>

```text
Vendor-Invoice-Intelligence-System/
│
├── data/
│
├── freight_cost_prediction/
│
├── invoice_flagging/
│
├── inference/
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
│
├── models/
│   ├── freight_model.pkl
│   ├── invoice_flagging_model.pkl
│   └── invoice_scaler.pkl
│
├── notebooks/
│
├── app.py
├── requirements.txt
└── README.md
```