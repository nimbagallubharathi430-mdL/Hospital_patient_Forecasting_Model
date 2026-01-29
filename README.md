# 🏥 AI Hospital Resource Command Center

### 🚀 Project Overview
A machine learning-powered dashboard that helps hospitals predict patient inflow and optimize doctor staffing. By analyzing historical trends, weekly seasonality, and annual patterns (like flu seasons), this system prevents understaffing and reduces operational costs.

### 🛠️ Tech Stack
* **Language:** Python 3.9
* **Framework:** Streamlit (Modular Architecture)
* **ML Engine:** Facebook Prophet (Additive Regression)
* **Google Technologies:** Google Fonts API, Google Maps Integration

### ⚙️ How It Works
1.  **Data Simulation:** A custom engine generates realistic patient traffic patterns (Trend + Seasonality + Noise).
2.  **Forecasting:** The Prophet model decomposes the time series to predict future demand.
3.  **Resource Allocation:** The system calculates the exact number of doctors needed based on patient-to-doctor ratios.

### 📸 Features
* Interactive Forecasting Dashboard
* Google Maps Integration for Location Services
* Real-time Accuracy Metrics (MAE)
* Dark Mode / Sci-Fi UI Interface
