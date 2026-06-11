# 🤖 AI Data Analytics Agent

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Code Style](https://img.shields.io/badge/Code%20Style-Black-black.svg)

> **Intelligent Multi-Agent System for Automated Data Analysis, Insights, and Predictions**

An enterprise-grade AI-powered data analytics platform that automatically cleans, analyzes, visualizes, and generates insights from any dataset. Built using Python, Streamlit, Machine Learning, and Agent-Based Architecture.

---

# ✨ Features

## 🎯 Multi-Agent Architecture

### 🧹 Data Cleaning Agent

* Missing value handling
* Duplicate removal
* Outlier detection
* Data type conversion

### 📊 EDA Agent

* Statistical summaries
* Correlation analysis
* Distribution analysis
* Missing value reports

### 📈 Visualization Agent

* Histograms
* Scatter plots
* Bar charts
* Line charts
* Correlation heatmaps

### 💡 Insight Agent

* Automated business insights
* Trend detection
* Pattern discovery
* Recommendations

### 💬 Chat Agent

* Natural language interaction
* Dataset Q&A
* Summary generation
* Data exploration assistance

### 🎯 Predictive Agent

* Automated ML model training
* Feature selection
* Performance evaluation
* Predictions

### 📄 Report Agent

* PDF report generation
* Executive summaries
* Visual analytics reports

---

# 🚀 Quick Start

## Prerequisites

* Python 3.9+
* pip

## Installation

### Clone Repository

```bash
git clone https://github.com/surabhi-chandrakant/AI-Data-Analytics-Agent-.git
cd AI-Data-Analytics-Agent-
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/Mac**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

# 📖 Usage

## Step 1: Upload Dataset

Supported formats:

* CSV
* Excel (.xlsx)
* JSON

---

## Step 2: Explore Dashboard

| Tab               | Description                        |
| ----------------- | ---------------------------------- |
| 📊 Overview       | Dataset information and statistics |
| 🔍 EDA            | Exploratory data analysis          |
| 📈 Visualizations | Interactive charts                 |
| 💡 Insights       | AI-generated insights              |
| 💬 Chat           | Ask questions about your data      |
| 🎯 Predictions    | Machine learning models            |

---

## Step 3: Chat with Your Data

Example questions:

```text
How many rows are there?

Show missing values.

What is the average revenue?

List all columns.

Show correlation between variables.
```

---

## Step 4: Train Prediction Models

1. Select target column
2. Click Train Model
3. View:

   * R² Score
   * RMSE
   * Feature Importance
   * Predictions

---

## Step 5: Generate Reports

Generate downloadable PDF reports containing:

* Dataset overview
* Visualizations
* Insights
* Predictions
* Recommendations

---

# 🏗️ Project Structure

```text
AI-Data-Analytics-Agent/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
│
├── agents/
│   ├── data_cleaning_agent.py
│   ├── eda_agent.py
│   ├── visualization_agent.py
│   ├── insight_agent.py
│   ├── predictive_agent.py
│   ├── report_agent.py
│   └── chat_agent.py
│
├── utils/
│   ├── data_loader.py
│   ├── helpers.py
│   └── __init__.py
│
└── sample_data/
```

---

# 🛠️ Tech Stack

| Category         | Technologies          |
| ---------------- | --------------------- |
| Frontend         | Streamlit             |
| Data Processing  | Pandas, NumPy         |
| Machine Learning | Scikit-Learn, XGBoost |
| Visualization    | Plotly, Matplotlib    |
| LLM Integration  | Google Gemini         |
| Reporting        | ReportLab             |
| Validation       | Pydantic              |

---

# 📊 Example Workflow

```text
Upload Dataset
        ↓
Data Cleaning Agent
        ↓
EDA Agent
        ↓
Visualization Agent
        ↓
Insight Agent
        ↓
Predictive Agent
        ↓
Report Generation
```

---

# 🔧 Configuration

## Optional Gemini Integration

Create a `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
```

Without an API key, the application works in rule-based mode.

---

# 📈 Sample Dataset

```csv
Sales,Region,Customer_Age,Product,Revenue,Quantity
1000,North,25,Electronics,5000,10
1500,South,32,Clothing,7500,15
2000,West,45,Home,10000,20
```

---

# 💻 Use Cases

## Sales Analytics

* Revenue analysis
* Regional performance
* Product performance

## Customer Analytics

* Segmentation
* Customer behavior analysis
* Retention insights

## Financial Forecasting

* Revenue prediction
* Trend analysis
* Risk identification

## Business Intelligence

* KPI tracking
* Automated reporting
* Strategic recommendations

---

# 🎯 Skills Demonstrated

This project showcases:

* Python Development
* Data Analytics
* Machine Learning
* Generative AI
* Agent-Based Systems
* Streamlit Development
* Data Visualization
* Software Architecture

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create feature branch

```bash
git checkout -b feature/new-feature
```

3. Commit changes

```bash
git commit -m "Add new feature"
```

4. Push changes

```bash
git push origin feature/new-feature
```

5. Create Pull Request

---

# 📝 License

This project is licensed under the MIT License.

---

# 👩‍💻 Author

**Surabhi Chandrakant Bhor**

GitHub:
https://github.com/surabhi-chandrakant

LinkedIn:
https://www.linkedin.com/in/surabhi-chandrakant/

---

# ⭐ Support

If you found this project useful:

⭐ Star the repository

🍴 Fork the project

📢 Share with others

---

## 🚀 Why This Project Stands Out

✔ Multi-Agent Architecture

✔ Automated Data Cleaning

✔ Interactive Dashboard

✔ Machine Learning Integration

✔ Natural Language Analytics

✔ PDF Report Generation

✔ Production-Ready Structure

✔ Recruiter-Friendly Portfolio Project
