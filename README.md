# HealthSync – Smart Health Monitoring System

A Smart Health Monitoring System designed to collect, process, and monitor patient health readings through a web-based dashboard. The system combines IoT-based data acquisition, backend APIs, database storage, and machine learning to identify unusual health readings and generate alerts.

## Project Overview

HealthSync aims to make health monitoring more accessible by providing a centralized platform for viewing patient health information.

The system receives health telemetry, processes the readings, stores records, and presents health summaries and alerts through a web dashboard.

## Features

- Health telemetry collection through REST APIs
- FastAPI-based backend server
- SQLite database integration
- Hybrid machine learning-based health assessment
- Health status classification: Normal, Watch, and Critical
- Quality checks for incoming sensor readings
- Device and caregiver token authentication
- Web-based health monitoring dashboard
- ESP32 edge-node code for hardware integration

## Technology Stack

| Component | Technologies |
|---|---|
| Backend | Python, FastAPI |
| Database | SQLite |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Frontend | HTML, CSS, JavaScript |
| Hardware | ESP32, MAX30102, DS18B20 |

## Project Structure

```text
Smart-Health-Monitoring-System/
│
├── esp32_edge_node/
│   ├── esp32_edge_node.ino
│   └── secrets.h.example
│
├── results/
│   ├── metrics.json
│   └── evaluation graphs
│
├── cloud_server_dashboard.py
├── dashboard_ui.py
├── ml_hybrid_pipeline.py
├── settings.py
├── run_experiments_and_plots.py
│
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Smart-Health-Monitoring-System.git
cd Smart-Health-Monitoring-System
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Use `.env.example` as a reference and configure your own environment variables.

For Windows PowerShell:

```powershell
$env:HEALTH_DEVICE_TOKEN="your_device_token"
$env:HEALTH_CAREGIVER_TOKEN="your_caregiver_token"
```

Never publish actual credentials or secret tokens.

## Running the Application

Start the FastAPI server:

```bash
uvicorn cloud_server_dashboard:app --reload --host 127.0.0.1 --port 8000
```

Open the following URLs in your browser:

- **HealthSync Portal:** http://127.0.0.1:8000/portal
- **Health Dashboard:** http://127.0.0.1:8000/dashboard
- **API Documentation:** http://127.0.0.1:8000/docs

## API Overview

| Endpoint | Purpose |
|---|---|
| `/api/v1/telemetry` | Receive health telemetry |
| `/api/v1/summary` | Retrieve health summaries |
| `/api/v1/alerts` | Retrieve health alerts |

## Project Status

The backend, dashboard, database integration, and simulated telemetry workflow have been developed and tested. The repository also includes ESP32 edge-node code for hardware integration. Physical sensor integration and real-world validation should be verified before production deployment.

## Future Enhancements

- Complete physical sensor integration
- Improve real-time health monitoring
- Enhance patient history and trend analysis
- Deploy the application to a cloud platform
- Improve alert notification mechanisms

## Disclaimer

This project is developed for academic and educational purposes. It is a prototype and is not intended to replace professional medical advice, diagnosis, or treatment.
