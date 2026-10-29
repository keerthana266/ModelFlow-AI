# ModelFlow AI — No-Code AutoML Platform

ModelFlow AI is a web-based AutoML platform designed to simplify machine learning workflows. It aims to help users prepare datasets, configure experiments, train models, and evaluate performance through an interactive interface instead of writing an entire machine learning pipeline manually.

## Overview

Building machine learning models often requires knowledge of data preprocessing, algorithm selection, hyperparameter tuning, and model evaluation. ModelFlow AI brings these steps into a unified interface to make experimentation more accessible and organized.

The project combines a React frontend with a FastAPI backend and supporting services for dataset management, asynchronous processing, and machine learning experiments.

## Key Features

- **Interactive Web Interface:** Work with machine learning workflows through a React-based frontend.
- **Dataset Management:** Upload datasets and inspect their columns before training.
- **Pipeline Configuration:** Configure dataset inputs, target columns, and training settings.
- **Automated Model Training:** Integrate automated machine learning tools to explore suitable models.
- **Experiment Execution:** Use asynchronous task processing for training workloads.
- **Model Evaluation:** Review model performance metrics to compare experiment results.
- **Extensible Architecture:** Designed to support additional capabilities such as hyperparameter optimization, experiment tracking, and model explainability.

*Feature availability depends on the current implementation and service configuration.*

## Technology Stack

| Component | Technologies |
|---|---|
| Frontend | React, Vite, JavaScript |
| Backend API | Python, FastAPI |
| Database | PostgreSQL |
| Background Tasks | Celery, Redis |
| Machine Learning | FLAML and related Python ML libraries |
| Database Integration | SQLAlchemy, Alembic |
| Infrastructure | Docker, Docker Compose |

## Architecture

```text
          User
           |
           v
    React + Vite UI
           |
           v
       FastAPI
       /     \
      v       v
PostgreSQL   Redis
                 |
                 v
               Celery
                 |
                 v
       ML Training Tasks
                 |
                 v
       Experiment Results
```

The frontend communicates with the backend API. PostgreSQL provides persistent data storage, while Redis and Celery support background task processing. Machine learning components execute training and evaluation workflows.

## Project Structure

```text
ModelFlow-AI/
├── backend/
│   ├── app/
│   ├── alembic/
│   ├── model_runtime/
│   └── docker-compose.yml
├── frontend/
│   ├── src/
│   ├── package.json
│   └── package-lock.json
└── README.md
```

## Getting Started

### Prerequisites

Install the following before running the project:

- Python 3.11 or a compatible version
- Node.js and npm
- Docker Desktop
- Git

### 1. Clone the repository

```bash
git clone https://github.com/keerthana266/ModelFlow-AI.git
cd ModelFlow-AI
```

### 2. Start the supporting services

Open Docker Desktop, then start the configured services from the backend directory:

```bash
cd backend
docker compose up -d postgres redis minio
```

The services and their ports must match your local Docker Compose configuration.

### 3. Configure the backend

Create or update the backend environment configuration using the variables required by the project. Configure database credentials, Redis, and any required object-storage settings for your environment.

Install the backend dependencies using the project's dependency configuration, then activate the backend virtual environment.

### 4. Start the backend API

From the `backend` directory, activate your virtual environment and run:

```bash
uvicorn app.main:app --reload
```

The API should be available at:

http://127.0.0.1:8000

Health check:

http://127.0.0.1:8000/health

### 5. Start the frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite in your terminal, typically:

http://localhost:5173

**Note:** Exact environment variables, dependency installation commands, and service requirements may vary with the repository configuration.

## Example Workflow

1. Open the web interface.
2. Upload a dataset.
3. Inspect the dataset and select the target column.
4. Configure the machine learning pipeline.
5. Start a training experiment.
6. Review the results and evaluation metrics.

## Future Improvements

- Enhanced automated feature engineering
- More robust hyperparameter optimization
- Experiment comparison and model versioning
- Model explainability with SHAP
- Dataset and model drift monitoring
- Model deployment and inference endpoints
- Improved experiment history and reporting

## Learning Objectives

This project provides practical experience with full-stack development, REST API integration, database management, asynchronous processing, and machine learning workflow automation.

## Author

**Keerthana Dasari**

GitHub: [@keerthana266](https://github.com/keerthana266)

## License

Refer to the repository's license file for the applicable license and usage terms.
