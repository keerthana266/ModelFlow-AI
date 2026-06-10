# AML005_AutoMLForge
Built AutoML Forge: no-code ML pipeline builder using React Flow; FLAML+Optuna AutoML with SHAP explainability; auto-deploys trained models as versioned REST APIs with drift monitoring

Domain experts (doctors, bankers, agronomists) build ML pipelines without code: drag-drop dataset → auto-EDA → feature suggestion (SHAP-based importance) → model selection tournament (AutoML) → hyperparameter tuning (Optuna) → explainability dashboard (SHAP/LIME) → one-click deploy as REST API → drift monitoring. Supports tabular, image, text, time-series. Versioned experiments. Collaboration. Custom model zoo (upload your PyTorch model, we deploy it).

Pipeline builder: React Flow (node graph editor) → pipeline JSON → execution engine (Python, Celery workers) → each node: data loader / preprocessor / AutoML (FLAML + Optuna) / evaluator / explainer (SHAP) / deployer (FastAPI auto-gen + Docker build). Experiment tracking: MLflow. Model registry: S3 + PostgreSQL. Drift monitor: Evidently AI. GPU workers: Modal/RunPod. Real-time logs: WebSocket.
