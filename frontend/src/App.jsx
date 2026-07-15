import React, { useCallback, useEffect, useRef, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
} from "reactflow";
import {
  Loader2,
  Play,
  RefreshCcw,
  Save,
  Server,
  Workflow,
} from "lucide-react";

import {
  api,
  PROJECT_STORAGE_KEY,
  TOKEN_STORAGE_KEY,
  WS_BASE_URL,
} from "./api";

import AutoMLNode from "./components/AutoMLNode";
import DataLoaderNode from "./components/DataLoaderNode";
import DeployNode from "./components/DeployNode";
import DriftNode from "./components/DriftNode";
import GenericPipelineNode from "./components/GenericPipelineNode";
import NodePalette, { NODE_TEMPLATES } from "./components/NodePalette";

import DatasetManager from "./components/DatasetManager";
import ModelZooPanel from "./components/ModelZooPanel";
import PredictPanel from "./components/PredictPanel";
import DriftPanel from "./components/DriftPanel";
import DeploymentPanel from "./components/DeploymentPanel";
import RegistryPanel from "./components/RegistryPanel";
import ResultsDashboard from "./components/ResultsDashboard";
import DetailsModal from "./components/DetailsModal";
import ExecutionConsole from "./components/ExecutionConsole";
import ToastStack from "./components/ToastStack";
import ProjectSelector from "./components/ProjectSelector";
import UserMenu from "./components/UserMenu";

const nodeTypes = {
  dataLoader: DataLoaderNode,
  automl: AutoMLNode,
  preprocessor: GenericPipelineNode,
  evaluator: GenericPipelineNode,
  explainer: GenericPipelineNode,
  mlflow: GenericPipelineNode,
  drift: DriftNode,
  deploy: DeployNode,
};

const edgeStyle = {
  stroke: "#64748b",
  strokeWidth: 2,
  strokeDasharray: "6 6",
};

const initialEdges = [
  {
    id: "edge_1",
    source: "node_1",
    target: "node_preprocessor",
    animated: true,
    style: edgeStyle,
  },
  {
    id: "edge_2",
    source: "node_preprocessor",
    target: "node_2",
    animated: true,
    style: edgeStyle,
  },
  {
    id: "edge_3",
    source: "node_2",
    target: "node_evaluator",
    animated: true,
    style: edgeStyle,
  },
  {
    id: "edge_4",
    source: "node_evaluator",
    target: "node_explainer",
    animated: true,
    style: edgeStyle,
  },
  {
    id: "edge_5",
    source: "node_explainer",
    target: "node_mlflow",
    animated: true,
    style: edgeStyle,
  },
  {
    id: "edge_6",
    source: "node_mlflow",
    target: "node_drift",
    animated: true,
    style: edgeStyle,
  },
  {
    id: "edge_7",
    source: "node_drift",
    target: "node_deploy",
    animated: true,
    style: edgeStyle,
  },
];

function createToastId() {
  if (window.crypto?.randomUUID) {
    return window.crypto.randomUUID();
  }

  return `toast_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function findTemplate(type) {
  return NODE_TEMPLATES.find((template) => template.type === type);
}

function sanitizeDeploymentName(value) {
  const cleaned = String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

  return cleaned || "automl-model-api";
}

function getApiErrorMessage(error) {
  const detail = error.response?.data?.detail;

  if (typeof detail === "string") {
    return detail;
  }

  if (detail && typeof detail === "object") {
    const messages = [];

    if (detail.message) {
      messages.push(detail.message);
    }

    if (Array.isArray(detail.errors)) {
      messages.push(...detail.errors);
    }

    return messages.join(" ") || JSON.stringify(detail);
  }

  return error.message || "Something went wrong.";
}

function normalizeDriftReport(payload = {}) {
  const reportId = payload.report_id || payload.id || "";
  const driftDetected =
    payload.dataset_drift ??
    payload.drift_detected ??
    payload.summary?.dataset_drift ??
    false;

  const driftShare =
    payload.drift_share ??
    payload.share_of_drifted_columns ??
    payload.summary?.share_of_drifted_columns ??
    null;

  const rawReportUrl =
    payload.html_report_url ||
    payload.html_url ||
    payload.report_url ||
    (reportId ? `/monitoring/drift/${reportId}/html` : "");

  const reportUrl = rawReportUrl.startsWith("http")
    ? rawReportUrl
    : `${api.defaults.baseURL}${rawReportUrl}`;

  return {
    ...payload,
    report_id: reportId,
    report_url: reportUrl,
    drift_detected: Boolean(driftDetected),
    drift_share: driftShare,
  };
}

export default function App() {
  const wsRef = useRef(null);
  const reactFlowWrapperRef = useRef(null);
  const nodesRef = useRef([]);

  const [reactFlowInstance, setReactFlowInstance] = useState(null);

  const [backendHealth, setBackendHealth] = useState("checking");

  const [datasets, setDatasets] = useState([]);
  const [columns, setColumns] = useState([]);
  const [experiments, setExperiments] = useState([]);
  const [models, setModels] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [pipelines, setPipelines] = useState([]);
  const [deployments, setDeployments] = useState([]);
  const [modelZooModels, setModelZooModels] = useState([]);

  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [selectedModelId, setSelectedModelId] = useState("");
  const [latestTrainedModelId, setLatestTrainedModelId] = useState("");

  const [isRunning, setIsRunning] = useState(false);
  const [logs, setLogs] = useState([]);
  const [taskStatus, setTaskStatus] = useState(null);
  const [finalResult, setFinalResult] = useState(null);

  const [consoleCollapsed, setConsoleCollapsed] = useState(false);
  const [toasts, setToasts] = useState([]);

  const [detailsModal, setDetailsModal] = useState({
    open: false,
    title: "",
    subtitle: "",
    data: null,
  });

  const updateNodeData = useCallback((nodeId, key, value) => {
    setNodes((currentNodes) =>
      currentNodes.map((node) =>
        node.id === nodeId
          ? {
              ...node,
              data: {
                ...node.data,
                [key]: value,
              },
            }
          : node
      )
    );
  }, []);

  const updateNodeDataByType = useCallback((nodeType, patch) => {
    setNodes((currentNodes) =>
      currentNodes.map((node) => {
        if (node.type !== nodeType) return node;

        const patchValue =
          typeof patch === "function" ? patch(node.data || {}) : patch;

        return {
          ...node,
          data: {
            ...node.data,
            ...patchValue,
          },
        };
      })
    );
  }, []);

  const createNodeFromTemplate = useCallback(
    (template, position) => {
      const id = `${template.type}_${Date.now().toString(16)}_${Math.random()
        .toString(16)
        .slice(2, 6)}`;

      if (template.type === "dataLoader") {
        return {
          id,
          type: "dataLoader",
          position,
          data: {
            dataset_id: "",
            target_column: "",
            datasets,
            columns,
            status: "idle",
            statusMessage: "Waiting for dataset",
            onChange: (key, value) => updateNodeData(id, key, value),
          },
        };
      }

      if (template.type === "automl") {
        return {
          id,
          type: "automl",
          position,
          data: {
            time_budget: 30,
            tuning_mode: "flaml",
            custom_model_ids: [],
            modelZooModels,
            status: "idle",
            statusMessage: "Waiting for training",
            onChange: (key, value) => updateNodeData(id, key, value),
          },
        };
      }

      if (template.type === "drift") {
        return {
          id,
          type: "drift",
          position,
          data: {
            datasets,
            current_dataset_id: "",
            auto_check: true,
            max_rows: 1000,
            status: "idle",
            statusMessage: "Select current dataset",
            lastReport: null,
            onChange: (key, value) => updateNodeData(id, key, value),
          },
        };
      }

      if (template.type === "deploy") {
        return {
          id,
          type: "deploy",
          position,
          data: {
            deployment_name: "automl-model-api",
            environment: "production",
            auto_deploy: true,
            activate: true,
            status: "idle",
            statusMessage: "Ready after training",
            lastDeployment: null,
            onChange: (key, value) => updateNodeData(id, key, value),
          },
        };
      }

      return {
        id,
        type: template.type,
        position,
        data: {
          label: template.label,
          description: template.description,
          icon: template.icon,
          color: template.color,
          nodeType: template.type,
          status: "idle",
          statusMessage: "Waiting",
        },
      };
    },
    [columns, datasets, modelZooModels, updateNodeData]
  );

  const [nodes, setNodes] = useState([
    {
      id: "node_1",
      type: "dataLoader",
      position: { x: 100, y: 80 },
      data: {
        dataset_id: "",
        target_column: "",
        datasets: [],
        columns: [],
        status: "idle",
        statusMessage: "Waiting for dataset",
        onChange: (key, value) => updateNodeData("node_1", key, value),
      },
    },
    {
      id: "node_preprocessor",
      type: "preprocessor",
      position: { x: 500, y: 80 },
      data: {
        label: "Preprocessor",
        description: "Clean missing values and encode features",
        icon: "workflow",
        color: "cyan",
        nodeType: "preprocessor",
        status: "idle",
        statusMessage: "Waiting",
      },
    },
    {
      id: "node_2",
      type: "automl",
      position: { x: 900, y: 80 },
      data: {
        time_budget: 30,
        tuning_mode: "flaml",
        custom_model_ids: [],
        modelZooModels: [],
        status: "idle",
        statusMessage: "Waiting for training",
        onChange: (key, value) => updateNodeData("node_2", key, value),
      },
    },
    {
      id: "node_evaluator",
      type: "evaluator",
      position: { x: 1300, y: 80 },
      data: {
        label: "Evaluator",
        description: "Calculate accuracy, F1, R2, MAE and RMSE",
        icon: "bar-chart",
        color: "emerald",
        nodeType: "evaluator",
        status: "idle",
        statusMessage: "Waiting",
      },
    },
    {
      id: "node_explainer",
      type: "explainer",
      position: { x: 1700, y: 80 },
      data: {
        label: "SHAP Explainer",
        description: "Generate feature importance explanations",
        icon: "sparkles",
        color: "amber",
        nodeType: "explainer",
        status: "idle",
        statusMessage: "Waiting",
      },
    },
    {
      id: "node_mlflow",
      type: "mlflow",
      position: { x: 2100, y: 80 },
      data: {
        label: "MLflow Tracker",
        description: "Log metrics, params and model artifacts",
        icon: "line-chart",
        color: "indigo",
        nodeType: "mlflow",
        status: "idle",
        statusMessage: "Waiting",
      },
    },
    {
      id: "node_drift",
      type: "drift",
      position: { x: 2500, y: 80 },
      data: {
        datasets: [],
        current_dataset_id: "",
        auto_check: true,
        max_rows: 1000,
        status: "idle",
        statusMessage: "Select current dataset",
        lastReport: null,
        onChange: (key, value) => updateNodeData("node_drift", key, value),
      },
    },
    {
      id: "node_deploy",
      type: "deploy",
      position: { x: 2900, y: 80 },
      data: {
        deployment_name: "automl-model-api",
        environment: "production",
        auto_deploy: true,
        activate: true,
        status: "idle",
        statusMessage: "Ready after training",
        lastDeployment: null,
        onChange: (key, value) => updateNodeData("node_deploy", key, value),
      },
    },
  ]);

  const [edges, setEdges] = useState(initialEdges);

  const pushToast = useCallback((type, title, message = "") => {
    const id = createToastId();

    setToasts((currentToasts) => [
      ...currentToasts,
      {
        id,
        type,
        title,
        message,
      },
    ]);

    setTimeout(() => {
      setToasts((currentToasts) =>
        currentToasts.filter((toast) => toast.id !== id)
      );
    }, 3500);
  }, []);

  const removeToast = useCallback((id) => {
    setToasts((currentToasts) =>
      currentToasts.filter((toast) => toast.id !== id)
    );
  }, []);

  const addLog = useCallback((text) => {
    setLogs((currentLogs) => [
      ...currentLogs,
      {
        time: new Date().toLocaleTimeString(),
        text,
      },
    ]);
  }, []);

  const setNodeStatusByType = useCallback(
    (nodeType, status, statusMessage = "") => {
      setNodes((currentNodes) =>
        currentNodes.map((node) =>
          node.type === nodeType
            ? {
                ...node,
                data: {
                  ...node.data,
                  status,
                  statusMessage,
                },
              }
            : node
        )
      );
    },
    []
  );

  const resetNodeStatuses = useCallback(() => {
    setNodes((currentNodes) =>
      currentNodes.map((node) => ({
        ...node,
        data: {
          ...node.data,
          status: "idle",
          statusMessage:
            node.type === "dataLoader"
              ? "Waiting for dataset"
              : node.type === "automl"
              ? "Waiting for training"
              : node.type === "drift"
              ? node.data?.current_dataset_id
                ? "Ready for drift check"
                : "Select current dataset"
              : node.type === "deploy"
              ? "Ready after training"
              : "Waiting",
          lastDeployment:
            node.type === "deploy" ? null : node.data?.lastDeployment,
          lastReport:
            node.type === "drift" ? null : node.data?.lastReport,
        },
      }))
    );
  }, []);

  const updateVisualNodesFromProgress = useCallback(
    (message = "") => {
      const text = message.toLowerCase();

      if (text.includes("preprocessing")) {
        setNodeStatusByType("preprocessor", "running", "Transforming features");
      }

      if (
        text.includes("flaml") ||
        text.includes("optuna") ||
        text.includes("optimizing") ||
        text.includes("tuning")
      ) {
        setNodeStatusByType("preprocessor", "success", "Features prepared");
        setNodeStatusByType("automl", "running", message);
      }

      if (text.includes("packaging")) {
        setNodeStatusByType("evaluator", "running", "Evaluating model");
      }

      if (text.includes("shap")) {
        setNodeStatusByType("evaluator", "success", "Metrics calculated");
        setNodeStatusByType("explainer", "running", "Generating SHAP");
      }

      if (text.includes("mlflow")) {
        setNodeStatusByType("explainer", "success", "SHAP generated");
        setNodeStatusByType("mlflow", "running", "Logging experiment");
      }
    },
    [setNodeStatusByType]
  );

  const runDriftCheckForNode = useCallback(
    async (modelIdOverride = "") => {
      const driftNode = nodesRef.current.find(
        (node) => node.type === "drift"
      );

      if (!driftNode) {
        return null;
      }

      const driftData = driftNode.data || {};
      const modelId =
        modelIdOverride || latestTrainedModelId || selectedModelId;
      const currentDatasetId = driftData.current_dataset_id;
      const maxRows = Math.max(
        2,
        Math.min(Number(driftData.max_rows || 1000), 10000)
      );

      if (!modelId) {
        updateNodeDataByType("drift", {
          status: "error",
          statusMessage: "Train or select a model first",
        });
        addLog("Drift node failed: no model is available");
        return null;
      }

      if (!currentDatasetId) {
        updateNodeDataByType("drift", {
          status: "idle",
          statusMessage: "Select current dataset",
        });
        addLog("Drift node skipped: no current dataset selected");
        return null;
      }

      try {
        updateNodeDataByType("drift", {
          status: "running",
          statusMessage: "Generating Evidently drift report",
          lastReport: null,
        });

        addLog(
          `Drift node: comparing model ${modelId} with dataset ${currentDatasetId}`
        );

        const response = await api.post("/monitoring/drift", {
          model_id: modelId,
          current_dataset_id: currentDatasetId,
          max_rows: maxRows,
        });

        const report = normalizeDriftReport(response.data);
        const driftDetected = report.drift_detected;

        updateNodeDataByType("drift", {
          status: driftDetected ? "warning" : "success",
          statusMessage: driftDetected
            ? "Dataset drift detected"
            : "No significant drift detected",
          lastReport: report,
        });

        addLog(
          `Drift node completed: ${
            report.report_id || "report generated"
          }`
        );

        pushToast(
          driftDetected ? "warning" : "success",
          driftDetected ? "Drift detected" : "Drift check complete",
          report.report_id || "Evidently report generated"
        );

        return report;
      } catch (error) {
        const message = getApiErrorMessage(error);

        updateNodeDataByType("drift", {
          status: "error",
          statusMessage: message,
        });

        addLog(`Drift node failed: ${message}`);
        pushToast("error", "Drift check failed", message);

        return null;
      }
    },
    [
      addLog,
      latestTrainedModelId,
      pushToast,
      selectedModelId,
      updateNodeDataByType,
    ]
  );

  const autoRunDriftCheck = useCallback(
    async (trainingResult) => {
      const driftNode = nodesRef.current.find(
        (node) => node.type === "drift"
      );

      if (!driftNode) {
        return null;
      }

      if (driftNode.data?.auto_check === false) {
        updateNodeDataByType("drift", {
          status: "idle",
          statusMessage: "Auto drift check disabled",
        });
        addLog("Drift node skipped: auto check disabled");
        return null;
      }

      return runDriftCheckForNode(trainingResult?.best_model_id || "");
    },
    [addLog, runDriftCheckForNode, updateNodeDataByType]
  );

  const autoDeployModel = async (trainingResult) => {
    const deployNode = nodesRef.current.find(
      (node) => node.type === "deploy"
    );

    if (!deployNode) return;

    const deployData = deployNode.data || {};

    if (deployData.auto_deploy === false) {
      updateNodeDataByType("deploy", {
        status: "idle",
        statusMessage: "Auto deploy disabled",
      });
      addLog("Deploy node skipped: auto deploy disabled");
      return;
    }

    const modelId = trainingResult?.best_model_id;

    if (!modelId) {
      updateNodeDataByType("deploy", {
        status: "error",
        statusMessage: "No trained model ID found",
      });
      addLog("Deploy node failed: no trained model ID found");
      return;
    }

    const targetColumn = trainingResult?.target_column || "model";
    const defaultName = `${targetColumn}-api`;
    const deploymentName = sanitizeDeploymentName(
      deployData.deployment_name || defaultName
    );

    const environment = deployData.environment || "production";
    const activate = deployData.activate !== false;

    try {
      updateNodeDataByType("deploy", {
        status: "running",
        statusMessage: "Creating REST API deployment",
        lastDeployment: null,
      });

      addLog(`Deploy node: creating deployment '${deploymentName}'`);

      const response = await api.post("/deployments", {
        model_id: modelId,
        deployment_name: deploymentName,
        environment,
        activate,
      });

      updateNodeDataByType("deploy", {
        status: "success",
        statusMessage: `Deployment v${response.data.version} created`,
        lastDeployment: response.data,
      });

      addLog(
        `Deploy node completed: ${response.data.versioned_endpoint_path || response.data.endpoint_path}`
      );

      pushToast(
        "success",
        "Deployment created",
        `${deploymentName} v${response.data.version}`
      );
    } catch (error) {
      const message = getApiErrorMessage(error);

      updateNodeDataByType("deploy", {
        status: "error",
        statusMessage: message,
      });

      addLog(`Deploy node failed: ${message}`);
      pushToast("error", "Deployment failed", message);
    }
  };

  const onNodesChange = useCallback((changes) => {
    setNodes((currentNodes) => applyNodeChanges(changes, currentNodes));
  }, []);

  const onEdgesChange = useCallback((changes) => {
    setEdges((currentEdges) => applyEdgeChanges(changes, currentEdges));
  }, []);

  const onConnect = useCallback((params) => {
    setEdges((currentEdges) =>
      addEdge(
        {
          ...params,
          animated: true,
          style: edgeStyle,
        },
        currentEdges
      )
    );
  }, []);

  const addNodeFromPalette = useCallback(
    (template) => {
      const position = {
        x: 460,
        y: 120 + Math.max(0, nodes.length - 2) * 120,
      };

      const newNode = createNodeFromTemplate(template, position);

      setNodes((currentNodes) => [...currentNodes, newNode]);

      pushToast("success", "Node added", `${template.label} added to canvas.`);
    },
    [createNodeFromTemplate, nodes.length, pushToast]
  );

  const onDragOver = useCallback((event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (event) => {
      event.preventDefault();

      if (!reactFlowWrapperRef.current || !reactFlowInstance) return;

      const raw = event.dataTransfer.getData("application/reactflow");

      if (!raw) return;

      const template = JSON.parse(raw);
      const bounds = reactFlowWrapperRef.current.getBoundingClientRect();

      const position = reactFlowInstance.project({
        x: event.clientX - bounds.left,
        y: event.clientY - bounds.top,
      });

      const newNode = createNodeFromTemplate(template, position);

      setNodes((currentNodes) => [...currentNodes, newNode]);

      pushToast("success", "Node added", `${template.label} added to canvas.`);
    },
    [createNodeFromTemplate, pushToast, reactFlowInstance]
  );

  const refreshHealth = async () => {
    try {
      await api.get("/health");
      setBackendHealth("online");
    } catch {
      setBackendHealth("offline");
    }
  };

  const refreshDatasets = async () => {
    const response = await api.get("/datasets");
    setDatasets(response.data.datasets || []);
  };

  const refreshModelZoo = async () => {
    const response = await api.get("/model-zoo");
    setModelZooModels(response.data.models || []);
  };

  const refreshColumns = async (datasetId) => {
    if (!datasetId) {
      setColumns([]);
      return;
    }

    const response = await api.get(`/datasets/${datasetId}/columns`);
    setColumns(response.data.columns || []);
  };

  const refreshRegistry = async () => {
    const [
      experimentResponse,
      modelResponse,
      predictionResponse,
      pipelineResponse,
      deploymentResponse,
      modelZooResponse,
    ] = await Promise.allSettled([
      api.get("/experiments"),
      api.get("/models"),
      api.get("/predictions"),
      api.get("/pipelines"),
      api.get("/deployments"),
      api.get("/model-zoo"),
    ]);

    if (experimentResponse.status === "fulfilled") {
      setExperiments(experimentResponse.value.data.experiments || []);
    }

    if (modelResponse.status === "fulfilled") {
      setModels(modelResponse.value.data.models || []);
    }

    if (predictionResponse.status === "fulfilled") {
      setPredictions(predictionResponse.value.data.predictions || []);
    }

    if (pipelineResponse.status === "fulfilled") {
      setPipelines(pipelineResponse.value.data.pipelines || []);
    }

    if (deploymentResponse.status === "fulfilled") {
      setDeployments(deploymentResponse.value.data.deployments || []);
    }

    if (modelZooResponse.status === "fulfilled") {
      setModelZooModels(modelZooResponse.value.data.models || []);
    }
  };

  const refreshAll = async () => {
    await refreshHealth();

    try {
      await refreshDatasets();
      await refreshRegistry();
    } catch (error) {
      console.error(error);
      pushToast(
        "error",
        "Refresh failed",
        error.response?.data?.detail || error.message
      );
    }
  };

  const openModelDetails = async (modelId) => {
    try {
      const response = await api.get(`/models/${modelId}`);

      setDetailsModal({
        open: true,
        title: "Model Details",
        subtitle: modelId,
        data: response.data,
      });
    } catch (error) {
      pushToast(
        "error",
        "Failed to load model",
        error.response?.data?.detail || error.message
      );
    }
  };

  const openExperimentDetails = async (experimentId) => {
    try {
      const response = await api.get(`/experiments/${experimentId}`);

      setDetailsModal({
        open: true,
        title: "Experiment Details",
        subtitle: experimentId,
        data: response.data,
      });
    } catch (error) {
      pushToast(
        "error",
        "Failed to load experiment",
        error.response?.data?.detail || error.message
      );
    }
  };

  const openDatasetProfile = async (datasetId) => {
    try {
      const response = await api.get(`/datasets/${datasetId}/profile`);

      setDetailsModal({
        open: true,
        title: "Dataset Profile",
        subtitle: datasetId,
        data: response.data,
      });
    } catch (error) {
      pushToast(
        "error",
        "Failed to load dataset profile",
        error.response?.data?.detail || error.message
      );
    }
  };

  useEffect(() => {
    nodesRef.current = nodes;
  }, [nodes]);

  useEffect(() => {
    refreshAll();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  useEffect(() => {
    refreshColumns(selectedDatasetId).catch((error) => {
      console.error(error);
      setColumns([]);
    });
  }, [selectedDatasetId]);

  useEffect(() => {
    setNodes((currentNodes) =>
      currentNodes.map((node) => {
        if (node.type !== "dataLoader") return node;

        return {
          ...node,
          data: {
            ...node.data,
            datasets,
            columns,
            dataset_id: node.data.dataset_id || selectedDatasetId,
            onChange: (key, value) => {
              if (key === "dataset_id") {
                setSelectedDatasetId(value);
                updateNodeData(node.id, "target_column", "");
              }

              updateNodeData(node.id, key, value);
            },
          },
        };
      })
    );
  }, [datasets, columns, selectedDatasetId, updateNodeData]);

  useEffect(() => {
    setNodes((currentNodes) =>
      currentNodes.map((node) => {
        if (node.type !== "automl") return node;

        return {
          ...node,
          data: {
            ...node.data,
            custom_model_ids: node.data?.custom_model_ids || [],
            modelZooModels,
            onChange: (key, value) => updateNodeData(node.id, key, value),
          },
        };
      })
    );
  }, [modelZooModels, updateNodeData]);

  useEffect(() => {
    setNodes((currentNodes) =>
      currentNodes.map((node) => {
        if (node.type !== "drift") return node;

        return {
          ...node,
          data: {
            ...node.data,
            datasets,
            onChange: (key, value) => updateNodeData(node.id, key, value),
            onRun: () => runDriftCheckForNode(),
            canRun: Boolean(latestTrainedModelId || selectedModelId),
          },
        };
      })
    );
  }, [
    datasets,
    latestTrainedModelId,
    runDriftCheckForNode,
    selectedModelId,
    updateNodeData,
  ]);

  useEffect(() => {
    setNodes((currentNodes) =>
      currentNodes.map((node) => {
        if (node.type !== "deploy") return node;

        return {
          ...node,
          data: {
            ...node.data,
            onChange: (key, value) => updateNodeData(node.id, key, value),
          },
        };
      })
    );
  }, [updateNodeData]);

  const cleanNodesForBackend = () => {
    return nodes.map((node) => {
      if (node.type === "dataLoader") {
        return {
          id: node.id,
          type: node.type,
          position: node.position,
          data: {
            dataset_id: node.data.dataset_id,
            target_column: node.data.target_column,
          },
        };
      }

      if (node.type === "automl") {
        return {
          id: node.id,
          type: node.type,
          position: node.position,
          data: {
            time_budget: Number(node.data.time_budget || 30),
            tuning_mode: node.data.tuning_mode || "flaml",
            custom_model_ids: node.data.custom_model_ids || [],
          },
        };
      }

      if (node.type === "drift") {
        return {
          id: node.id,
          type: node.type,
          position: node.position,
          data: {
            current_dataset_id: node.data.current_dataset_id || "",
            auto_check: node.data.auto_check !== false,
            max_rows: Number(node.data.max_rows || 1000),
          },
        };
      }

      if (node.type === "deploy") {
        return {
          id: node.id,
          type: node.type,
          position: node.position,
          data: {
            deployment_name: node.data.deployment_name || "automl-model-api",
            environment: node.data.environment || "production",
            auto_deploy: node.data.auto_deploy !== false,
            activate: node.data.activate !== false,
          },
        };
      }

      return {
        id: node.id,
        type: node.type,
        position: node.position,
        data: {
          label: node.data.label,
          description: node.data.description,
          nodeType: node.data.nodeType || node.type,
        },
      };
    });
  };

  const cleanEdgesForBackend = () => {
    return edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
    }));
  };

  const hydrateSavedNodes = (savedNodes) => {
    return savedNodes.map((node) => {
      if (node.type === "dataLoader") {
        return {
          ...node,
          position: node.position || { x: 120, y: 80 },
          data: {
            dataset_id: node.data?.dataset_id || "",
            target_column: node.data?.target_column || "",
            datasets,
            columns,
            status: "idle",
            statusMessage: "Loaded from saved pipeline",
            onChange: (key, value) => {
              if (key === "dataset_id") {
                setSelectedDatasetId(value);
                updateNodeData(node.id, "target_column", "");
              }

              updateNodeData(node.id, key, value);
            },
          },
        };
      }

      if (node.type === "automl") {
        return {
          ...node,
          position: node.position || { x: 120, y: 380 },
          data: {
            time_budget: node.data?.time_budget || 30,
            tuning_mode: node.data?.tuning_mode || "flaml",
            custom_model_ids: node.data?.custom_model_ids || [],
            modelZooModels,
            status: "idle",
            statusMessage: "Loaded from saved pipeline",
            onChange: (key, value) => updateNodeData(node.id, key, value),
          },
        };
      }

      if (node.type === "drift") {
        return {
          ...node,
          position: node.position || { x: 2500, y: 80 },
          data: {
            datasets,
            current_dataset_id: node.data?.current_dataset_id || "",
            auto_check: node.data?.auto_check !== false,
            max_rows: node.data?.max_rows || 1000,
            status: "idle",
            statusMessage: node.data?.current_dataset_id
              ? "Ready for drift check"
              : "Select current dataset",
            lastReport: null,
            onChange: (key, value) => updateNodeData(node.id, key, value),
            onRun: () => runDriftCheckForNode(),
            canRun: Boolean(latestTrainedModelId || selectedModelId),
          },
        };
      }

      if (node.type === "deploy") {
        return {
          ...node,
          position: node.position || { x: 2900, y: 80 },
          data: {
            deployment_name:
              node.data?.deployment_name || "automl-model-api",
            environment: node.data?.environment || "production",
            auto_deploy: node.data?.auto_deploy !== false,
            activate: node.data?.activate !== false,
            status: "idle",
            statusMessage: "Loaded from saved pipeline",
            lastDeployment: null,
            onChange: (key, value) => updateNodeData(node.id, key, value),
          },
        };
      }

      const template = findTemplate(node.type);

      return {
        ...node,
        data: {
          label: node.data?.label || template?.label || node.type,
          description:
            node.data?.description ||
            template?.description ||
            "Visual pipeline step",
          icon: node.data?.icon || template?.icon || "workflow",
          color: node.data?.color || template?.color || "blue",
          nodeType: node.data?.nodeType || node.type,
          status: "idle",
          statusMessage: "Loaded from saved pipeline",
        },
      };
    });
  };

  const hydrateSavedEdges = (savedEdges) => {
    return savedEdges.map((edge) => ({
      ...edge,
      animated: true,
      style: edgeStyle,
    }));
  };

  const loadSavedPipeline = async (graphId) => {
    try {
      const response = await api.get(`/pipelines/${graphId}`);

      const savedNodes = response.data.nodes || [];
      const savedEdges = response.data.edges || [];

      const hydratedNodes = hydrateSavedNodes(savedNodes);
      const hydratedEdges = hydrateSavedEdges(savedEdges);

      setNodes(hydratedNodes);
      setEdges(hydratedEdges);

      const dataLoaderNode = hydratedNodes.find(
        (node) => node.type === "dataLoader"
      );

      if (dataLoaderNode?.data?.dataset_id) {
        setSelectedDatasetId(dataLoaderNode.data.dataset_id);
      }

      setFinalResult(null);
      setConsoleCollapsed(false);

      addLog(`Loaded saved pipeline: ${graphId}`);
      pushToast(
        "success",
        "Pipeline loaded",
        "Saved pipeline loaded onto canvas."
      );
    } catch (error) {
      const message = error.response?.data?.detail || error.message;

      addLog(`Pipeline load failed: ${message}`);
      pushToast("error", "Pipeline load failed", message);
    }
  };

  const connectWebSocket = (taskId) => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    const token = localStorage.getItem(TOKEN_STORAGE_KEY) || "";
    const projectId = localStorage.getItem(PROJECT_STORAGE_KEY) || "";
    const wsUrl =
      `${WS_BASE_URL}/ws/task/${taskId}` +
      `?token=${encodeURIComponent(token)}` +
      `&project_id=${encodeURIComponent(projectId)}`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      addLog("WebSocket connected");
    };

    ws.onmessage = async (event) => {
      const data = JSON.parse(event.data);

      setTaskStatus(data.status);

      if (data.status === "PROGRESS") {
        setNodeStatusByType(
          "automl",
          "running",
          data.message || "Training in progress"
        );

        updateVisualNodesFromProgress(data.message || "");
      }

      if (data.message) {
        addLog(`${data.status}: ${data.message}`);
      }

      if (data.status === "SUCCESS") {
        addLog("Pipeline execution complete");

        setFinalResult(data.result);
        setSelectedModelId(data.result?.best_model_id || "");
        setLatestTrainedModelId(data.result?.best_model_id || "");
        setConsoleCollapsed(false);

        setNodeStatusByType("dataLoader", "success", "Completed");
        setNodeStatusByType("preprocessor", "success", "Features prepared");
        setNodeStatusByType("automl", "success", "Model trained successfully");
        setNodeStatusByType("evaluator", "success", "Metrics calculated");
        setNodeStatusByType("explainer", "success", "SHAP generated");
        setNodeStatusByType("mlflow", "success", "Experiment logged");
        setNodeStatusByType("drift", "running", "Preparing drift check");
        setNodeStatusByType("deploy", "idle", "Waiting for drift step");

        pushToast(
          "success",
          "Training complete",
          "A new model was added to the model registry."
        );

        await autoRunDriftCheck(data.result);
        await autoDeployModel(data.result);
        await refreshRegistry();

        setIsRunning(false);
      }

      if (data.status === "FAILURE") {
        addLog(`ERROR: ${data.error}`);

        setIsRunning(false);
        setConsoleCollapsed(false);

        setNodeStatusByType("automl", "error", data.error || "Training failed");
        setNodeStatusByType("drift", "idle", "Training failed");
        setNodeStatusByType("deploy", "idle", "Training failed");

        pushToast(
          "error",
          "Training failed",
          data.error || "Celery task failed."
        );

        await refreshRegistry();
      }
    };

    ws.onclose = () => {
      addLog("WebSocket closed");
    };

    ws.onerror = () => {
      addLog("WebSocket error");
      setNodeStatusByType("automl", "error", "WebSocket error");
      pushToast("error", "WebSocket error", "Could not read live task updates.");
    };
  };

  const handleRunPipeline = async () => {
    setIsRunning(true);
    setLogs([]);
    setTaskStatus("STARTING");
    setFinalResult(null);
    setConsoleCollapsed(false);

    resetNodeStatuses();
    setNodeStatusByType("dataLoader", "running", "Validating dataset");
    setNodeStatusByType("automl", "idle", "Waiting for data loader");

    try {
      const payload = {
        nodes: cleanNodesForBackend(),
        edges: cleanEdgesForBackend(),
      };

      addLog("Sending pipeline graph to backend");

      const response = await api.post("/pipeline/execute", payload);
      const taskId = response.data.final_state?.task_id;

      addLog("Backend accepted graph");

      setNodeStatusByType("dataLoader", "success", "Dataset accepted");
      setNodeStatusByType("preprocessor", "running", "Waiting for worker");
      setNodeStatusByType("automl", "running", "Training task queued");

      const backendLogs = response.data.log || [];
      backendLogs.forEach((entry) => addLog(entry));

      if (!taskId) {
        throw new Error("Backend did not return task_id.");
      }

      addLog(`Celery task created: ${taskId}`);
      connectWebSocket(taskId);
    } catch (error) {
      const message = error.response?.data?.detail || error.message;

      addLog(`Pipeline error: ${message}`);
      setIsRunning(false);
      setTaskStatus("FAILURE");

      setNodeStatusByType("dataLoader", "error", "Pipeline validation failed");
      setNodeStatusByType("automl", "error", message);

      pushToast("error", "Pipeline failed", message);
    }
  };

  const handleSavePipeline = async () => {
    try {
      const payload = {
        nodes: cleanNodesForBackend(),
        edges: cleanEdgesForBackend(),
      };

      const response = await api.post("/pipeline/save", payload);

      addLog(`Pipeline saved: ${response.data.graph_id}`);
      pushToast("success", "Pipeline saved", response.data.graph_id);

      await refreshRegistry();
    } catch (error) {
      const message = error.response?.data?.detail || error.message;

      addLog(`Save failed: ${message}`);
      pushToast("error", "Save failed", message);
    }
  };

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-[#f6f8fb]">
      <nav className="z-30 flex h-16 flex-shrink-0 items-center justify-between border-b border-slate-200 bg-white px-6 shadow-sm">
        <div className="flex items-center gap-4">
          <div>
            <h1 className="text-xl font-black tracking-tight text-slate-950">
              AutoML <span className="text-blue-600">Forge</span>
            </h1>
            <p className="-mt-0.5 text-xs text-slate-500">
              No-code ML pipeline builder
            </p>
          </div>

          <div
            className={`flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold ${
              backendHealth === "online"
                ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                : backendHealth === "offline"
                ? "border-red-200 bg-red-50 text-red-700"
                : "border-slate-200 bg-slate-50 text-slate-600"
            }`}
          >
            <Server size={13} />
            Backend {backendHealth}
          </div>

          <ProjectSelector />
        </div>

        <div className="flex items-center gap-2">
          <TopButton onClick={refreshAll} icon={<RefreshCcw size={16} />}>
            Refresh
          </TopButton>

          <TopButton onClick={handleSavePipeline} icon={<Save size={16} />}>
            Save
          </TopButton>

          <button
            onClick={handleRunPipeline}
            disabled={isRunning}
            className={`flex items-center gap-2 rounded-2xl px-5 py-3 text-sm font-bold text-white shadow-sm transition ${
              isRunning
                ? "cursor-not-allowed bg-slate-400"
                : "bg-blue-600 hover:bg-blue-700"
            }`}
          >
            {isRunning ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <Play size={16} fill="currentColor" />
            )}
            {isRunning ? "Executing..." : "Run Pipeline"}
          </button>

          <UserMenu />
        </div>
      </nav>

      <div className="flex flex-grow overflow-hidden">
        <aside className="soft-scroll w-[360px] flex-shrink-0 space-y-5 overflow-auto border-r border-slate-200 bg-white/70 p-5">
          <NodePalette onAddNode={addNodeFromPalette} />

          <DatasetManager
            datasets={datasets}
            selectedDatasetId={selectedDatasetId}
            onSelectDataset={setSelectedDatasetId}
            onRefresh={refreshDatasets}
            onOpenProfile={openDatasetProfile}
          />

          <ModelZooPanel
            models={modelZooModels}
            onRefresh={refreshModelZoo}
            onChanged={refreshModelZoo}
          />

          <PredictPanel
            models={models}
            selectedModelId={selectedModelId}
            onSelectModel={setSelectedModelId}
            onDone={refreshRegistry}
          />

          <DriftPanel models={models} datasets={datasets} />

          <DeploymentPanel
            models={models}
            deployments={deployments}
            onRefresh={refreshRegistry}
          />
        </aside>

        <main className="relative flex flex-grow flex-col overflow-hidden">
          <div className="pointer-events-none absolute left-6 top-5 z-10 rounded-3xl border border-slate-200 bg-white/90 px-5 py-4 shadow-sm backdrop-blur">
            <div className="flex items-center gap-3">
              <div className="grid h-10 w-10 place-items-center rounded-2xl bg-slate-950 text-white">
                <Workflow size={18} />
              </div>

              <div>
                <h2 className="text-sm font-black text-slate-900">
                  Visual Pipeline Canvas
                </h2>
                <p className="text-xs text-slate-500">
                  Drag nodes → connect pipeline → execute
                </p>
              </div>
            </div>
          </div>

          <div ref={reactFlowWrapperRef} className="relative flex-grow">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              nodeTypes={nodeTypes}
              onInit={setReactFlowInstance}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onDrop={onDrop}
              onDragOver={onDragOver}
              fitView
              fitViewOptions={{ padding: 0.25 }}
              minZoom={0.2}
              maxZoom={1.5}
            >
              <Background color="#dbe4ef" gap={20} size={1} />
              <Controls className="!rounded-2xl !border !border-slate-200 !bg-white !shadow-sm" />
              <MiniMap
                pannable
                zoomable
                className="!rounded-2xl !border !border-slate-200 !bg-white !shadow-sm"
              />
            </ReactFlow>

            <ResultsDashboard
              result={finalResult}
              onClose={() => setFinalResult(null)}
            />
          </div>

          <ExecutionConsole
            logs={logs}
            status={taskStatus}
            isCollapsed={consoleCollapsed}
            onToggle={() => setConsoleCollapsed((value) => !value)}
          />
        </main>

        <aside className="soft-scroll w-[360px] flex-shrink-0 overflow-auto border-l border-slate-200 bg-white/70 p-5">
          <RegistryPanel
            experiments={experiments}
            models={models}
            predictions={predictions}
            pipelines={pipelines}
            onRefresh={refreshRegistry}
            onSelectModel={setSelectedModelId}
            onOpenModel={openModelDetails}
            onOpenExperiment={openExperimentDetails}
            onLoadPipeline={loadSavedPipeline}
          />
        </aside>
      </div>

      <DetailsModal
        open={detailsModal.open}
        title={detailsModal.title}
        subtitle={detailsModal.subtitle}
        data={detailsModal.data}
        onClose={() =>
          setDetailsModal({
            open: false,
            title: "",
            subtitle: "",
            data: null,
          })
        }
      />

      <ToastStack toasts={toasts} onRemove={removeToast} />
    </div>
  );
}

function TopButton({ children, icon, onClick }) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-2 rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm font-bold text-slate-700 shadow-sm transition hover:bg-slate-50"
    >
      {icon}
      {children}
    </button>
  );
}