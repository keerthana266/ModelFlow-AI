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

import { api, WS_BASE_URL } from "./api";

import AutoMLNode from "./components/AutoMLNode";
import DataLoaderNode from "./components/DataLoaderNode";
import DatasetManager from "./components/DatasetManager";
import PredictPanel from "./components/PredictPanel";
import RegistryPanel from "./components/RegistryPanel";
import ResultsDashboard from "./components/ResultsDashboard";
import DetailsModal from "./components/DetailsModal";
import ExecutionConsole from "./components/ExecutionConsole";
import ToastStack from "./components/ToastStack";
import DriftPanel from "./components/DriftPanel";

const nodeTypes = {
  dataLoader: DataLoaderNode,
  automl: AutoMLNode,
};

const initialEdges = [
  {
    id: "edge_1",
    source: "node_1",
    target: "node_2",
    animated: true,
    style: {
      stroke: "#64748b",
      strokeWidth: 2,
      strokeDasharray: "6 6",
    },
  },
];

function createToastId() {
  if (window.crypto?.randomUUID) {
    return window.crypto.randomUUID();
  }

  return `toast_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

export default function App() {
  const wsRef = useRef(null);

  const [backendHealth, setBackendHealth] = useState("checking");

  const [datasets, setDatasets] = useState([]);
  const [columns, setColumns] = useState([]);
  const [experiments, setExperiments] = useState([]);
  const [models, setModels] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [pipelines, setPipelines] = useState([]);

  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [selectedModelId, setSelectedModelId] = useState("");

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

  const [nodes, setNodes] = useState([
    {
      id: "node_1",
      type: "dataLoader",
      position: { x: 120, y: 80 },
      data: {
        dataset_id: "",
        target_column: "",
        datasets: [],
        columns: [],
        onChange: (key, value) => updateNodeData("node_1", key, value),
      },
    },
    {
      id: "node_2",
      type: "automl",
      position: { x: 120, y: 380 },
      data: {
        time_budget: 30,
        onChange: (key, value) => updateNodeData("node_2", key, value),
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
          style: {
            stroke: "#64748b",
            strokeWidth: 2,
            strokeDasharray: "6 6",
          },
        },
        currentEdges
      )
    );
  }, []);

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
    ] = await Promise.allSettled([
      api.get("/experiments"),
      api.get("/models"),
      api.get("/predictions"),
      api.get("/pipelines"),
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
        if (node.id !== "node_1") return node;

        return {
          ...node,
          data: {
            ...node.data,
            datasets,
            columns,
            dataset_id: selectedDatasetId || node.data.dataset_id,
            onChange: (key, value) => {
              if (key === "dataset_id") {
                setSelectedDatasetId(value);
                updateNodeData("node_1", "target_column", "");
              }

              updateNodeData("node_1", key, value);
            },
          },
        };
      })
    );
  }, [datasets, columns, selectedDatasetId, updateNodeData]);

  useEffect(() => {
    setNodes((currentNodes) =>
      currentNodes.map((node) => {
        if (node.id !== "node_2") return node;

        return {
          ...node,
          data: {
            ...node.data,
            onChange: (key, value) => updateNodeData("node_2", key, value),
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
          },
        };
      }

      return {
        id: node.id,
        type: node.type,
        position: node.position,
        data: {},
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
            onChange: (key, value) => updateNodeData(node.id, key, value),
          },
        };
      }

      return {
        ...node,
        data: node.data || {},
      };
    });
  };

  const hydrateSavedEdges = (savedEdges) => {
    return savedEdges.map((edge) => ({
      ...edge,
      animated: true,
      style: {
        stroke: "#64748b",
        strokeWidth: 2,
        strokeDasharray: "6 6",
      },
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

    const ws = new WebSocket(`${WS_BASE_URL}/ws/task/${taskId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      addLog("WebSocket connected");
    };

    ws.onmessage = async (event) => {
      const data = JSON.parse(event.data);
      setTaskStatus(data.status);

      if (data.message) {
        addLog(`${data.status}: ${data.message}`);
      }

      if (data.status === "SUCCESS") {
        addLog("Pipeline execution complete");

        setFinalResult(data.result);
        setSelectedModelId(data.result?.best_model_id || "");
        setIsRunning(false);
        setConsoleCollapsed(false);

        pushToast(
          "success",
          "Training complete",
          "A new model was added to the model registry."
        );

        await refreshRegistry();
      }

      if (data.status === "FAILURE") {
        addLog(`ERROR: ${data.error}`);

        setIsRunning(false);
        setConsoleCollapsed(false);

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
      pushToast("error", "WebSocket error", "Could not read live task updates.");
    };
  };

  const handleRunPipeline = async () => {
    setIsRunning(true);
    setLogs([]);
    setTaskStatus("STARTING");
    setFinalResult(null);
    setConsoleCollapsed(false);

    try {
      const payload = {
        nodes: cleanNodesForBackend(),
        edges: cleanEdgesForBackend(),
      };

      addLog("Sending pipeline graph to backend");

      const response = await api.post("/pipeline/execute", payload);
      const taskId = response.data.final_state?.task_id;

      addLog("Backend accepted graph");

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
        </div>
      </nav>

      <div className="flex flex-grow overflow-hidden">
        <aside className="soft-scroll w-[360px] flex-shrink-0 space-y-5 overflow-auto border-r border-slate-200 bg-white/70 p-5">
          <DatasetManager
            datasets={datasets}
            selectedDatasetId={selectedDatasetId}
            onSelectDataset={setSelectedDatasetId}
            onRefresh={refreshDatasets}
            onOpenProfile={openDatasetProfile}
          />

          <PredictPanel
            models={models}
            selectedModelId={selectedModelId}
            onSelectModel={setSelectedModelId}
            onDone={refreshRegistry}
          />
          <DriftPanel models={models} datasets={datasets} />
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
                  Data Loader → AutoML Engine → Registry
                </p>
              </div>
            </div>
          </div>

          <div className="relative flex-grow">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              nodeTypes={nodeTypes}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              fitView
              fitViewOptions={{ padding: 0.35 }}
              minZoom={0.45}
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