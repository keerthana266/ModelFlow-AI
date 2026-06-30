import React, { useCallback, useEffect, useRef, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
} from "reactflow";
import {
  CheckCircle2,
  Loader2,
  Play,
  RefreshCcw,
  Save,
  Server,
  Terminal,
  XCircle,
} from "lucide-react";

import { api, WS_BASE_URL } from "./api";
import AutoMLNode from "./components/AutoMLNode";
import DataLoaderNode from "./components/DataLoaderNode";
import DatasetManager from "./components/DatasetManager";
import PredictPanel from "./components/PredictPanel";
import RegistryPanel from "./components/RegistryPanel";
import ResultsDashboard from "./components/ResultsDashboard";

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
  },
];

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

  const updateNodeData = useCallback((nodeId, key, value) => {
    setNodes((nds) =>
      nds.map((node) =>
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
      position: { x: 250, y: 80 },
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
      position: { x: 250, y: 360 },
      data: {
        time_budget: 30,
        onChange: (key, value) => updateNodeData("node_2", key, value),
      },
    },
  ]);

  const [edges, setEdges] = useState(initialEdges);

  const onNodesChange = useCallback(
    (changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
    []
  );

  const onEdgesChange = useCallback(
    (changes) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    []
  );

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge({ ...params, animated: true }, eds)),
    []
  );

  const addLog = (text) => {
    setLogs((prev) => [
      ...prev,
      {
        time: new Date().toLocaleTimeString(),
        text,
      },
    ]);
  };

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
    const [expRes, modelRes, predRes, pipeRes] = await Promise.allSettled([
      api.get("/experiments"),
      api.get("/models"),
      api.get("/predictions"),
      api.get("/pipelines"),
    ]);

    if (expRes.status === "fulfilled") {
      setExperiments(expRes.value.data.experiments || []);
    }

    if (modelRes.status === "fulfilled") {
      setModels(modelRes.value.data.models || []);
    }

    if (predRes.status === "fulfilled") {
      setPredictions(predRes.value.data.predictions || []);
    }

    if (pipeRes.status === "fulfilled") {
      setPipelines(pipeRes.value.data.pipelines || []);
    }
  };

  const refreshAll = async () => {
    await refreshHealth();

    try {
      await refreshDatasets();
      await refreshRegistry();
    } catch (error) {
      console.error(error);
    }
  };

  useEffect(() => {
    refreshAll();

    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  useEffect(() => {
    refreshColumns(selectedDatasetId);
  }, [selectedDatasetId]);

  useEffect(() => {
    setNodes((nds) =>
      nds.map((node) => {
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
    setNodes((nds) =>
      nds.map((node) => {
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
          data: {
            time_budget: Number(node.data.time_budget || 30),
          },
        };
      }

      return {
        id: node.id,
        type: node.type,
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

  const connectWebSocket = (taskId) => {
    if (wsRef.current) wsRef.current.close();

    const ws = new WebSocket(`${WS_BASE_URL}/ws/task/${taskId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      addLog("🟢 WebSocket connected.");
    };

    ws.onmessage = async (event) => {
      const data = JSON.parse(event.data);
      setTaskStatus(data.status);

      if (data.message) {
        addLog(`[${data.status}] ${data.message}`);
      }

      if (data.status === "SUCCESS") {
        addLog("✅ Pipeline execution complete.");
        setFinalResult(data.result);
        setSelectedModelId(data.result?.best_model_id || "");
        setIsRunning(false);
        await refreshRegistry();
      }

      if (data.status === "FAILURE") {
        addLog(`❌ ERROR: ${data.error}`);
        setIsRunning(false);
        await refreshRegistry();
      }
    };

    ws.onclose = () => {
      addLog("🔴 WebSocket closed.");
    };

    ws.onerror = () => {
      addLog("❌ WebSocket error.");
    };
  };

  const handleRunPipeline = async () => {
    setIsRunning(true);
    setLogs([]);
    setTaskStatus("STARTING");
    setFinalResult(null);

    try {
      const payload = {
        nodes: cleanNodesForBackend(),
        edges: cleanEdgesForBackend(),
      };

      addLog("🚀 Sending pipeline graph to backend...");

      const response = await api.post("/pipeline/execute", payload);
      const taskId = response.data.final_state?.task_id;

      addLog("✅ Backend accepted graph.");
      addLog(JSON.stringify(response.data.log || []));

      if (!taskId) {
        throw new Error("Backend did not return task_id.");
      }

      addLog(`📌 Celery task created: ${taskId}`);
      connectWebSocket(taskId);
    } catch (error) {
      addLog(`❌ Pipeline error: ${error.response?.data?.detail || error.message}`);
      setIsRunning(false);
      setTaskStatus("FAILURE");
    }
  };

  const handleSavePipeline = async () => {
    try {
      const payload = {
        nodes: cleanNodesForBackend(),
        edges: cleanEdgesForBackend(),
      };

      const response = await api.post("/pipeline/save", payload);
      addLog(`💾 Pipeline saved: ${response.data.graph_id}`);
      await refreshRegistry();
    } catch (error) {
      addLog(`❌ Save failed: ${error.response?.data?.detail || error.message}`);
    }
  };

  return (
    <div className="h-screen w-screen flex flex-col bg-slate-50 overflow-hidden">
      <nav className="h-14 bg-white border-b border-slate-200 flex items-center justify-between px-6 shadow-sm z-20 flex-shrink-0">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-bold text-slate-800 tracking-tight">
            AutoML <span className="text-blue-600">Forge</span>
          </h1>

          <div
            className={`text-xs px-2 py-1 rounded-full flex items-center gap-1 ${
              backendHealth === "online"
                ? "bg-green-50 text-green-700 border border-green-200"
                : backendHealth === "offline"
                ? "bg-red-50 text-red-700 border border-red-200"
                : "bg-slate-50 text-slate-600 border border-slate-200"
            }`}
          >
            <Server size={12} />
            Backend {backendHealth}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={refreshAll}
            className="flex items-center gap-2 px-3 py-2 rounded-md font-medium transition-colors text-sm bg-white border border-slate-200 text-slate-700 hover:bg-slate-50"
          >
            <RefreshCcw size={15} />
            Refresh
          </button>

          <button
            onClick={handleSavePipeline}
            className="flex items-center gap-2 px-3 py-2 rounded-md font-medium transition-colors text-sm bg-white border border-slate-200 text-slate-700 hover:bg-slate-50"
          >
            <Save size={15} />
            Save Pipeline
          </button>

          <button
            onClick={handleRunPipeline}
            disabled={isRunning}
            className={`flex items-center gap-2 px-4 py-2 rounded-md font-medium transition-colors text-sm text-white ${
              isRunning
                ? "bg-slate-400 cursor-not-allowed"
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
        <aside className="w-80 border-r border-slate-200 bg-slate-50 p-4 overflow-auto flex flex-col gap-4">
          <DatasetManager
            datasets={datasets}
            selectedDatasetId={selectedDatasetId}
            onSelectDataset={setSelectedDatasetId}
            onRefresh={refreshDatasets}
          />

          <PredictPanel
            models={models}
            selectedModelId={selectedModelId}
            onSelectModel={setSelectedModelId}
            onDone={refreshRegistry}
          />
        </aside>

        <main className="flex-grow flex flex-col overflow-hidden relative">
          <div className="flex-grow relative">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              nodeTypes={nodeTypes}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              fitView
            >
              <Background color="#cbd5e1" gap={16} size={1} />
              <Controls className="bg-white border-slate-200 shadow-md" />
            </ReactFlow>

            <ResultsDashboard
              result={finalResult}
              onClose={() => setFinalResult(null)}
            />
          </div>

          {(logs.length > 0 || finalResult) && (
            <div className="h-64 bg-slate-900 border-t border-slate-700 flex flex-col flex-shrink-0 z-20">
              <div className="h-10 bg-slate-800 flex items-center justify-between px-4 border-b border-slate-700">
                <div className="flex items-center gap-2 text-slate-300">
                  <Terminal size={16} />
                  <span className="font-mono text-sm font-semibold tracking-wider">
                    EXECUTION LOGS
                  </span>
                </div>

                <div>
                  {taskStatus === "PROGRESS" && (
                    <span className="text-blue-400 text-xs font-mono animate-pulse flex items-center gap-2">
                      <Loader2 size={14} className="animate-spin" /> RUNNING
                    </span>
                  )}

                  {taskStatus === "SUCCESS" && (
                    <span className="text-green-400 text-xs font-mono flex items-center gap-1">
                      <CheckCircle2 size={14} /> SUCCESS
                    </span>
                  )}

                  {taskStatus === "FAILURE" && (
                    <span className="text-red-400 text-xs font-mono flex items-center gap-1">
                      <XCircle size={14} /> FAILED
                    </span>
                  )}
                </div>
              </div>

              <div className="p-4 overflow-y-auto font-mono text-xs flex-grow space-y-1">
                {logs.map((log, index) => (
                  <div key={index} className="text-slate-300">
                    <span className="text-slate-500 mr-3">[{log.time}]</span>
                    <span
                      className={
                        log.text.includes("❌")
                          ? "text-red-400"
                          : log.text.includes("✅")
                          ? "text-green-400"
                          : log.text.includes("📌") || log.text.includes("💾")
                          ? "text-blue-300"
                          : ""
                      }
                    >
                      {log.text}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </main>

        <aside className="w-80 border-l border-slate-200 bg-slate-50 p-4 overflow-auto">
          <RegistryPanel
            experiments={experiments}
            models={models}
            predictions={predictions}
            pipelines={pipelines}
            onRefresh={refreshRegistry}
            onSelectModel={setSelectedModelId}
          />
        </aside>
      </div>
    </div>
  );
}