import React, { useState, useCallback, useRef, useEffect } from 'react';
import ReactFlow, {
  Background,
  Controls,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Play, Loader2, Terminal, CheckCircle2, XCircle } from 'lucide-react';
import axios from 'axios';

import DataLoaderNode from './components/DataLoaderNode';
import AutoMLNode from './components/AutoMLNode';
import ResultsDashboard from './components/ResultsDashboard';

const nodeTypes = {
  dataLoader: DataLoaderNode,
  automl: AutoMLNode,
};

const initialEdges = [
  { id: 'edge_1', source: 'node_1', target: 'node_2', animated: true },
];

export default function App() {
  const [isRunning, setIsRunning] = useState(false);
  const [edges, setEdges] = useState(initialEdges);
  
  // Terminal and WebSocket State
  const [logs, setLogs] = useState([]);
  const [taskStatus, setTaskStatus] = useState(null); // 'PROGRESS', 'SUCCESS', 'FAILURE'
  const [finalResult, setFinalResult] = useState(null);
  const wsRef = useRef(null);

  const [nodes, setNodes] = useState([
    {
      id: 'node_1',
      type: 'dataLoader',
      position: { x: 250, y: 100 },
      data: { 
        dataset_id: '',
        target_column: '',
        onChange: (key, value) => {
          setNodes((nds) => 
            nds.map((node) => 
              node.id === 'node_1' ? { ...node, data: { ...node.data, [key]: value } } : node
            )
          );
        }
      },
    },
    {
      id: 'node_2',
      type: 'automl',
      position: { x: 250, y: 350 },
      data: { 
        time_budget: 30,
        onChange: (key, value) => {
          setNodes((nds) => 
            nds.map((node) => 
              node.id === 'node_2' ? { ...node, data: { ...node.data, [key]: value } } : node
            )
          );
        }
      },
    },
  ]);

  const onNodesChange = useCallback((changes) => setNodes((nds) => applyNodeChanges(changes, nds)), []);
  const onEdgesChange = useCallback((changes) => setEdges((eds) => applyEdgeChanges(changes, eds)), []);
  const onConnect = useCallback((params) => setEdges((eds) => addEdge({ ...params, animated: true }, eds)), []);

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const connectWebSocket = (taskId) => {
    // Close existing connection if any
    if (wsRef.current) wsRef.current.close();

    const ws = new WebSocket(`ws://127.0.0.1:8000/ws/task/${taskId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setLogs((prev) => [...prev, { time: new Date().toLocaleTimeString(), text: "🟢 WebSocket Tunnel Connected." }]);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setTaskStatus(data.status);

      if (data.message) {
        setLogs((prev) => [...prev, { time: new Date().toLocaleTimeString(), text: `[${data.status}] ${data.message}` }]);
      }

      if (data.status === "SUCCESS") {
        setLogs((prev) => [...prev, { time: new Date().toLocaleTimeString(), text: "✅ Pipeline Execution Complete!" }]);
        setFinalResult(data.result);
        setIsRunning(false);
      } else if (data.status === "FAILURE") {
        setLogs((prev) => [...prev, { time: new Date().toLocaleTimeString(), text: `❌ ERROR: ${data.error}` }]);
        setIsRunning(false);
      }
    };

    ws.onclose = () => {
      setLogs((prev) => [...prev, { time: new Date().toLocaleTimeString(), text: "🔴 Tunnel Closed." }]);
    };
  };

  const handleRunPipeline = async () => {
    setIsRunning(true);
    setLogs([]); // Clear old logs
    setTaskStatus('STARTING');
    setFinalResult(null);

    try {
      const cleanNodes = nodes.map(node => {
        const { onChange, ...cleanData } = node.data;
        return { id: node.id, type: node.type, data: cleanData };
      });

      const payload = {
        nodes: cleanNodes,
        edges: edges.map(edge => ({ id: edge.id, source: edge.source, target: edge.target }))
      };

      setLogs([{ time: new Date().toLocaleTimeString(), text: "🚀 Transpiling graph and submitting to FastAPI..." }]);
      
      const response = await axios.post('http://127.0.0.1:8000/pipeline/execute', payload);
      const taskId = response.data.final_state.task_id;
      
      if (!taskId) throw new Error("Backend did not return a task_id.");

      // WAKE UP THE TUNNEL
      connectWebSocket(taskId);

    } catch (error) {
      setLogs([{ time: new Date().toLocaleTimeString(), text: `❌ Transpiler Error: ${error.response?.data?.detail || error.message}` }]);
      setIsRunning(false);
      setTaskStatus('FAILURE');
    }
  };

  return (
    <div className="h-screen w-screen flex flex-col bg-slate-50">
      {/* Top Navigation Bar */}
      <nav className="h-14 bg-white border-b border-slate-200 flex items-center justify-between px-6 shadow-sm z-10 flex-shrink-0">
        <h1 className="text-xl font-bold text-slate-800 tracking-tight">
          AutoML <span className="text-blue-600">Forge</span>
        </h1>
        <button 
          onClick={handleRunPipeline}
          disabled={isRunning}
          className={`flex items-center gap-2 px-4 py-2 rounded-md font-medium transition-colors text-sm text-white ${
            isRunning ? 'bg-slate-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
          }`}
        >
          {isRunning ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} fill="currentColor" />}
          {isRunning ? 'Executing...' : 'Run Pipeline'}
        </button>
      </nav>

      {/* Main Layout: Canvas + Terminal Split */}
      <div className="flex-grow flex flex-col overflow-hidden relative">
        
        {/* React Flow Canvas */}
        <div className="flex-grow relative">
          <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={onConnect} fitView>
            <Background color="#cbd5e1" gap={16} size={1} />
            <Controls className="bg-white border-slate-200 shadow-md" />
          </ReactFlow>

          <ResultsDashboard 
            result={finalResult} 
            onClose={() => setFinalResult(null)} 
          />
        </div>

        {/* Live Execution Terminal Overlay */}
        {(logs.length > 0 || finalResult) && (
          <div className="h-64 bg-slate-900 border-t border-slate-700 flex flex-col flex-shrink-0 z-20 shadow-[0_-10px_40px_-15px_rgba(0,0,0,0.3)]">
            <div className="h-10 bg-slate-800 flex items-center justify-between px-4 border-b border-slate-700">
              <div className="flex items-center gap-2 text-slate-300">
                <Terminal size={16} />
                <span className="font-mono text-sm font-semibold tracking-wider">EXECUTION LOGS</span>
              </div>
              <div>
                {taskStatus === 'PROGRESS' && <span className="text-blue-400 text-xs font-mono animate-pulse flex items-center gap-2"><Loader2 size={14} className="animate-spin"/> RUNNING</span>}
                {taskStatus === 'SUCCESS' && <span className="text-green-400 text-xs font-mono flex items-center gap-1"><CheckCircle2 size={14}/> SUCCESS</span>}
                {taskStatus === 'FAILURE' && <span className="text-red-400 text-xs font-mono flex items-center gap-1"><XCircle size={14}/> FAILED</span>}
              </div>
            </div>
            
            <div className="p-4 overflow-y-auto font-mono text-xs flex-grow space-y-1">
              {logs.map((log, index) => (
                <div key={index} className="text-slate-300">
                  <span className="text-slate-500 mr-3">[{log.time}]</span>
                  <span className={log.text.includes('❌') ? 'text-red-400' : log.text.includes('✅') ? 'text-green-400' : ''}>
                    {log.text}
                  </span>
                </div>
              ))}
              {/* Auto-scroll anchor */}
              <div style={{ float:"left", clear: "both" }} />
            </div>
          </div>
        )}

      </div>
    </div>
  );
}