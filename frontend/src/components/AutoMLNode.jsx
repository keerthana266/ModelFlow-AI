// src/components/AutoMLNode.jsx
import React from 'react';
import { Handle, Position } from 'reactflow';
import { BrainCircuit } from 'lucide-react';

export default function AutoMLNode({ data }) {
  return (
    <div className="bg-white border-2 border-purple-300 rounded-lg shadow-md w-64 overflow-hidden">
      {/* Top connection point (Receives Data) */}
      <Handle 
        type="target" 
        position={Position.Top} 
        className="w-3 h-3 bg-purple-500 border-2 border-white" 
      />

      {/* Node Header */}
      <div className="bg-purple-50 px-4 py-2 border-b border-purple-100 flex items-center gap-2">
        <BrainCircuit size={16} className="text-purple-600" />
        <span className="font-bold text-sm text-slate-800">AutoML Engine</span>
      </div>
      
      {/* Node Body with Inputs */}
      <div className="p-4 flex flex-col gap-3">
        <div>
          <label className="text-xs font-bold text-slate-500 uppercase tracking-wider">
            Time Budget (Seconds)
          </label>
          <input 
            type="number" 
            className="nodrag w-full mt-1 px-2 py-1 border border-slate-300 rounded-md text-sm focus:outline-purple-500" 
            defaultValue={30}
            min={10}
            max={300}
            onChange={(e) => data.onChange('time_budget', parseInt(e.target.value))}
          />
        </div>
        <p className="text-xs text-slate-400 mt-1 italic">
          FLAML will optimize models for this duration.
        </p>
      </div>

      {/* Bottom connection point (Outputs Model) */}
      <Handle 
        type="source" 
        position={Position.Bottom} 
        className="w-3 h-3 bg-purple-500 border-2 border-white" 
      />
    </div>
  );
}