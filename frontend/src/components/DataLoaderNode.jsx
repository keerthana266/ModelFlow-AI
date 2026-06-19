// src/components/DataLoaderNode.jsx
import React from 'react';
import { Handle, Position } from 'reactflow';
import { Database } from 'lucide-react';

export default function DataLoaderNode({ data }) {
  return (
    <div className="bg-white border-2 border-blue-300 rounded-lg shadow-md w-64 overflow-hidden">
      {/* Node Header */}
      <div className="bg-blue-50 px-4 py-2 border-b border-blue-100 flex items-center gap-2">
        <Database size={16} className="text-blue-600" />
        <span className="font-bold text-sm text-slate-800">Data Loader</span>
      </div>
      
      {/* Node Body with Inputs */}
      <div className="p-4 flex flex-col gap-3">
        <div>
          <label className="text-xs font-bold text-slate-500 uppercase tracking-wider">Dataset ID</label>
          <input 
            type="text" 
            className="nodrag w-full mt-1 px-2 py-1 border border-slate-300 rounded-md text-sm focus:outline-blue-500" 
            placeholder="dataset_abc123" 
            onChange={(e) => data.onChange('dataset_id', e.target.value)}
          />
        </div>
        <div>
          <label className="text-xs font-bold text-slate-500 uppercase tracking-wider">Target Column</label>
          <input 
            type="text" 
            className="nodrag w-full mt-1 px-2 py-1 border border-slate-300 rounded-md text-sm focus:outline-blue-500" 
            placeholder="e.g., Survived" 
            onChange={(e) => data.onChange('target_column', e.target.value)}
          />
        </div>
      </div>

      {/* The bottom connection point */}
      <Handle 
        type="source" 
        position={Position.Bottom} 
        className="w-3 h-3 bg-blue-500 border-2 border-white" 
      />
    </div>
  );
}