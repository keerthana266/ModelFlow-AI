import React from "react";
import { Handle, Position } from "reactflow";
import { BrainCircuit } from "lucide-react";

export default function AutoMLNode({ data }) {
  return (
    <div className="bg-white border-2 border-purple-300 rounded-lg shadow-md w-72 overflow-hidden">
      <Handle
        type="target"
        position={Position.Top}
        className="w-3 h-3 bg-purple-500 border-2 border-white"
      />

      <div className="bg-purple-50 px-4 py-2 border-b border-purple-100 flex items-center gap-2">
        <BrainCircuit size={16} className="text-purple-600" />
        <span className="font-bold text-sm text-slate-800">AutoML Engine</span>
      </div>

      <div className="p-4 flex flex-col gap-3">
        <div>
          <label className="text-xs font-bold text-slate-500 uppercase tracking-wider">
            Time Budget Seconds
          </label>

          <input
            type="number"
            className="nodrag w-full mt-1 px-2 py-1 border border-slate-300 rounded-md text-sm focus:outline-purple-500"
            value={data.time_budget || 30}
            min={10}
            max={300}
            onChange={(e) =>
              data.onChange("time_budget", Number(e.target.value))
            }
          />
        </div>

        <p className="text-xs text-slate-400 italic">
          FLAML will optimize models for this duration.
        </p>
      </div>

      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 bg-purple-500 border-2 border-white"
      />
    </div>
  );
}