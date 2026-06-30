import React from "react";
import { Handle, Position } from "reactflow";
import { BrainCircuit, Gauge } from "lucide-react";

export default function AutoMLNode({ data }) {
  return (
    <div className="w-80 overflow-hidden rounded-2xl border border-violet-200 bg-white shadow-[0_12px_35px_rgba(124,58,237,0.16)]">
      <Handle
        type="target"
        position={Position.Top}
        className="!h-3 !w-3 !border-2 !border-white !bg-violet-600"
      />

      <div className="flex items-center justify-between border-b border-violet-100 bg-gradient-to-r from-violet-50 to-fuchsia-50 px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="grid h-8 w-8 place-items-center rounded-xl bg-violet-600 text-white">
            <BrainCircuit size={16} />
          </div>

          <div>
            <div className="text-sm font-bold text-slate-900">
              AutoML Engine
            </div>
            <div className="text-[11px] text-slate-500">
              FLAML optimization node
            </div>
          </div>
        </div>

        <Gauge size={16} className="text-violet-500" />
      </div>

      <div className="space-y-4 p-4">
        <div>
          <label className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
            Time Budget
          </label>

          <div className="mt-1 flex items-center gap-2">
            <input
              type="number"
              className="nodrag w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 outline-none transition focus:border-violet-400 focus:bg-white focus:ring-4 focus:ring-violet-100"
              value={data.time_budget || 30}
              min={10}
              max={300}
              onChange={(e) =>
                data.onChange("time_budget", Number(e.target.value))
              }
            />

            <span className="text-xs font-medium text-slate-500">sec</span>
          </div>
        </div>

        <div className="rounded-xl bg-violet-50 px-3 py-2 text-[11px] leading-relaxed text-violet-700">
          AutoML will search multiple models and tune the best candidate within
          the selected budget.
        </div>
      </div>

      <Handle
        type="source"
        position={Position.Bottom}
        className="!h-3 !w-3 !border-2 !border-white !bg-violet-600"
      />
    </div>
  );
}