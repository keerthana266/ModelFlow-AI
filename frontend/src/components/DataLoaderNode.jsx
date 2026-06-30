import React from "react";
import { Handle, Position } from "reactflow";
import { Database, Table2 } from "lucide-react";

export default function DataLoaderNode({ data }) {
  return (
    <div className="w-80 overflow-hidden rounded-2xl border border-blue-200 bg-white shadow-[0_12px_35px_rgba(37,99,235,0.14)]">
      <div className="flex items-center justify-between border-b border-blue-100 bg-gradient-to-r from-blue-50 to-sky-50 px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="grid h-8 w-8 place-items-center rounded-xl bg-blue-600 text-white">
            <Database size={16} />
          </div>

          <div>
            <div className="text-sm font-bold text-slate-900">Data Loader</div>
            <div className="text-[11px] text-slate-500">CSV source node</div>
          </div>
        </div>

        <Table2 size={16} className="text-blue-500" />
      </div>

      <div className="space-y-4 p-4">
        <div>
          <label className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
            Dataset
          </label>

          <select
            className="nodrag mt-1 w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 outline-none transition focus:border-blue-400 focus:bg-white focus:ring-4 focus:ring-blue-100"
            value={data.dataset_id || ""}
            onChange={(e) => data.onChange("dataset_id", e.target.value)}
          >
            <option value="">Select dataset</option>
            {(data.datasets || []).map((dataset) => (
              <option key={dataset.dataset_id} value={dataset.dataset_id}>
                {dataset.original_filename} · {dataset.rows} rows
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
            Prediction Target
          </label>

          <select
            className="nodrag mt-1 w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 outline-none transition focus:border-blue-400 focus:bg-white focus:ring-4 focus:ring-blue-100 disabled:cursor-not-allowed disabled:opacity-60"
            value={data.target_column || ""}
            onChange={(e) => data.onChange("target_column", e.target.value)}
            disabled={!data.dataset_id}
          >
            <option value="">Select target column</option>
            {(data.columns || []).map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        </div>

        <div className="rounded-xl bg-slate-50 px-3 py-2 text-[11px] text-slate-500">
          {data.dataset_id ? (
            <span className="break-all">Dataset ID: {data.dataset_id}</span>
          ) : (
            "Upload a dataset from the left panel first."
          )}
        </div>
      </div>

      <Handle
        type="source"
        position={Position.Bottom}
        className="!h-3 !w-3 !border-2 !border-white !bg-blue-600"
      />
    </div>
  );
}