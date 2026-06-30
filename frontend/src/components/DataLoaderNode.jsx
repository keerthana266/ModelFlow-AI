import React from "react";
import { Handle, Position } from "reactflow";
import { Database } from "lucide-react";

export default function DataLoaderNode({ data }) {
  return (
    <div className="bg-white border-2 border-blue-300 rounded-lg shadow-md w-72 overflow-hidden">
      <div className="bg-blue-50 px-4 py-2 border-b border-blue-100 flex items-center gap-2">
        <Database size={16} className="text-blue-600" />
        <span className="font-bold text-sm text-slate-800">Data Loader</span>
      </div>

      <div className="p-4 flex flex-col gap-3">
        <div>
          <label className="text-xs font-bold text-slate-500 uppercase tracking-wider">
            Dataset
          </label>

          <select
            className="nodrag w-full mt-1 px-2 py-1 border border-slate-300 rounded-md text-sm focus:outline-blue-500 bg-white"
            value={data.dataset_id || ""}
            onChange={(e) => data.onChange("dataset_id", e.target.value)}
          >
            <option value="">Select dataset</option>
            {(data.datasets || []).map((dataset) => (
              <option key={dataset.dataset_id} value={dataset.dataset_id}>
                {dataset.original_filename} ({dataset.rows} rows)
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="text-xs font-bold text-slate-500 uppercase tracking-wider">
            Target Column
          </label>

          <select
            className="nodrag w-full mt-1 px-2 py-1 border border-slate-300 rounded-md text-sm focus:outline-blue-500 bg-white"
            value={data.target_column || ""}
            onChange={(e) => data.onChange("target_column", e.target.value)}
            disabled={!data.dataset_id}
          >
            <option value="">Select target</option>
            {(data.columns || []).map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        </div>

        <div className="text-[11px] text-slate-400 break-all">
          {data.dataset_id ? `ID: ${data.dataset_id}` : "Upload/select dataset first"}
        </div>
      </div>

      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 bg-blue-500 border-2 border-white"
      />
    </div>
  );
}