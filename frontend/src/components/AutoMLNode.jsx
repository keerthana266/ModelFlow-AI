import React from "react";
import { Handle, Position } from "reactflow";
import {
  Bot,
  CheckCircle2,
  CircleDashed,
  Loader2,
  SlidersHorizontal,
  XCircle,
} from "lucide-react";

export default function AutoMLNode({ data }) {
  const status = data.status || "idle";
  const theme = getStatusTheme(status);

  return (
    <div
      className={`w-80 overflow-hidden rounded-3xl border-2 bg-white shadow-[0_18px_45px_rgba(15,23,42,0.12)] transition-all duration-300 ${theme.border} ${theme.glow}`}
    >
      <Handle
        type="target"
        position={Position.Top}
        className="!h-4 !w-4 !border-2 !border-white !bg-violet-500"
      />

      <div
        className={`flex items-center gap-3 border-b px-4 py-4 ${theme.header}`}
      >
        <div
          className={`grid h-10 w-10 place-items-center rounded-2xl ${theme.iconBox}`}
        >
          <Bot size={18} />
        </div>

        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-black text-slate-900">AutoML Engine</h3>
          <p className="text-xs text-slate-500">FLAML / Optuna optimizer</p>
        </div>

        <StatusIcon status={status} />
      </div>

      <div className="space-y-4 p-4">
        <StatusPill status={status} message={data.statusMessage} />

        <div>
          <label className="flex items-center gap-2 text-[11px] font-bold uppercase tracking-wider text-slate-500">
            <SlidersHorizontal size={13} />
            Training Mode
          </label>

          <select
            value={data.tuning_mode || "flaml"}
            onChange={(e) => data.onChange?.("tuning_mode", e.target.value)}
            className="nodrag mt-1 w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 outline-none transition focus:border-violet-400 focus:bg-white focus:ring-4 focus:ring-violet-100"
          >
            <option value="flaml">FLAML Fast AutoML</option>
            <option value="optuna">Optuna Advanced Tuning</option>
          </select>
        </div>

        <div>
          <label className="flex items-center gap-2 text-[11px] font-bold uppercase tracking-wider text-slate-500">
            <SlidersHorizontal size={13} />
            Time Budget
          </label>

          <div className="mt-1 flex items-center gap-2">
            <input
              type="number"
              min="10"
              max="300"
              value={data.time_budget || 30}
              onChange={(e) =>
                data.onChange?.("time_budget", Number(e.target.value))
              }
              className="nodrag w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 outline-none transition focus:border-violet-400 focus:bg-white focus:ring-4 focus:ring-violet-100"
            />

            <span className="rounded-xl bg-slate-100 px-3 py-2 text-xs font-bold text-slate-500">
              sec
            </span>
          </div>
        </div>

        <div className="rounded-2xl bg-slate-50 p-3">
          <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
            What happens here?
          </div>
          <p className="mt-1 text-xs leading-relaxed text-slate-600">
            FLAML gives fast AutoML search. Optuna performs advanced
            hyperparameter tuning across selected sklearn models.
          </p>
        </div>
      </div>

      <Handle
        type="source"
        position={Position.Bottom}
        className="!h-4 !w-4 !border-2 !border-white !bg-violet-500"
      />
    </div>
  );
}

function StatusIcon({ status }) {
  if (status === "running") {
    return <Loader2 size={18} className="animate-spin text-blue-600" />;
  }

  if (status === "success") {
    return <CheckCircle2 size={18} className="text-emerald-600" />;
  }

  if (status === "error") {
    return <XCircle size={18} className="text-red-600" />;
  }

  return <CircleDashed size={18} className="text-slate-400" />;
}

function StatusPill({ status, message }) {
  const label =
    status === "running"
      ? "Running"
      : status === "success"
      ? "Completed"
      : status === "error"
      ? "Failed"
      : "Idle";

  const className =
    status === "running"
      ? "bg-blue-50 text-blue-700 border-blue-100"
      : status === "success"
      ? "bg-emerald-50 text-emerald-700 border-emerald-100"
      : status === "error"
      ? "bg-red-50 text-red-700 border-red-100"
      : "bg-slate-50 text-slate-500 border-slate-100";

  return (
    <div
      className={`rounded-2xl border px-3 py-2 text-xs font-bold ${className}`}
    >
      {label}
      {message && (
        <span className="ml-1 font-medium opacity-80">· {message}</span>
      )}
    </div>
  );
}

function getStatusTheme(status) {
  if (status === "running") {
    return {
      border: "border-blue-400",
      glow: "shadow-blue-100",
      header: "border-blue-100 bg-blue-50",
      iconBox: "bg-blue-600 text-white",
    };
  }

  if (status === "success") {
    return {
      border: "border-emerald-400",
      glow: "shadow-emerald-100",
      header: "border-emerald-100 bg-emerald-50",
      iconBox: "bg-emerald-600 text-white",
    };
  }

  if (status === "error") {
    return {
      border: "border-red-400",
      glow: "shadow-red-100",
      header: "border-red-100 bg-red-50",
      iconBox: "bg-red-600 text-white",
    };
  }

  return {
    border: "border-slate-200",
    glow: "",
    header: "border-slate-100 bg-white",
    iconBox: "bg-violet-50 text-violet-600",
  };
}