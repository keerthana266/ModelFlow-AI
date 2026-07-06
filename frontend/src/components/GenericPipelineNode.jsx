import React from "react";
import { Handle, Position } from "reactflow";
import {
  Activity,
  BarChart3,
  Bot,
  CheckCircle2,
  CircleDashed,
  Database,
  LineChart,
  Loader2,
  Rocket,
  Sparkles,
  Workflow,
  XCircle,
} from "lucide-react";

export default function GenericPipelineNode({ data }) {
  const status = data.status || "idle";
  const color = data.color || "slate";
  const theme = getStatusTheme(status, color);
  const Icon = iconMap[data.icon] || Workflow;

  return (
    <div
      className={`w-80 overflow-hidden rounded-3xl border-2 bg-white shadow-[0_18px_45px_rgba(15,23,42,0.12)] transition-all duration-300 ${theme.border} ${theme.glow}`}
    >
      <Handle
        type="target"
        position={Position.Top}
        className="!h-4 !w-4 !border-2 !border-white !bg-slate-500"
      />

      <div
        className={`flex items-center gap-3 border-b px-4 py-4 ${theme.header}`}
      >
        <div
          className={`grid h-10 w-10 place-items-center rounded-2xl ${theme.iconBox}`}
        >
          <Icon size={18} />
        </div>

        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-black text-slate-900">
            {data.label || "Pipeline Node"}
          </h3>
          <p className="text-xs text-slate-500">
            {data.description || "Visual pipeline step"}
          </p>
        </div>

        <StatusIcon status={status} />
      </div>

      <div className="space-y-3 p-4">
        <StatusPill status={status} message={data.statusMessage} />

        <div className="rounded-2xl bg-slate-50 p-3">
          <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
            Node Type
          </div>

          <div className="mt-1 font-mono text-xs text-slate-600">
            {data.nodeType || "visual"}
          </div>
        </div>
      </div>

      <Handle
        type="source"
        position={Position.Bottom}
        className="!h-4 !w-4 !border-2 !border-white !bg-slate-500"
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

const iconMap = {
  database: Database,
  workflow: Workflow,
  bot: Bot,
  "bar-chart": BarChart3,
  sparkles: Sparkles,
  "line-chart": LineChart,
  activity: Activity,
  rocket: Rocket,
};

function getStatusTheme(status, color) {
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

  const idleThemes = {
    blue: {
      border: "border-slate-200",
      glow: "",
      header: "border-slate-100 bg-white",
      iconBox: "bg-blue-50 text-blue-600",
    },
    cyan: {
      border: "border-slate-200",
      glow: "",
      header: "border-slate-100 bg-white",
      iconBox: "bg-cyan-50 text-cyan-600",
    },
    violet: {
      border: "border-slate-200",
      glow: "",
      header: "border-slate-100 bg-white",
      iconBox: "bg-violet-50 text-violet-600",
    },
    emerald: {
      border: "border-slate-200",
      glow: "",
      header: "border-slate-100 bg-white",
      iconBox: "bg-emerald-50 text-emerald-600",
    },
    amber: {
      border: "border-slate-200",
      glow: "",
      header: "border-slate-100 bg-white",
      iconBox: "bg-amber-50 text-amber-600",
    },
    indigo: {
      border: "border-slate-200",
      glow: "",
      header: "border-slate-100 bg-white",
      iconBox: "bg-indigo-50 text-indigo-600",
    },
    orange: {
      border: "border-slate-200",
      glow: "",
      header: "border-slate-100 bg-white",
      iconBox: "bg-orange-50 text-orange-600",
    },
    rose: {
      border: "border-slate-200",
      glow: "",
      header: "border-slate-100 bg-white",
      iconBox: "bg-rose-50 text-rose-600",
    },
  };

  return idleThemes[color] || idleThemes.blue;
}