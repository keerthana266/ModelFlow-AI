import React from "react";
import { Handle, Position } from "reactflow";
import {
  CheckCircle2,
  CircleDashed,
  Globe2,
  Loader2,
  Rocket,
  XCircle,
} from "lucide-react";

export default function DeployNode({ data }) {
  const status = data.status || "idle";
  const theme = getStatusTheme(status);

  return (
    <div
      className={`w-80 overflow-hidden rounded-3xl border-2 bg-white shadow-[0_18px_45px_rgba(15,23,42,0.12)] transition-all duration-300 ${theme.border} ${theme.glow}`}
    >
      <Handle
        type="target"
        position={Position.Top}
        className="!h-4 !w-4 !border-2 !border-white !bg-rose-500"
      />

      <div
        className={`flex items-center gap-3 border-b px-4 py-4 ${theme.header}`}
      >
        <div
          className={`grid h-10 w-10 place-items-center rounded-2xl ${theme.iconBox}`}
        >
          <Rocket size={18} />
        </div>

        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-black text-slate-900">Deploy API</h3>
          <p className="text-xs text-slate-500">
            Auto-create versioned REST endpoint
          </p>
        </div>

        <StatusIcon status={status} />
      </div>

      <div className="space-y-4 p-4">
        <StatusPill status={status} message={data.statusMessage} />

        <div>
          <label className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
            Deployment Name
          </label>

          <input
            value={data.deployment_name || ""}
            onChange={(event) =>
              data.onChange?.("deployment_name", event.target.value)
            }
            placeholder="churn-api"
            className="nodrag mt-1 w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 outline-none transition focus:border-rose-400 focus:bg-white focus:ring-4 focus:ring-rose-100"
          />
        </div>

        <div>
          <label className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
            Environment
          </label>

          <select
            value={data.environment || "production"}
            onChange={(event) =>
              data.onChange?.("environment", event.target.value)
            }
            className="nodrag mt-1 w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 outline-none transition focus:border-rose-400 focus:bg-white focus:ring-4 focus:ring-rose-100"
          >
            <option value="development">Development</option>
            <option value="staging">Staging</option>
            <option value="production">Production</option>
          </select>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <label className="nodrag flex cursor-pointer items-center gap-2 rounded-2xl border border-slate-200 bg-slate-50 p-3 text-xs font-bold text-slate-600">
            <input
              type="checkbox"
              checked={data.auto_deploy !== false}
              onChange={(event) =>
                data.onChange?.("auto_deploy", event.target.checked)
              }
              className="h-4 w-4 accent-rose-600"
            />
            Auto Deploy
          </label>

          <label className="nodrag flex cursor-pointer items-center gap-2 rounded-2xl border border-slate-200 bg-slate-50 p-3 text-xs font-bold text-slate-600">
            <input
              type="checkbox"
              checked={data.activate !== false}
              onChange={(event) =>
                data.onChange?.("activate", event.target.checked)
              }
              className="h-4 w-4 accent-rose-600"
            />
            Activate
          </label>
        </div>

        {data.lastDeployment && (
          <div className="rounded-2xl border border-emerald-100 bg-emerald-50 p-3">
            <div className="flex items-center gap-2 text-xs font-black text-emerald-700">
              <Globe2 size={14} />
              Deployment Created
            </div>

            <div className="mt-2 space-y-1 font-mono text-[11px] text-emerald-800">
              <div>ID: {data.lastDeployment.deployment_id}</div>
              <div>Version: v{data.lastDeployment.version}</div>
              <div>{data.lastDeployment.endpoint_path}</div>
            </div>
          </div>
        )}

        <div className="rounded-2xl bg-slate-50 p-3">
          <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
            What happens here?
          </div>
          <p className="mt-1 text-xs leading-relaxed text-slate-600">
            After training succeeds, this node calls the backend deployment API
            and creates a versioned REST prediction endpoint.
          </p>
        </div>
      </div>

      <Handle
        type="source"
        position={Position.Bottom}
        className="!h-4 !w-4 !border-2 !border-white !bg-rose-500"
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
      ? "Deploying"
      : status === "success"
      ? "Deployed"
      : status === "error"
      ? "Failed"
      : "Ready";

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
    iconBox: "bg-rose-50 text-rose-600",
  };
}