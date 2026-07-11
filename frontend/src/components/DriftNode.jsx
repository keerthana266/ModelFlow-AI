import React from "react";
import { api } from "../api";
import { Handle, Position } from "reactflow";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  CircleDashed,
  ExternalLink,
  Loader2,
  Play,
  XCircle,
} from "lucide-react";

export default function DriftNode({ data }) {
  const status = data.status || "idle";
  const theme = getStatusTheme(status);
  const datasets = data.datasets || [];
  const report = data.lastReport || null;

  const driftShare =
    report?.drift_share === null || report?.drift_share === undefined
      ? null
      : Number(report.drift_share);

  return (
    <div
      className={`w-80 overflow-hidden rounded-3xl border-2 bg-white shadow-[0_18px_45px_rgba(15,23,42,0.12)] transition-all duration-300 ${theme.border} ${theme.glow}`}
    >
      <Handle
        type="target"
        position={Position.Top}
        className="!h-4 !w-4 !border-2 !border-white !bg-orange-500"
      />

      <div
        className={`flex items-center gap-3 border-b px-4 py-4 ${theme.header}`}
      >
        <div
          className={`grid h-10 w-10 place-items-center rounded-2xl ${theme.iconBox}`}
        >
          <Activity size={18} />
        </div>

        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-black text-slate-900">Drift Monitor</h3>
          <p className="text-xs text-slate-500">
            Evidently AI data drift report
          </p>
        </div>

        <StatusIcon status={status} />
      </div>

      <div className="space-y-4 p-4">
        <StatusPill status={status} message={data.statusMessage} />

        <div>
          <label className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
            Current Dataset
          </label>

          <select
            value={data.current_dataset_id || ""}
            onChange={(event) =>
              data.onChange?.("current_dataset_id", event.target.value)
            }
            className="nodrag mt-1 w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 outline-none transition focus:border-orange-400 focus:bg-white focus:ring-4 focus:ring-orange-100"
          >
            <option value="">Select dataset for comparison</option>

            {datasets.map((dataset) => (
              <option key={dataset.dataset_id} value={dataset.dataset_id}>
                {dataset.original_filename || dataset.dataset_id}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
            Maximum Rows
          </label>

          <input
            type="number"
            min="2"
            max="10000"
            value={data.max_rows || 1000}
            onChange={(event) =>
              data.onChange?.("max_rows", Number(event.target.value))
            }
            className="nodrag mt-1 w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 outline-none transition focus:border-orange-400 focus:bg-white focus:ring-4 focus:ring-orange-100"
          />
        </div>

        <label className="nodrag flex cursor-pointer items-center gap-2 rounded-2xl border border-slate-200 bg-slate-50 p-3 text-xs font-bold text-slate-600">
          <input
            type="checkbox"
            checked={data.auto_check !== false}
            onChange={(event) =>
              data.onChange?.("auto_check", event.target.checked)
            }
            className="h-4 w-4 accent-orange-600"
          />
          Auto-run after training
        </label>

        <button
          type="button"
          onClick={() => data.onRun?.()}
          disabled={
            status === "running" ||
            !data.current_dataset_id ||
            !data.canRun
          }
          className={`nodrag flex w-full items-center justify-center gap-2 rounded-2xl px-4 py-3 text-sm font-bold text-white transition ${
            status === "running" ||
            !data.current_dataset_id ||
            !data.canRun
              ? "cursor-not-allowed bg-slate-400"
              : "bg-orange-600 hover:bg-orange-700"
          }`}
        >
          {status === "running" ? (
            <Loader2 size={16} className="animate-spin" />
          ) : (
            <Play size={16} fill="currentColor" />
          )}
          {status === "running" ? "Checking Drift..." : "Run Drift Check"}
        </button>

        {report && (
          <div
            className={`rounded-2xl border p-3 ${
              report.drift_detected
                ? "border-amber-100 bg-amber-50"
                : "border-emerald-100 bg-emerald-50"
            }`}
          >
            <div
              className={`flex items-center gap-2 text-xs font-black ${
                report.drift_detected
                  ? "text-amber-800"
                  : "text-emerald-700"
              }`}
            >
              {report.drift_detected ? (
                <AlertTriangle size={14} />
              ) : (
                <CheckCircle2 size={14} />
              )}
              {report.drift_detected
                ? "Dataset drift detected"
                : "No significant drift detected"}
            </div>

            <div className="mt-2 space-y-1 text-[11px] text-slate-600">
              {report.report_id && (
                <div className="truncate font-mono">
                  Report: {report.report_id}
                </div>
              )}

              {driftShare !== null && !Number.isNaN(driftShare) && (
                <div>
                  Drift share:{" "}
                  <span className="font-black">
                    {(driftShare * 100).toFixed(2)}%
                  </span>
                </div>
              )}

              {report.number_of_drifted_columns !== undefined && (
                <div>
                  Drifted columns:{" "}
                  <span className="font-black">
                    {report.number_of_drifted_columns}
                  </span>
                </div>
              )}
            </div>

            {report.report_url && (
              <button
                type="button"
                onClick={() => openProtectedReport(report.report_url)}
                className="nodrag mt-3 flex w-full items-center justify-center gap-2 rounded-xl bg-white px-3 py-2 text-xs font-bold text-slate-700 shadow-sm transition hover:bg-slate-950 hover:text-white"
              >
                <ExternalLink size={13} />
                Open Evidently Report
              </button>
            )}
          </div>
        )}

        {!data.canRun && (
          <div className="rounded-2xl border border-slate-100 bg-slate-50 p-3 text-xs leading-relaxed text-slate-500">
            Train a model first, or select an existing model in the Model
            Registry, before running a manual drift check.
          </div>
        )}
      </div>

      <Handle
        type="source"
        position={Position.Bottom}
        className="!h-4 !w-4 !border-2 !border-white !bg-orange-500"
      />
    </div>
  );
}


async function openProtectedReport(reportUrl) {
  const reportWindow = window.open("", "_blank");

  try {
    const response = await api.get(reportUrl, {
      responseType: "blob",
    });

    const objectUrl = URL.createObjectURL(response.data);

    if (reportWindow) {
      reportWindow.location.href = objectUrl;
    } else {
      window.location.href = objectUrl;
    }

    window.setTimeout(() => URL.revokeObjectURL(objectUrl), 60_000);
  } catch (error) {
    reportWindow?.close();
    window.alert(
      error.response?.data?.detail ||
        error.message ||
        "Could not open the drift report."
    );
  }
}

function StatusIcon({ status }) {
  if (status === "running") {
    return <Loader2 size={18} className="animate-spin text-blue-600" />;
  }

  if (status === "success") {
    return <CheckCircle2 size={18} className="text-emerald-600" />;
  }

  if (status === "warning") {
    return <AlertTriangle size={18} className="text-amber-600" />;
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
      : status === "warning"
      ? "Drift Found"
      : status === "error"
      ? "Failed"
      : "Ready";

  const className =
    status === "running"
      ? "border-blue-100 bg-blue-50 text-blue-700"
      : status === "success"
      ? "border-emerald-100 bg-emerald-50 text-emerald-700"
      : status === "warning"
      ? "border-amber-100 bg-amber-50 text-amber-800"
      : status === "error"
      ? "border-red-100 bg-red-50 text-red-700"
      : "border-slate-100 bg-slate-50 text-slate-500";

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

  if (status === "warning") {
    return {
      border: "border-amber-400",
      glow: "shadow-amber-100",
      header: "border-amber-100 bg-amber-50",
      iconBox: "bg-amber-500 text-white",
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
    iconBox: "bg-orange-50 text-orange-600",
  };
}
