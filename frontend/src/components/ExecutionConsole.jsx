import React from "react";
import {
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Loader2,
  Terminal,
  XCircle,
} from "lucide-react";

export default function ExecutionConsole({
  logs,
  status,
  isCollapsed,
  onToggle,
}) {
  if (logs.length === 0) return null;

  return (
    <div
      className={`z-20 flex-shrink-0 border-t border-slate-800 bg-slate-950 transition-all ${
        isCollapsed ? "h-12" : "h-64"
      }`}
    >
      <div className="flex h-12 items-center justify-between border-b border-slate-800 bg-slate-900 px-5">
        <div className="flex items-center gap-2 text-slate-300">
          <Terminal size={16} />
          <span className="font-mono text-xs font-bold uppercase tracking-wider">
            Execution Console
          </span>
          <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[11px] text-slate-400">
            {logs.length} logs
          </span>
        </div>

        <div className="flex items-center gap-3">
          <StatusBadge status={status} />

          <button
            onClick={onToggle}
            className="rounded-lg p-1 text-slate-400 hover:bg-slate-800 hover:text-white"
          >
            {isCollapsed ? <ChevronUp size={17} /> : <ChevronDown size={17} />}
          </button>
        </div>
      </div>

      {!isCollapsed && (
        <div className="soft-scroll h-[calc(100%-48px)] overflow-y-auto p-4 font-mono text-xs">
          {logs.map((log, index) => (
            <div key={index} className="mb-1.5 flex gap-3">
              <span className="w-24 flex-shrink-0 text-slate-600">
                [{log.time}]
              </span>

              <span className={logClass(log.text)}>{log.text}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function StatusBadge({ status }) {
  if (status === "PROGRESS") {
    return (
      <span className="flex items-center gap-2 rounded-full bg-blue-500/10 px-3 py-1 text-xs font-bold text-blue-300">
        <Loader2 size={13} className="animate-spin" />
        RUNNING
      </span>
    );
  }

  if (status === "SUCCESS") {
    return (
      <span className="flex items-center gap-2 rounded-full bg-emerald-500/10 px-3 py-1 text-xs font-bold text-emerald-300">
        <CheckCircle2 size={13} />
        SUCCESS
      </span>
    );
  }

  if (status === "FAILURE") {
    return (
      <span className="flex items-center gap-2 rounded-full bg-red-500/10 px-3 py-1 text-xs font-bold text-red-300">
        <XCircle size={13} />
        FAILED
      </span>
    );
  }

  return (
    <span className="rounded-full bg-slate-700 px-3 py-1 text-xs font-bold text-slate-300">
      IDLE
    </span>
  );
}

function logClass(text) {
  const value = String(text).toLowerCase();

  if (value.includes("error") || value.includes("failed")) {
    return "text-red-300";
  }

  if (
    value.includes("complete") ||
    value.includes("accepted") ||
    value.includes("found") ||
    value.includes("saved")
  ) {
    return "text-emerald-300";
  }

  if (
    value.includes("progress") ||
    value.includes("automl") ||
    value.includes("celery") ||
    value.includes("websocket")
  ) {
    return "text-blue-300";
  }

  return "text-slate-300";
}