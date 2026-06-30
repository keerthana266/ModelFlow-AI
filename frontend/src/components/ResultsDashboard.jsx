import React from "react";
import { X, Trophy, BarChart3, Activity, Cpu, Target } from "lucide-react";

export default function ResultsDashboard({ result, onClose }) {
  if (!result) return null;

  const bestModel = result.leaderboard?.[0];
  const features = result.feature_importance || {};

  const maxShap = Math.max(
    ...Object.values(features).map((v) => (typeof v === "number" ? v : 0)),
    0.01
  );

  return (
    <div className="absolute right-6 top-6 z-50 w-[430px] overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-[0_30px_80px_rgba(15,23,42,0.22)]">
      <div className="flex items-center justify-between bg-slate-950 px-5 py-4 text-white">
        <div className="flex items-center gap-3">
          <div className="grid h-9 w-9 place-items-center rounded-2xl bg-amber-400 text-slate-950">
            <Trophy size={18} />
          </div>

          <div>
            <h2 className="text-sm font-bold tracking-wide">
              Pipeline Completed
            </h2>
            <p className="text-xs text-slate-400">
              Best model and explainability summary
            </p>
          </div>
        </div>

        <button
          onClick={onClose}
          className="rounded-full p-1 text-slate-400 transition hover:bg-white/10 hover:text-white"
        >
          <X size={18} />
        </button>
      </div>

      <div className="soft-scroll max-h-[76vh] overflow-y-auto p-5">
        <div className="grid grid-cols-2 gap-3">
          <MetricCard
            icon={<Cpu size={16} />}
            label="Best Model"
            value={bestModel?.model_name || "Unknown"}
          />

          <MetricCard
            icon={<Target size={16} />}
            label={result.main_metric?.replace("_", " ") || "Score"}
            value={
              typeof bestModel?.score === "number"
                ? `${(bestModel.score * 100).toFixed(2)}%`
                : "N/A"
            }
            highlight
          />
        </div>

        <div className="mt-5 rounded-2xl border border-slate-200 bg-slate-50 p-4">
          <h3 className="mb-4 flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-500">
            <Activity size={14} /> Training Summary
          </h3>

          <div className="grid grid-cols-2 gap-3 text-sm">
            <Info label="Problem" value={result.problem_type} />
            <Info label="Target" value={result.target_column} />
            <Info label="Rows Used" value={result.rows_used} />
            <Info label="Time Budget" value={`${result.time_budget || 30}s`} />
          </div>
        </div>

        <div className="mt-5 rounded-2xl border border-slate-200 bg-white p-4">
          <h3 className="mb-4 flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-500">
            <BarChart3 size={14} /> Feature Importance
          </h3>

          {features.error ? (
            <div className="rounded-xl border border-red-100 bg-red-50 p-3 text-xs text-red-600">
              {features.error}
            </div>
          ) : (
            <div className="space-y-3">
              {Object.entries(features)
                .slice(0, 8)
                .map(([feature, value]) => {
                  const percentage = (value / maxShap) * 100;

                  return (
                    <div key={feature}>
                      <div className="mb-1 flex justify-between gap-3 text-xs">
                        <span className="truncate font-semibold text-slate-700">
                          {feature}
                        </span>
                        <span className="font-mono text-slate-500">
                          {Number(value).toFixed(4)}
                        </span>
                      </div>

                      <div className="h-2.5 overflow-hidden rounded-full bg-slate-100">
                        <div
                          className="h-full rounded-full bg-gradient-to-r from-blue-500 to-violet-600"
                          style={{ width: `${percentage}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
            </div>
          )}
        </div>

        <details className="mt-5 rounded-2xl border border-slate-200 bg-slate-950">
          <summary className="cursor-pointer px-4 py-3 text-xs font-semibold uppercase tracking-wider text-slate-300">
            Raw Result JSON
          </summary>

          <pre className="soft-scroll max-h-56 overflow-auto border-t border-slate-800 p-4 text-[11px] text-slate-300">
            {JSON.stringify(result, null, 2)}
          </pre>
        </details>
      </div>
    </div>
  );
}

function MetricCard({ icon, label, value, highlight }) {
  return (
    <div
      className={`rounded-2xl border p-4 ${
        highlight
          ? "border-blue-200 bg-blue-50"
          : "border-slate-200 bg-slate-50"
      }`}
    >
      <div
        className={`mb-2 flex items-center gap-2 text-xs font-bold uppercase tracking-wider ${
          highlight ? "text-blue-600" : "text-slate-500"
        }`}
      >
        {icon}
        {label}
      </div>

      <div
        className={`line-clamp-2 text-lg font-bold ${
          highlight ? "text-blue-700" : "text-slate-900"
        }`}
      >
        {value}
      </div>
    </div>
  );
}

function Info({ label, value }) {
  return (
    <div>
      <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
        {label}
      </div>
      <div className="truncate text-sm font-semibold text-slate-800">
        {value ?? "N/A"}
      </div>
    </div>
  );
}