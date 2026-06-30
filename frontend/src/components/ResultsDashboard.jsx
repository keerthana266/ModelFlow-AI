import React from "react";
import { X, Trophy, BarChart3, Activity } from "lucide-react";

export default function ResultsDashboard({ result, onClose }) {
  if (!result) return null;

  const bestModel = result.leaderboard?.[0];
  const features = result.feature_importance || {};

  const maxShap = Math.max(
    ...Object.values(features).map((v) => (typeof v === "number" ? v : 0)),
    0.01
  );

  return (
    <div className="absolute top-4 right-4 w-[420px] bg-white border border-slate-200 rounded-xl shadow-2xl z-50 flex flex-col overflow-hidden">
      <div className="bg-gradient-to-r from-blue-600 to-indigo-700 p-4 flex justify-between items-center text-white">
        <div className="flex items-center gap-2">
          <Trophy size={18} className="text-yellow-300" />
          <h2 className="font-bold text-sm tracking-wide">PIPELINE SUCCESS</h2>
        </div>

        <button
          onClick={onClose}
          className="text-blue-100 hover:text-white transition-colors"
        >
          <X size={18} />
        </button>
      </div>

      <div className="p-5 overflow-y-auto max-h-[72vh] flex flex-col gap-6">
        <div>
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
            <Activity size={14} /> Winning Model
          </h3>

          <div className="bg-slate-50 border border-slate-100 rounded-lg p-3">
            <div className="font-semibold text-slate-800 text-lg mb-1">
              {bestModel?.model_name || "Unknown model"}
            </div>

            <div className="flex justify-between items-end">
              <span className="text-xs text-slate-500 uppercase">
                {result.main_metric?.replace("_", " ")}
              </span>

              <span className="text-xl font-bold text-blue-600">
                {typeof bestModel?.score === "number"
                  ? `${(bestModel.score * 100).toFixed(2)}%`
                  : "N/A"}
              </span>
            </div>
          </div>
        </div>

        <div>
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
            <BarChart3 size={14} /> Feature Importance SHAP
          </h3>

          {features.error ? (
            <div className="text-xs text-red-500 bg-red-50 p-2 rounded border border-red-100">
              {features.error}
            </div>
          ) : (
            <div className="flex flex-col gap-2">
              {Object.entries(features)
                .slice(0, 8)
                .map(([feature, value]) => {
                  const percentage = (value / maxShap) * 100;

                  return (
                    <div key={feature} className="flex flex-col gap-1">
                      <div className="flex justify-between text-xs">
                        <span className="font-medium text-slate-700 truncate">
                          {feature}
                        </span>
                        <span className="text-slate-500 font-mono">
                          {Number(value).toFixed(4)}
                        </span>
                      </div>

                      <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-indigo-500 rounded-full"
                          style={{ width: `${percentage}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
            </div>
          )}
        </div>

        <div className="bg-slate-950 text-slate-200 p-3 rounded-lg text-xs font-mono overflow-auto">
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      </div>
    </div>
  );
}