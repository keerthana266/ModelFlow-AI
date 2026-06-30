import React from "react";
import { Activity, Brain, Clock3, DatabaseZap, RefreshCcw } from "lucide-react";

export default function RegistryPanel({
  experiments,
  models,
  predictions,
  pipelines,
  onRefresh,
  onSelectModel,
}) {
  return (
    <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-100 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <DatabaseZap size={17} className="text-indigo-600" />
          <h2 className="font-bold text-sm text-slate-800">Backend Registry</h2>
        </div>

        <button
          onClick={onRefresh}
          className="text-xs flex items-center gap-1 text-slate-500 hover:text-indigo-600"
        >
          <RefreshCcw size={13} /> Refresh
        </button>
      </div>

      <div className="p-4 flex flex-col gap-5 max-h-[calc(100vh-180px)] overflow-auto">
        <section>
          <h3 className="text-xs font-bold text-slate-400 uppercase mb-2 flex items-center gap-1">
            <Activity size={13} /> Experiments
          </h3>

          {experiments.length === 0 ? (
            <Empty />
          ) : (
            experiments.slice(0, 5).map((exp) => (
              <div
                key={exp.experiment_id}
                className="border border-slate-200 rounded-lg p-2 mb-2 text-xs"
              >
                <div className="font-semibold text-slate-700">
                  {exp.problem_type} · {exp.target_column}
                </div>
                <div className="text-slate-500">
                  {exp.main_metric}: {exp.best_score}
                </div>
                <div className="text-[10px] text-slate-400 truncate">
                  {exp.experiment_id}
                </div>
              </div>
            ))
          )}
        </section>

        <section>
          <h3 className="text-xs font-bold text-slate-400 uppercase mb-2 flex items-center gap-1">
            <Brain size={13} /> Models
          </h3>

          {models.length === 0 ? (
            <Empty />
          ) : (
            models.slice(0, 5).map((model) => (
              <button
                key={model.model_id}
                onClick={() => onSelectModel(model.model_id)}
                className="w-full text-left border border-slate-200 rounded-lg p-2 mb-2 text-xs hover:bg-indigo-50 hover:border-indigo-300"
              >
                <div className="font-semibold text-slate-700">
                  {model.model_name}
                </div>
                <div className="text-slate-500">
                  {model.main_metric}: {model.score}
                </div>
                <div className="text-[10px] text-slate-400 truncate">
                  {model.model_id}
                </div>
              </button>
            ))
          )}
        </section>

        <section>
          <h3 className="text-xs font-bold text-slate-400 uppercase mb-2 flex items-center gap-1">
            <Clock3 size={13} /> Prediction Logs
          </h3>

          {predictions.length === 0 ? (
            <Empty />
          ) : (
            predictions.slice(0, 5).map((log) => (
              <div
                key={log.prediction_id}
                className="border border-slate-200 rounded-lg p-2 mb-2 text-xs"
              >
                <div className="font-semibold text-slate-700 truncate">
                  {log.model_id}
                </div>
                <div className="text-slate-500 truncate">
                  Prediction: {JSON.stringify(log.prediction)}
                </div>
              </div>
            ))
          )}
        </section>

        <section>
          <h3 className="text-xs font-bold text-slate-400 uppercase mb-2">
            Saved Pipelines
          </h3>

          {pipelines.length === 0 ? (
            <Empty />
          ) : (
            pipelines.slice(0, 5).map((pipeline) => (
              <div
                key={pipeline.graph_id}
                className="border border-slate-200 rounded-lg p-2 mb-2 text-xs"
              >
                <div className="font-semibold text-slate-700">
                  {pipeline.nodes?.length || 0} nodes ·{" "}
                  {pipeline.edges?.length || 0} edges
                </div>
                <div className="text-[10px] text-slate-400 truncate">
                  {pipeline.graph_id}
                </div>
              </div>
            ))
          )}
        </section>
      </div>
    </div>
  );
}

function Empty() {
  return (
    <div className="text-xs text-slate-400 border border-dashed border-slate-200 rounded-lg p-3 text-center">
      No data yet.
    </div>
  );
}