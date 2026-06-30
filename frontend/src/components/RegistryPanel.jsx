import React, { useState } from "react";
import {
  Activity,
  Brain,
  Clock3,
  DatabaseZap,
  RefreshCcw,
  Workflow,
} from "lucide-react";

const tabs = [
  { id: "experiments", label: "Experiments", icon: Activity },
  { id: "models", label: "Models", icon: Brain },
  { id: "predictions", label: "Predictions", icon: Clock3 },
  { id: "pipelines", label: "Pipelines", icon: Workflow },
];

export default function RegistryPanel({
  experiments,
  models,
  predictions,
  pipelines,
  onRefresh,
  onSelectModel,
  onOpenModel,
  onOpenExperiment,
}) {
  const [activeTab, setActiveTab] = useState("experiments");

  const counts = {
    experiments: experiments.length,
    models: models.length,
    predictions: predictions.length,
    pipelines: pipelines.length,
  };

  return (
    <section className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
      <div className="flex items-center justify-between border-b border-slate-100 px-4 py-4">
        <div className="flex items-center gap-2">
          <div className="grid h-9 w-9 place-items-center rounded-2xl bg-indigo-50 text-indigo-600">
            <DatabaseZap size={17} />
          </div>

          <div>
            <h2 className="text-sm font-bold text-slate-900">
              Platform Registry
            </h2>
            <p className="text-xs text-slate-500">
              Browse platform artifacts
            </p>
          </div>
        </div>

        <button
          onClick={onRefresh}
          className="rounded-xl p-2 text-slate-400 transition hover:bg-slate-100 hover:text-indigo-600"
        >
          <RefreshCcw size={15} />
        </button>
      </div>

      <div className="grid grid-cols-2 gap-2 border-b border-slate-100 p-3">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const active = activeTab === tab.id;

          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`rounded-2xl px-3 py-2 text-left transition ${
                active
                  ? "bg-slate-950 text-white"
                  : "bg-slate-50 text-slate-600 hover:bg-slate-100"
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="flex items-center gap-2 text-xs font-bold">
                  <Icon size={13} />
                  {tab.label}
                </span>
                <span
                  className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${
                    active ? "bg-white/15 text-white" : "bg-white text-slate-500"
                  }`}
                >
                  {counts[tab.id]}
                </span>
              </div>
            </button>
          );
        })}
      </div>

      <div className="soft-scroll max-h-[calc(100vh-245px)] overflow-auto p-4">
        {activeTab === "experiments" && (
          <div className="space-y-3">
            {experiments.length === 0 ? (
              <Empty />
            ) : (
              experiments.map((exp) => (
                <button
                  key={exp.experiment_id}
                  onClick={() => onOpenExperiment(exp.experiment_id)}
                  className="w-full rounded-2xl border border-slate-200 bg-slate-50 p-4 text-left transition hover:border-blue-300 hover:bg-blue-50"
                >
                  <div className="text-sm font-black capitalize text-slate-900">
                    {exp.problem_type} · {exp.target_column}
                  </div>
                  <div className="mt-1 text-xs text-slate-500">
                    {exp.main_metric}:{" "}
                    <span className="font-bold text-slate-800">
                      {exp.best_score}
                    </span>
                  </div>
                  <div className="mt-2 truncate text-[10px] text-slate-400">
                    {exp.experiment_id}
                  </div>
                </button>
              ))
            )}
          </div>
        )}

        {activeTab === "models" && (
          <div className="space-y-3">
            {models.length === 0 ? (
              <Empty />
            ) : (
              models.map((model) => (
                <div
                  key={model.model_id}
                  className="rounded-2xl border border-slate-200 bg-white p-4 transition hover:border-indigo-300 hover:bg-indigo-50"
                >
                  <button
                    onClick={() => onOpenModel(model.model_id)}
                    className="w-full text-left"
                  >
                    <div className="text-sm font-black text-slate-900">
                      {model.model_name}
                    </div>
                    <div className="mt-1 text-xs text-slate-500">
                      {model.main_metric}:{" "}
                      <span className="font-bold text-slate-800">
                        {model.score}
                      </span>
                    </div>
                    <div className="mt-2 truncate text-[10px] text-slate-400">
                      {model.model_id}
                    </div>
                  </button>

                  <button
                    onClick={() => onSelectModel(model.model_id)}
                    className="mt-3 rounded-xl bg-slate-950 px-3 py-2 text-xs font-bold text-white hover:bg-slate-800"
                  >
                    Use for prediction
                  </button>
                </div>
              ))
            )}
          </div>
        )}

        {activeTab === "predictions" && (
          <div className="space-y-3">
            {predictions.length === 0 ? (
              <Empty />
            ) : (
              predictions.map((log) => (
                <div
                  key={log.prediction_id}
                  className="rounded-2xl border border-slate-200 bg-slate-50 p-4"
                >
                  <div className="truncate text-xs font-bold text-slate-700">
                    {log.model_id}
                  </div>
                  <div className="mt-2 rounded-xl bg-white p-2 text-xs text-slate-600">
                    {JSON.stringify(log.prediction)}
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {activeTab === "pipelines" && (
          <div className="space-y-3">
            {pipelines.length === 0 ? (
              <Empty />
            ) : (
              pipelines.map((pipeline) => (
                <div
                  key={pipeline.graph_id}
                  className="rounded-2xl border border-slate-200 bg-slate-50 p-4"
                >
                  <div className="text-sm font-black text-slate-900">
                    {pipeline.nodes?.length || 0} nodes ·{" "}
                    {pipeline.edges?.length || 0} edges
                  </div>
                  <div className="mt-2 truncate text-[10px] text-slate-400">
                    {pipeline.graph_id}
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </section>
  );
}

function Empty() {
  return (
    <div className="rounded-2xl border border-dashed border-slate-200 p-6 text-center text-xs text-slate-400">
      No records yet.
    </div>
  );
}