import React from "react";
import {
  Activity,
  BarChart3,
  Bot,
  Database,
  LineChart,
  Rocket,
  Sparkles,
  Workflow,
} from "lucide-react";

export const NODE_TEMPLATES = [
  {
    type: "dataLoader",
    label: "Data Loader",
    description: "Upload/select CSV and target column",
    icon: "database",
    color: "blue",
  },
  {
    type: "preprocessor",
    label: "Preprocessor",
    description: "Clean missing values and encode features",
    icon: "workflow",
    color: "cyan",
  },
  {
    type: "automl",
    label: "AutoML Engine",
    description: "FLAML / Optuna model optimization",
    icon: "bot",
    color: "violet",
  },
  {
    type: "evaluator",
    label: "Evaluator",
    description: "Calculate accuracy, F1, R2, MAE and RMSE",
    icon: "bar-chart",
    color: "emerald",
  },
  {
    type: "explainer",
    label: "SHAP Explainer",
    description: "Generate feature importance explanations",
    icon: "sparkles",
    color: "amber",
  },
  {
    type: "mlflow",
    label: "MLflow Tracker",
    description: "Log metrics, params and model artifacts",
    icon: "line-chart",
    color: "indigo",
  },
  {
    type: "drift",
    label: "Drift Monitor",
    description: "Compare current data with training data",
    icon: "activity",
    color: "orange",
  },
  {
    type: "deploy",
    label: "Deploy API",
    description: "Deploy model as versioned REST API",
    icon: "rocket",
    color: "rose",
  },
];

export default function NodePalette({ onAddNode }) {
  const onDragStart = (event, template) => {
    event.dataTransfer.setData(
      "application/reactflow",
      JSON.stringify(template)
    );
    event.dataTransfer.effectAllowed = "move";
  };

  return (
    <section className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-100 px-4 py-4">
        <div className="flex items-center gap-2">
          <div className="grid h-9 w-9 place-items-center rounded-2xl bg-slate-950 text-white">
            <Workflow size={17} />
          </div>

          <div>
            <h2 className="text-sm font-bold text-slate-900">Node Palette</h2>
            <p className="text-xs text-slate-500">
              Drag nodes onto the canvas
            </p>
          </div>
        </div>
      </div>

      <div className="soft-scroll max-h-80 space-y-2 overflow-auto p-4">
        {NODE_TEMPLATES.map((template) => {
          const Icon = iconMap[template.icon] || Workflow;

          return (
            <div
              key={template.type}
              draggable
              onDragStart={(event) => onDragStart(event, template)}
              className="cursor-grab rounded-2xl border border-slate-200 bg-slate-50 p-3 transition hover:border-blue-300 hover:bg-blue-50 active:cursor-grabbing"
            >
              <div className="flex items-start gap-3">
                <div
                  className={`grid h-9 w-9 flex-shrink-0 place-items-center rounded-2xl ${iconClass(
                    template.color
                  )}`}
                >
                  <Icon size={16} />
                </div>

                <div className="min-w-0 flex-1">
                  <div className="text-sm font-black text-slate-900">
                    {template.label}
                  </div>

                  <p className="mt-0.5 text-xs leading-relaxed text-slate-500">
                    {template.description}
                  </p>

                  <button
                    onClick={() => onAddNode?.(template)}
                    className="mt-2 rounded-xl bg-white px-3 py-1.5 text-[11px] font-bold text-slate-600 shadow-sm transition hover:bg-slate-950 hover:text-white"
                  >
                    Add to canvas
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
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

function iconClass(color) {
  const classes = {
    blue: "bg-blue-50 text-blue-600",
    cyan: "bg-cyan-50 text-cyan-600",
    violet: "bg-violet-50 text-violet-600",
    emerald: "bg-emerald-50 text-emerald-600",
    amber: "bg-amber-50 text-amber-600",
    indigo: "bg-indigo-50 text-indigo-600",
    orange: "bg-orange-50 text-orange-600",
    rose: "bg-rose-50 text-rose-600",
  };

  return classes[color] || classes.blue;
}