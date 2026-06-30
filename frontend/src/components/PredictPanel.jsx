import React, { useState } from "react";
import { PlayCircle, Wand2 } from "lucide-react";
import { api } from "../api";

export default function PredictPanel({
  models,
  selectedModelId,
  onSelectModel,
  onDone,
}) {
  const [inputJson, setInputJson] = useState(`{
  "Age": 25,
  "Fare": 50,
  "Sex": "female"
}`);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const runPrediction = async () => {
    try {
      setError("");
      setResult(null);

      if (!selectedModelId) {
        setError("Select a trained model first.");
        return;
      }

      const parsed = JSON.parse(inputJson);

      const response = await api.post(`/models/${selectedModelId}/predict`, {
        data: parsed,
      });

      setResult(response.data);
      await onDone?.();
    } catch (error) {
      setError(error.response?.data?.detail || error.message);
    }
  };

  return (
    <section className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
      <div className="flex items-center gap-2 border-b border-slate-100 px-4 py-4">
        <div className="grid h-9 w-9 place-items-center rounded-2xl bg-emerald-50 text-emerald-600">
          <Wand2 size={17} />
        </div>

        <div>
          <h2 className="text-sm font-bold text-slate-900">
            Prediction Tester
          </h2>
          <p className="text-xs text-slate-500">Test deployed model endpoint</p>
        </div>
      </div>

      <div className="space-y-4 p-4">
        <div>
          <label className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
            Model
          </label>

          <select
            value={selectedModelId || ""}
            onChange={(e) => onSelectModel(e.target.value)}
            className="mt-1 w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 outline-none transition focus:border-emerald-400 focus:bg-white focus:ring-4 focus:ring-emerald-100"
          >
            <option value="">Select model</option>

            {models.map((model) => (
              <option key={model.model_id} value={model.model_id}>
                {model.model_name} · {model.score}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
            Input JSON
          </label>

          <textarea
            value={inputJson}
            onChange={(e) => setInputJson(e.target.value)}
            className="soft-scroll mt-1 h-36 w-full rounded-2xl border border-slate-200 bg-slate-950 px-3 py-3 font-mono text-xs text-slate-200 outline-none transition focus:border-emerald-400 focus:ring-4 focus:ring-emerald-100"
          />
        </div>

        <button
          onClick={runPrediction}
          className="flex w-full items-center justify-center gap-2 rounded-2xl bg-emerald-600 px-4 py-3 text-sm font-bold text-white shadow-sm transition hover:bg-emerald-700"
        >
          <PlayCircle size={17} />
          Run Prediction
        </button>

        {error && (
          <div className="rounded-2xl border border-red-100 bg-red-50 p-3 text-xs text-red-600">
            {error}
          </div>
        )}

        {result && (
          <div className="rounded-2xl border border-emerald-100 bg-emerald-50 p-3">
            <div className="text-xs font-bold uppercase tracking-wider text-emerald-700">
              Prediction Output
            </div>

            <pre className="soft-scroll mt-2 max-h-44 overflow-auto text-xs text-slate-800">
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </section>
  );
}