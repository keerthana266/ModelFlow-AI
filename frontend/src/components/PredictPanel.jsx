import React, { useState } from "react";
import { PlayCircle } from "lucide-react";
import { api } from "../api";

export default function PredictPanel({ models, selectedModelId, onSelectModel, onDone }) {
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
        setError("Select a model first.");
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
    <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-100 flex items-center gap-2">
        <PlayCircle size={17} className="text-green-600" />
        <h2 className="font-bold text-sm text-slate-800">Prediction Tester</h2>
      </div>

      <div className="p-4 flex flex-col gap-3">
        <div>
          <label className="text-xs font-bold text-slate-500 uppercase">
            Model
          </label>

          <select
            value={selectedModelId || ""}
            onChange={(e) => onSelectModel(e.target.value)}
            className="w-full mt-1 px-2 py-2 text-sm border border-slate-300 rounded-md bg-white"
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
          <label className="text-xs font-bold text-slate-500 uppercase">
            Input JSON
          </label>

          <textarea
            value={inputJson}
            onChange={(e) => setInputJson(e.target.value)}
            className="w-full mt-1 h-40 px-3 py-2 text-xs font-mono border border-slate-300 rounded-md"
          />
        </div>

        <button
          onClick={runPrediction}
          className="bg-green-600 hover:bg-green-700 text-white rounded-md px-3 py-2 text-sm font-medium"
        >
          Run Prediction
        </button>

        {error && (
          <div className="text-xs text-red-500 bg-red-50 border border-red-100 rounded p-2">
            {error}
          </div>
        )}

        {result && (
          <div className="bg-slate-950 text-slate-200 p-3 rounded-lg text-xs font-mono overflow-auto max-h-60">
            <pre>{JSON.stringify(result, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  );
}