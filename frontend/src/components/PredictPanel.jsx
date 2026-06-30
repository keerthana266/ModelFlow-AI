import React, { useEffect, useState } from "react";
import {
  Code2,
  PlayCircle,
  RefreshCcw,
  Sparkles,
  Wand2,
} from "lucide-react";
import { api } from "../api";

export default function PredictPanel({
  models,
  selectedModelId,
  onSelectModel,
  onDone,
}) {
  const [inputJson, setInputJson] = useState("{}");
  const [modelDetails, setModelDetails] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);

  useEffect(() => {
    if (!selectedModelId) {
      setModelDetails(null);
      setInputJson("{}");
      setResult(null);
      setError("");
      return;
    }

    loadSelectedModelDetails(selectedModelId);
  }, [selectedModelId]);

  const loadSelectedModelDetails = async (modelId) => {
    try {
      setIsLoadingModel(true);
      setError("");
      setResult(null);

      const response = await api.get(`/models/${modelId}`);
      const details = response.data;

      setModelDetails(details);
      setInputJson(JSON.stringify(generateExampleInput(details), null, 2));
    } catch (error) {
      setError(error.response?.data?.detail || error.message);
    } finally {
      setIsLoadingModel(false);
    }
  };

  const generateExampleInput = (details) => {
    const featureColumns = details.feature_columns || [];
    const numericalColumns = new Set(details.numerical_columns || []);
    const categoricalColumns = new Set(details.categorical_columns || []);

    const example = {};

    featureColumns.forEach((column) => {
      if (numericalColumns.has(column)) {
        example[column] = getSmartNumericExample(column);
      } else if (categoricalColumns.has(column)) {
        example[column] = getSmartCategoricalExample(column);
      } else {
        example[column] = getFallbackExample(column);
      }
    });

    return example;
  };

  const regenerateInput = () => {
    if (!modelDetails) return;
    setInputJson(JSON.stringify(generateExampleInput(modelDetails), null, 2));
    setResult(null);
    setError("");
  };

  const runPrediction = async () => {
    try {
      setError("");
      setResult(null);

      if (!selectedModelId) {
        setError("Select a trained model first.");
        return;
      }

      const parsed = JSON.parse(inputJson);

      setIsPredicting(true);

      const response = await api.post(`/models/${selectedModelId}/predict`, {
        data: parsed,
      });

      setResult(response.data);
      await onDone?.();
    } catch (error) {
      if (error instanceof SyntaxError) {
        setError("Invalid JSON format. Please check commas, quotes, and braces.");
      } else {
        setError(error.response?.data?.detail || error.message);
      }
    } finally {
      setIsPredicting(false);
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
          <p className="text-xs text-slate-500">
            Test model using its real schema
          </p>
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

        {modelDetails && (
          <div className="rounded-2xl border border-emerald-100 bg-emerald-50 p-3">
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="text-xs font-bold uppercase tracking-wider text-emerald-700">
                  Active Model Schema
                </div>

                <div className="mt-1 text-sm font-bold text-slate-900">
                  {modelDetails.model_name}
                </div>

                <div className="mt-1 text-xs text-slate-600">
                  Target:{" "}
                  <span className="font-semibold">
                    {modelDetails.target_column}
                  </span>
                  {" · "}
                  Type:{" "}
                  <span className="font-semibold">
                    {modelDetails.problem_type}
                  </span>
                </div>

                <div className="mt-1 text-xs text-slate-500">
                  Features: {modelDetails.feature_columns?.length || 0}
                </div>
              </div>

              <button
                onClick={regenerateInput}
                className="rounded-xl bg-white px-3 py-2 text-xs font-bold text-emerald-700 shadow-sm transition hover:bg-emerald-100"
                title="Regenerate example input"
              >
                <RefreshCcw size={14} />
              </button>
            </div>
          </div>
        )}

        <div>
          <div className="mb-1 flex items-center justify-between">
            <label className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
              Input JSON
            </label>

            <button
              onClick={regenerateInput}
              disabled={!modelDetails}
              className="flex items-center gap-1 rounded-xl px-2 py-1 text-[11px] font-bold text-slate-500 transition hover:bg-slate-100 hover:text-emerald-600 disabled:opacity-40"
            >
              <Sparkles size={12} />
              Auto-fill
            </button>
          </div>

          <textarea
            value={inputJson}
            onChange={(e) => setInputJson(e.target.value)}
            className="soft-scroll h-40 w-full rounded-2xl border border-slate-200 bg-slate-950 px-3 py-3 font-mono text-xs text-slate-200 outline-none transition focus:border-emerald-400 focus:ring-4 focus:ring-emerald-100"
          />
        </div>

        {modelDetails?.feature_columns?.length > 0 && (
          <details className="rounded-2xl border border-slate-200 bg-slate-50">
            <summary className="flex cursor-pointer items-center gap-2 px-3 py-2 text-xs font-bold uppercase tracking-wider text-slate-500">
              <Code2 size={13} />
              Required Features
            </summary>

            <div className="soft-scroll max-h-32 overflow-auto border-t border-slate-200 p-3">
              <div className="flex flex-wrap gap-1.5">
                {modelDetails.feature_columns.map((feature) => (
                  <span
                    key={feature}
                    className="rounded-full bg-white px-2 py-1 text-[11px] font-medium text-slate-600 shadow-sm"
                  >
                    {feature}
                  </span>
                ))}
              </div>
            </div>
          </details>
        )}

        <button
          onClick={runPrediction}
          disabled={isPredicting || isLoadingModel || !selectedModelId}
          className="flex w-full items-center justify-center gap-2 rounded-2xl bg-emerald-600 px-4 py-3 text-sm font-bold text-white shadow-sm transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-slate-400"
        >
          <PlayCircle size={17} />
          {isPredicting
            ? "Predicting..."
            : isLoadingModel
            ? "Loading model..."
            : "Run Prediction"}
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

function getSmartNumericExample(column) {
  const key = column.toLowerCase();

  if (key.includes("age")) return 30;
  if (key.includes("area") || key.includes("sqft") || key.includes("size")) return 1200;
  if (key.includes("bedroom") || key.includes("bed")) return 3;
  if (key.includes("bathroom") || key.includes("bath")) return 2;
  if (key.includes("fare")) return 50;
  if (key.includes("salary") || key.includes("income")) return 50000;
  if (key.includes("year")) return 2024;
  if (key.includes("price")) return 100000;
  if (key.includes("distance")) return 10;
  if (key.includes("rating")) return 4.5;
  if (key.includes("count") || key.includes("quantity")) return 1;

  return 0;
}

function getSmartCategoricalExample(column) {
  const key = column.toLowerCase();

  if (key.includes("sex") || key.includes("gender")) return "female";
  if (key.includes("location") || key.includes("city")) return "Urban";
  if (key.includes("embarked")) return "S";
  if (key.includes("class")) return "First";
  if (key.includes("type")) return "standard";
  if (key.includes("category")) return "A";
  if (key.includes("status")) return "active";
  if (key.includes("yes") || key.includes("no")) return "yes";

  return "sample";
}

function getFallbackExample(column) {
  const key = column.toLowerCase();

  if (
    key.includes("id") ||
    key.includes("number") ||
    key.includes("amount") ||
    key.includes("score")
  ) {
    return 1;
  }

  return "sample";
}