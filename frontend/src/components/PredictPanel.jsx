import React, { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Code2,
  Loader2,
  Play,
  RefreshCcw,
  Wand2,
  XCircle,
} from "lucide-react";

import { api } from "../api";

export default function PredictPanel({
  models = [],
  selectedModelId = "",
  onSelectModel,
  onDone,
}) {
  const [modelDetails, setModelDetails] = useState(null);
  const [inputText, setInputText] = useState("{}");
  const [predictionResult, setPredictionResult] = useState(null);
  const [schemaError, setSchemaError] = useState(null);
  const [genericError, setGenericError] = useState("");
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);

  const activeModelId = selectedModelId || models[0]?.model_id || "";

  const activeModel = useMemo(() => {
    return models.find((model) => model.model_id === activeModelId);
  }, [models, activeModelId]);

  useEffect(() => {
    if (!activeModelId) {
      setModelDetails(null);
      setInputText("{}");
      return;
    }

    loadModelDetails(activeModelId);
  }, [activeModelId]);

  const loadModelDetails = async (modelId) => {
    try {
      setIsLoadingModel(true);
      setGenericError("");
      setSchemaError(null);

      const response = await api.get(`/models/${modelId}`);
      const details = response.data;

      setModelDetails(details);

      const example =
        details.input_schema?.example ||
        buildFallbackExample(details.feature_columns || []);

      setInputText(
        JSON.stringify(
          {
            data: example,
          },
          null,
          2
        )
      );
    } catch (error) {
      setGenericError(getReadableError(error));
    } finally {
      setIsLoadingModel(false);
    }
  };

  const handlePredict = async () => {
    if (!activeModelId) {
      setGenericError("Please select a model first.");
      return;
    }

    let payload;

    try {
      payload = JSON.parse(inputText);
    } catch {
      setGenericError("Invalid JSON input.");
      setSchemaError(null);
      setPredictionResult(null);
      return;
    }

    try {
      setIsPredicting(true);
      setGenericError("");
      setSchemaError(null);
      setPredictionResult(null);

      const response = await api.post(`/models/${activeModelId}/predict`, payload);

      setPredictionResult(response.data);
      onDone?.();
    } catch (error) {
      const parsedError = parsePredictionError(error);

      if (parsedError.schemaError) {
        setSchemaError(parsedError.schemaError);
      } else {
        setGenericError(parsedError.message);
      }
    } finally {
      setIsPredicting(false);
    }
  };

  const handleUseExample = () => {
    const example =
      modelDetails?.input_schema?.example ||
      buildFallbackExample(modelDetails?.feature_columns || []);

    setInputText(
      JSON.stringify(
        {
          data: example,
        },
        null,
        2
      )
    );

    setGenericError("");
    setSchemaError(null);
    setPredictionResult(null);
  };

  const handleUseBatchExample = () => {
    const example =
      modelDetails?.input_schema?.example ||
      buildFallbackExample(modelDetails?.feature_columns || []);

    setInputText(
      JSON.stringify(
        {
          data: [example, example],
        },
        null,
        2
      )
    );

    setGenericError("");
    setSchemaError(null);
    setPredictionResult(null);
  };

  return (
    <section className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-100 px-4 py-4">
        <div className="flex items-center gap-2">
          <div className="grid h-9 w-9 place-items-center rounded-2xl bg-emerald-50 text-emerald-600">
            <Play size={17} fill="currentColor" />
          </div>

          <div className="min-w-0 flex-1">
            <h2 className="text-sm font-bold text-slate-900">
              Prediction Tester
            </h2>
            <p className="text-xs text-slate-500">
              Test trained models with schema validation
            </p>
          </div>
        </div>
      </div>

      <div className="space-y-4 p-4">
        <div>
          <label className="text-xs font-bold uppercase tracking-wider text-slate-500">
            Model
          </label>

          <select
            value={activeModelId}
            onChange={(event) => onSelectModel?.(event.target.value)}
            className="mt-1 w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 outline-none transition focus:border-emerald-400 focus:bg-white focus:ring-4 focus:ring-emerald-100"
          >
            <option value="">Select model</option>

            {models.map((model) => (
              <option key={model.model_id} value={model.model_id}>
                {model.model_name || model.model_id}
              </option>
            ))}
          </select>
        </div>

        {isLoadingModel && (
          <div className="flex items-center gap-2 rounded-2xl border border-slate-100 bg-slate-50 px-3 py-2 text-xs font-semibold text-slate-500">
            <Loader2 size={14} className="animate-spin" />
            Loading model schema...
          </div>
        )}

        {modelDetails && (
          <div className="rounded-2xl border border-slate-100 bg-slate-50 p-3">
            <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
              Required Features
            </div>

            <div className="mt-2 flex flex-wrap gap-1.5">
              {(modelDetails.feature_columns || []).map((column) => (
                <span
                  key={column}
                  className="rounded-full bg-white px-2 py-1 text-[11px] font-semibold text-slate-600 shadow-sm"
                >
                  {column}
                </span>
              ))}
            </div>

            <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
              <InfoBox
                label="Problem"
                value={modelDetails.problem_type || activeModel?.problem_type}
              />
              <InfoBox
                label="Target"
                value={modelDetails.target_column || activeModel?.target_column}
              />
            </div>
          </div>
        )}

        <div>
          <div className="mb-1 flex items-center justify-between gap-2">
            <label className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-500">
              <Code2 size={13} />
              JSON Input
            </label>

            <div className="flex items-center gap-1.5">
              <button
                onClick={handleUseExample}
                className="rounded-xl bg-slate-100 px-2.5 py-1.5 text-[11px] font-bold text-slate-600 transition hover:bg-slate-950 hover:text-white"
              >
                Example
              </button>

              <button
                onClick={handleUseBatchExample}
                className="rounded-xl bg-slate-100 px-2.5 py-1.5 text-[11px] font-bold text-slate-600 transition hover:bg-slate-950 hover:text-white"
              >
                Batch
              </button>
            </div>
          </div>

          <textarea
            value={inputText}
            onChange={(event) => setInputText(event.target.value)}
            rows={9}
            spellCheck={false}
            className="soft-scroll w-full resize-none rounded-2xl border border-slate-200 bg-slate-950 p-3 font-mono text-xs leading-relaxed text-emerald-100 outline-none transition focus:border-emerald-400 focus:ring-4 focus:ring-emerald-100"
          />
        </div>

        <button
          onClick={handlePredict}
          disabled={isPredicting || !activeModelId}
          className={`flex w-full items-center justify-center gap-2 rounded-2xl px-4 py-3 text-sm font-bold text-white transition ${
            isPredicting || !activeModelId
              ? "cursor-not-allowed bg-slate-400"
              : "bg-emerald-600 hover:bg-emerald-700"
          }`}
        >
          {isPredicting ? (
            <Loader2 size={16} className="animate-spin" />
          ) : (
            <Play size={16} fill="currentColor" />
          )}
          {isPredicting ? "Predicting..." : "Run Prediction"}
        </button>

        {genericError && <ErrorBox message={genericError} />}

        {schemaError && <SchemaErrorBox schemaError={schemaError} />}

        {predictionResult && <PredictionResult result={predictionResult} />}

        <button
          onClick={() => activeModelId && loadModelDetails(activeModelId)}
          disabled={!activeModelId}
          className="flex w-full items-center justify-center gap-2 rounded-2xl border border-slate-200 bg-white px-4 py-2.5 text-xs font-bold text-slate-600 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
        >
          <RefreshCcw size={14} />
          Reload Schema
        </button>
      </div>
    </section>
  );
}

function InfoBox({ label, value }) {
  return (
    <div className="rounded-xl bg-white p-2 shadow-sm">
      <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
        {label}
      </div>
      <div className="mt-0.5 truncate text-xs font-bold text-slate-700">
        {value || "-"}
      </div>
    </div>
  );
}

function ErrorBox({ message }) {
  return (
    <div className="rounded-2xl border border-red-100 bg-red-50 p-3">
      <div className="flex items-center gap-2 text-sm font-black text-red-700">
        <XCircle size={16} />
        Error
      </div>

      <p className="mt-1 text-xs leading-relaxed text-red-600">{message}</p>
    </div>
  );
}

function SchemaErrorBox({ schemaError }) {
  return (
    <div className="rounded-2xl border border-amber-100 bg-amber-50 p-3">
      <div className="flex items-center gap-2 text-sm font-black text-amber-800">
        <AlertTriangle size={16} />
        {schemaError.message || "Prediction input schema validation failed."}
      </div>

      {Array.isArray(schemaError.errors) && schemaError.errors.length > 0 && (
        <ul className="mt-2 space-y-1">
          {schemaError.errors.map((item, index) => (
            <li
              key={`${item}_${index}`}
              className="rounded-xl bg-white px-3 py-2 text-xs font-semibold text-amber-800 shadow-sm"
            >
              {item}
            </li>
          ))}
        </ul>
      )}

      {Array.isArray(schemaError.warnings) &&
        schemaError.warnings.length > 0 && (
          <div className="mt-3 rounded-xl bg-white p-3">
            <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
              Warnings
            </div>

            <ul className="mt-1 space-y-1">
              {schemaError.warnings.map((item, index) => (
                <li
                  key={`${item}_${index}`}
                  className="text-xs font-medium text-slate-600"
                >
                  {item}
                </li>
              ))}
            </ul>
          </div>
        )}

      {schemaError.input_schema?.example && (
        <div className="mt-3 rounded-xl bg-white p-3">
          <div className="mb-2 flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider text-slate-400">
            <Wand2 size={12} />
            Expected Example
          </div>

          <pre className="soft-scroll max-h-44 overflow-auto rounded-xl bg-slate-950 p-3 text-[11px] leading-relaxed text-emerald-100">
            {JSON.stringify(
              {
                data: schemaError.input_schema.example,
              },
              null,
              2
            )}
          </pre>
        </div>
      )}
    </div>
  );
}

function PredictionResult({ result }) {
  return (
    <div className="rounded-2xl border border-emerald-100 bg-emerald-50 p-3">
      <div className="flex items-center gap-2 text-sm font-black text-emerald-700">
        <CheckCircle2 size={16} />
        Prediction Complete
      </div>

      <div className="mt-3 rounded-xl bg-white p-3">
        <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
          Prediction
        </div>

        <pre className="soft-scroll mt-2 max-h-52 overflow-auto rounded-xl bg-slate-950 p-3 text-xs leading-relaxed text-emerald-100">
          {JSON.stringify(result.prediction, null, 2)}
        </pre>
      </div>

      {result.schema_validation && (
        <div className="mt-3 rounded-xl bg-white p-3">
          <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
            Schema Validation
          </div>

          <pre className="soft-scroll mt-2 max-h-44 overflow-auto rounded-xl bg-slate-950 p-3 text-[11px] leading-relaxed text-emerald-100">
            {JSON.stringify(result.schema_validation, null, 2)}
          </pre>
        </div>
      )}

      <div className="mt-3 rounded-xl bg-white p-3">
        <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
          Full Response
        </div>

        <pre className="soft-scroll mt-2 max-h-52 overflow-auto rounded-xl bg-slate-950 p-3 text-[11px] leading-relaxed text-emerald-100">
          {JSON.stringify(result, null, 2)}
        </pre>
      </div>
    </div>
  );
}

function parsePredictionError(error) {
  const detail = error.response?.data?.detail;

  if (detail && typeof detail === "object") {
    return {
      message: detail.message || "Prediction failed.",
      schemaError: {
        message: detail.message || "Prediction input schema validation failed.",
        errors: Array.isArray(detail.errors) ? detail.errors : [],
        warnings: Array.isArray(detail.warnings) ? detail.warnings : [],
        input_schema: detail.input_schema || null,
      },
    };
  }

  return {
    message: getReadableError(error),
    schemaError: null,
  };
}

function getReadableError(error) {
  const detail = error.response?.data?.detail;

  if (typeof detail === "string") {
    return detail;
  }

  if (detail && typeof detail === "object") {
    return detail.message || JSON.stringify(detail);
  }

  return error.message || "Something went wrong.";
}

function buildFallbackExample(featureColumns) {
  const example = {};

  featureColumns.forEach((column) => {
    const key = String(column).toLowerCase();

    if (key.includes("age")) {
      example[column] = 30;
    } else if (key.includes("tenure")) {
      example[column] = 12;
    } else if (key.includes("spend") || key.includes("amount")) {
      example[column] = 75;
    } else if (key.includes("ticket") || key.includes("count")) {
      example[column] = 1;
    } else if (key.includes("plan")) {
      example[column] = "Premium";
    } else if (key.includes("region")) {
      example[column] = "East";
    } else if (key.includes("sex") || key.includes("gender")) {
      example[column] = "female";
    } else if (key.includes("embarked")) {
      example[column] = "S";
    } else {
      example[column] = "sample";
    }
  });

  return example;
}