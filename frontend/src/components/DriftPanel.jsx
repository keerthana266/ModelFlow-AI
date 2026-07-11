import React, { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  ExternalLink,
  Loader2,
  Play,
} from "lucide-react";

import { api } from "../api";

export default function DriftPanel({ models = [], datasets = [] }) {
  const [modelId, setModelId] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [maxRows, setMaxRows] = useState(1000);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const selectedModel = useMemo(
    () => models.find((model) => model.model_id === modelId),
    [models, modelId]
  );

  const runCheck = async () => {
    if (!modelId || !datasetId) {
      setError("Select both a trained model and a current dataset.");
      return;
    }

    setRunning(true);
    setError("");
    setResult(null);

    try {
      const response = await api.post("/monitoring/drift", {
        model_id: modelId,
        current_dataset_id: datasetId,
        max_rows: Number(maxRows || 1000),
      });
      setResult(response.data);
    } catch (requestError) {
      const detail = requestError.response?.data?.detail;
      setError(
        typeof detail === "string"
          ? detail
          : detail
          ? JSON.stringify(detail)
          : requestError.message
      );
    } finally {
      setRunning(false);
    }
  };

  const openReport = async () => {
    if (!result?.report_id) return;

    const reportWindow = window.open("", "_blank");

    try {
      const response = await api.get(
        `/monitoring/drift/${result.report_id}/html`,
        { responseType: "blob" }
      );
      const objectUrl = URL.createObjectURL(response.data);

      if (reportWindow) {
        reportWindow.location.href = objectUrl;
      } else {
        window.location.href = objectUrl;
      }

      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 60_000);
    } catch (requestError) {
      reportWindow?.close();
      setError(
        requestError.response?.data?.detail ||
          requestError.message ||
          "Could not open the drift report."
      );
    }
  };

  return (
    <section className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-100 px-4 py-4">
        <div className="flex items-center gap-3">
          <div className="grid h-9 w-9 place-items-center rounded-2xl bg-orange-50 text-orange-600">
            <Activity size={17} />
          </div>
          <div>
            <h2 className="text-sm font-black text-slate-900">Drift Monitor</h2>
            <p className="text-xs text-slate-500">
              Compare current data with training data
            </p>
          </div>
        </div>
      </div>

      <div className="space-y-3 p-4">
        <Field label="Model">
          <select
            value={modelId}
            onChange={(event) => setModelId(event.target.value)}
            className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm outline-none focus:border-orange-400"
          >
            <option value="">Select trained model</option>
            {models.map((model) => (
              <option key={model.model_id} value={model.model_id}>
                {model.model_name || model.model_id}
              </option>
            ))}
          </select>
        </Field>

        <Field label="Current dataset">
          <select
            value={datasetId}
            onChange={(event) => setDatasetId(event.target.value)}
            className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm outline-none focus:border-orange-400"
          >
            <option value="">Select comparison dataset</option>
            {datasets.map((dataset) => (
              <option key={dataset.dataset_id} value={dataset.dataset_id}>
                {dataset.original_filename || dataset.dataset_id}
              </option>
            ))}
          </select>
        </Field>

        <Field label="Maximum rows">
          <input
            type="number"
            min="2"
            max="10000"
            value={maxRows}
            onChange={(event) => setMaxRows(event.target.value)}
            className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm outline-none focus:border-orange-400"
          />
        </Field>

        {selectedModel && (
          <div className="rounded-2xl bg-slate-50 p-3 text-xs text-slate-500">
            Target: <strong className="text-slate-800">{selectedModel.target_column}</strong>
          </div>
        )}

        <button
          type="button"
          onClick={runCheck}
          disabled={running || !modelId || !datasetId}
          className="flex w-full items-center justify-center gap-2 rounded-2xl bg-orange-600 px-4 py-3 text-sm font-black text-white transition hover:bg-orange-700 disabled:cursor-not-allowed disabled:bg-slate-400"
        >
          {running ? (
            <Loader2 size={16} className="animate-spin" />
          ) : (
            <Play size={16} fill="currentColor" />
          )}
          {running ? "Checking drift..." : "Run Drift Check"}
        </button>

        {error && (
          <div className="rounded-2xl border border-red-100 bg-red-50 p-3 text-xs text-red-700">
            {error}
          </div>
        )}

        {result && (
          <div
            className={`rounded-2xl border p-3 ${
              result.drift_detected
                ? "border-amber-200 bg-amber-50"
                : "border-emerald-200 bg-emerald-50"
            }`}
          >
            <div className="flex items-center gap-2 text-sm font-black">
              {result.drift_detected ? (
                <AlertTriangle size={16} className="text-amber-700" />
              ) : (
                <CheckCircle2 size={16} className="text-emerald-700" />
              )}
              {result.drift_detected ? "Drift detected" : "No significant drift"}
            </div>
            <div className="mt-2 space-y-1 text-xs text-slate-600">
              <div>Report: {result.report_id}</div>
              <div>
                Drift share: {((Number(result.drift_score) || 0) * 100).toFixed(2)}%
              </div>
              <div>
                Drifted columns: {result.drifted_columns || 0}/{result.total_columns || 0}
              </div>
            </div>
            <button
              type="button"
              onClick={openReport}
              className="mt-3 flex w-full items-center justify-center gap-2 rounded-xl bg-white px-3 py-2 text-xs font-black text-slate-700 shadow-sm"
            >
              <ExternalLink size={13} />
              Open Evidently Report
            </button>
          </div>
        )}
      </div>
    </section>
  );
}

function Field({ label, children }) {
  return (
    <label className="block">
      <span className="mb-1 block text-[11px] font-black uppercase tracking-wider text-slate-500">
        {label}
      </span>
      {children}
    </label>
  );
}
