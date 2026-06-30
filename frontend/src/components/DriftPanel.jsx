import React, { useEffect, useState } from "react";
import { Activity, ExternalLink, RefreshCcw, ShieldCheck } from "lucide-react";
import { API_BASE_URL, api } from "../api";

export default function DriftPanel({ models, datasets }) {
  const [selectedModelId, setSelectedModelId] = useState("");
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [reports, setReports] = useState([]);
  const [result, setResult] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState("");

  const refreshReports = async () => {
    const response = await api.get("/monitoring/drift");
    setReports(response.data.drift_reports || []);
  };

  useEffect(() => {
    refreshReports().catch(console.error);
  }, []);

  const runDriftCheck = async () => {
    try {
      setError("");
      setResult(null);

      if (!selectedModelId) {
        setError("Select a model first.");
        return;
      }

      if (!selectedDatasetId) {
        setError("Select current dataset first.");
        return;
      }

      setIsRunning(true);

      const response = await api.post("/monitoring/drift", {
        model_id: selectedModelId,
        current_dataset_id: selectedDatasetId,
        max_rows: 1000,
      });

      setResult(response.data);
      await refreshReports();
    } catch (error) {
      setError(error.response?.data?.detail || error.message);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <section className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
      <div className="flex items-center justify-between border-b border-slate-100 px-4 py-4">
        <div className="flex items-center gap-2">
          <div className="grid h-9 w-9 place-items-center rounded-2xl bg-orange-50 text-orange-600">
            <Activity size={17} />
          </div>

          <div>
            <h2 className="text-sm font-bold text-slate-900">
              Drift Monitor
            </h2>
            <p className="text-xs text-slate-500">
              Compare current data with training data
            </p>
          </div>
        </div>

        <button
          onClick={refreshReports}
          className="rounded-xl p-2 text-slate-400 transition hover:bg-slate-100 hover:text-orange-600"
        >
          <RefreshCcw size={15} />
        </button>
      </div>

      <div className="space-y-4 p-4">
        <div>
          <label className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
            Model
          </label>

          <select
            value={selectedModelId}
            onChange={(e) => setSelectedModelId(e.target.value)}
            className="mt-1 w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 outline-none focus:border-orange-400 focus:ring-4 focus:ring-orange-100"
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
            Current Dataset
          </label>

          <select
            value={selectedDatasetId}
            onChange={(e) => setSelectedDatasetId(e.target.value)}
            className="mt-1 w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 outline-none focus:border-orange-400 focus:ring-4 focus:ring-orange-100"
          >
            <option value="">Select current dataset</option>

            {datasets.map((dataset) => (
              <option key={dataset.dataset_id} value={dataset.dataset_id}>
                {dataset.original_filename}
              </option>
            ))}
          </select>
        </div>

        <button
          onClick={runDriftCheck}
          disabled={isRunning}
          className="flex w-full items-center justify-center gap-2 rounded-2xl bg-orange-600 px-4 py-3 text-sm font-bold text-white shadow-sm transition hover:bg-orange-700 disabled:bg-slate-400"
        >
          <ShieldCheck size={17} />
          {isRunning ? "Checking Drift..." : "Run Drift Check"}
        </button>

        {error && (
          <div className="rounded-2xl border border-red-100 bg-red-50 p-3 text-xs text-red-600">
            {error}
          </div>
        )}

        {result && (
          <div
            className={`rounded-2xl border p-3 ${
              result.drift_detected
                ? "border-red-100 bg-red-50"
                : "border-emerald-100 bg-emerald-50"
            }`}
          >
            <div
              className={`text-xs font-bold uppercase tracking-wider ${
                result.drift_detected ? "text-red-700" : "text-emerald-700"
              }`}
            >
              {result.drift_detected ? "Drift Detected" : "No Drift Detected"}
            </div>

            <div className="mt-2 text-xs text-slate-700">
              Drifted columns:{" "}
              <span className="font-bold">
                {result.drifted_columns}/{result.total_columns}
              </span>
            </div>

            <a
              href={`${API_BASE_URL}${result.html_report_url}`}
              target="_blank"
              rel="noreferrer"
              className="mt-3 inline-flex items-center gap-1 rounded-xl bg-white px-3 py-2 text-xs font-bold text-slate-700 shadow-sm hover:bg-slate-50"
            >
              Open HTML Report
              <ExternalLink size={13} />
            </a>
          </div>
        )}

        <div>
          <div className="mb-2 text-[11px] font-bold uppercase tracking-wider text-slate-500">
            Recent Drift Reports
          </div>

          <div className="soft-scroll max-h-48 space-y-2 overflow-auto pr-1">
            {reports.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-slate-200 p-4 text-center text-xs text-slate-400">
                No drift reports yet.
              </div>
            ) : (
              reports.map((report) => (
                <div
                  key={report.report_id}
                  className="rounded-2xl border border-slate-200 bg-slate-50 p-3"
                >
                  <div className="text-xs font-bold text-slate-800">
                    {report.drift_detected ? "Drift Detected" : "No Drift"}
                  </div>

                  <div className="mt-1 text-xs text-slate-500">
                    {report.drifted_columns}/{report.total_columns} drifted
                  </div>

                  <a
                    href={`${API_BASE_URL}${report.html_report_url}`}
                    target="_blank"
                    rel="noreferrer"
                    className="mt-2 inline-flex items-center gap-1 text-xs font-bold text-orange-600 hover:text-orange-700"
                  >
                    View report
                    <ExternalLink size={12} />
                  </a>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </section>
  );
}