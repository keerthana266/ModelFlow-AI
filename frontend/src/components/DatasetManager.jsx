import React, { useRef, useState } from "react";
import { Database, Eye, RefreshCcw, Trash2, UploadCloud } from "lucide-react";
import { api } from "../api";

export default function DatasetManager({
  datasets,
  selectedDatasetId,
  onSelectDataset,
  onRefresh,
  onOpenProfile,
}) {
  const fileRef = useRef(null);
  const [isUploading, setIsUploading] = useState(false);
  const [message, setMessage] = useState("");

  const handleUpload = async () => {
    const file = fileRef.current?.files?.[0];

    if (!file) {
      setMessage("Select a CSV file first.");
      return;
    }

    try {
      setIsUploading(true);
      setMessage("");

      const formData = new FormData();
      formData.append("file", file);

      const response = await api.post("/datasets/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const datasetId = response.data.dataset.dataset_id;

      setMessage(`Uploaded ${datasetId}`);
      fileRef.current.value = "";

      await onRefresh();
      onSelectDataset(datasetId);
    } catch (error) {
      setMessage(error.response?.data?.detail || error.message);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDelete = async (datasetId) => {
    if (!window.confirm("Delete this dataset?")) return;

    try {
      await api.delete(`/datasets/${datasetId}`);
      await onRefresh();

      if (selectedDatasetId === datasetId) {
        onSelectDataset("");
      }
    } catch (error) {
      alert(error.response?.data?.detail || error.message);
    }
  };

  return (
    <section className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
      <div className="flex items-center justify-between border-b border-slate-100 px-4 py-4">
        <div className="flex items-center gap-2">
          <div className="grid h-9 w-9 place-items-center rounded-2xl bg-blue-50 text-blue-600">
            <Database size={17} />
          </div>

          <div>
            <h2 className="text-sm font-bold text-slate-900">Datasets</h2>
            <p className="text-xs text-slate-500">Upload and inspect CSV data</p>
          </div>
        </div>

        <button
          onClick={onRefresh}
          className="rounded-xl p-2 text-slate-400 transition hover:bg-slate-100 hover:text-blue-600"
        >
          <RefreshCcw size={15} />
        </button>
      </div>

      <div className="space-y-4 p-4">
        <div className="rounded-2xl border border-dashed border-blue-200 bg-blue-50/60 p-4">
          <input
            ref={fileRef}
            type="file"
            accept=".csv"
            className="w-full text-xs text-slate-600 file:mr-3 file:rounded-xl file:border-0 file:bg-white file:px-3 file:py-2 file:text-xs file:font-semibold file:text-slate-700"
          />

          <button
            onClick={handleUpload}
            disabled={isUploading}
            className="mt-3 flex w-full items-center justify-center gap-2 rounded-2xl bg-blue-600 px-4 py-3 text-sm font-bold text-white shadow-sm transition hover:bg-blue-700 disabled:bg-slate-400"
          >
            <UploadCloud size={17} />
            {isUploading ? "Uploading..." : "Upload Dataset"}
          </button>
        </div>

        {message && (
          <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3 text-xs text-slate-600">
            {message}
          </div>
        )}

        <div className="soft-scroll max-h-72 space-y-2 overflow-auto pr-1">
          {datasets.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-slate-200 p-5 text-center text-xs text-slate-400">
              No datasets yet. Upload a CSV to begin.
            </div>
          ) : (
            datasets.map((dataset) => (
              <div
                key={dataset.dataset_id}
                onClick={() => onSelectDataset(dataset.dataset_id)}
                className={`cursor-pointer rounded-2xl border p-3 transition ${
                  selectedDatasetId === dataset.dataset_id
                    ? "border-blue-500 bg-blue-50 shadow-sm"
                    : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50"
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-bold text-slate-800">
                      {dataset.original_filename}
                    </div>

                    <div className="mt-1 text-xs text-slate-500">
                      {dataset.rows} rows · {dataset.columns} columns
                    </div>

                    <div className="mt-1 truncate text-[10px] text-slate-400">
                      {dataset.dataset_id}
                    </div>
                  </div>

                  <div className="flex gap-1">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onOpenProfile(dataset.dataset_id);
                      }}
                      className="rounded-lg p-1 text-slate-300 transition hover:bg-blue-50 hover:text-blue-500"
                    >
                      <Eye size={15} />
                    </button>

                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(dataset.dataset_id);
                      }}
                      className="rounded-lg p-1 text-slate-300 transition hover:bg-red-50 hover:text-red-500"
                    >
                      <Trash2 size={15} />
                    </button>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </section>
  );
}