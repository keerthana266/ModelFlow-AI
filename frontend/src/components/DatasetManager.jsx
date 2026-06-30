import React, { useRef, useState } from "react";
import { Database, RefreshCcw, Trash2, Upload } from "lucide-react";
import { api } from "../api";

export default function DatasetManager({
  datasets,
  selectedDatasetId,
  onSelectDataset,
  onRefresh,
}) {
  const fileRef = useRef(null);
  const [isUploading, setIsUploading] = useState(false);
  const [message, setMessage] = useState("");

  const handleUpload = async () => {
    const file = fileRef.current?.files?.[0];

    if (!file) {
      setMessage("Please select a CSV file first.");
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

      setMessage(`Uploaded: ${response.data.dataset.dataset_id}`);
      fileRef.current.value = "";

      await onRefresh();
      onSelectDataset(response.data.dataset.dataset_id);
    } catch (error) {
      setMessage(error.response?.data?.detail || error.message);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDelete = async (datasetId) => {
    const ok = window.confirm("Delete this dataset?");
    if (!ok) return;

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
    <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-100 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Database size={17} className="text-blue-600" />
          <h2 className="font-bold text-sm text-slate-800">Datasets</h2>
        </div>

        <button
          onClick={onRefresh}
          className="text-xs flex items-center gap-1 text-slate-500 hover:text-blue-600"
        >
          <RefreshCcw size={13} /> Refresh
        </button>
      </div>

      <div className="p-4 flex flex-col gap-3">
        <input
          ref={fileRef}
          type="file"
          accept=".csv"
          className="text-xs border border-slate-200 rounded p-2"
        />

        <button
          onClick={handleUpload}
          disabled={isUploading}
          className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-400 text-white rounded-md px-3 py-2 text-sm font-medium flex items-center justify-center gap-2"
        >
          <Upload size={15} />
          {isUploading ? "Uploading..." : "Upload CSV"}
        </button>

        {message && (
          <div className="text-xs bg-slate-50 border border-slate-200 rounded p-2 text-slate-600">
            {message}
          </div>
        )}

        <div className="max-h-72 overflow-auto flex flex-col gap-2">
          {datasets.length === 0 ? (
            <div className="text-xs text-slate-400 text-center py-4">
              No datasets uploaded yet.
            </div>
          ) : (
            datasets.map((dataset) => (
              <div
                key={dataset.dataset_id}
                className={`border rounded-lg p-3 cursor-pointer transition ${
                  selectedDatasetId === dataset.dataset_id
                    ? "border-blue-500 bg-blue-50"
                    : "border-slate-200 hover:bg-slate-50"
                }`}
                onClick={() => onSelectDataset(dataset.dataset_id)}
              >
                <div className="flex justify-between gap-2">
                  <div className="min-w-0">
                    <div className="font-semibold text-sm text-slate-800 truncate">
                      {dataset.original_filename}
                    </div>
                    <div className="text-xs text-slate-500">
                      {dataset.rows} rows · {dataset.columns} columns
                    </div>
                    <div className="text-[10px] text-slate-400 truncate">
                      {dataset.dataset_id}
                    </div>
                  </div>

                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(dataset.dataset_id);
                    }}
                    className="text-slate-400 hover:text-red-500"
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}