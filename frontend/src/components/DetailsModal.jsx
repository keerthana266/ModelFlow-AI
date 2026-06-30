import React from "react";
import { X } from "lucide-react";

export default function DetailsModal({ open, title, subtitle, data, onClose }) {
  if (!open || !data) return null;

  return (
    <div className="fixed inset-0 z-[90] grid place-items-center bg-slate-950/40 p-6 backdrop-blur-sm">
      <div className="soft-scroll max-h-[86vh] w-full max-w-3xl overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-[0_30px_90px_rgba(15,23,42,0.35)]">
        <div className="flex items-center justify-between border-b border-slate-100 px-6 py-5">
          <div>
            <h2 className="text-lg font-black text-slate-950">{title}</h2>
            {subtitle && <p className="text-sm text-slate-500">{subtitle}</p>}
          </div>

          <button
            onClick={onClose}
            className="rounded-2xl p-2 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
          >
            <X size={19} />
          </button>
        </div>

        <div className="soft-scroll max-h-[70vh] overflow-auto p-6">
          <div className="grid grid-cols-2 gap-4">
            {Object.entries(data)
              .filter(([_, value]) => typeof value !== "object" || value === null)
              .map(([key, value]) => (
                <div
                  key={key}
                  className="rounded-2xl border border-slate-200 bg-slate-50 p-4"
                >
                  <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                    {key.replaceAll("_", " ")}
                  </div>
                  <div className="mt-1 break-all text-sm font-semibold text-slate-800">
                    {String(value ?? "N/A")}
                  </div>
                </div>
              ))}
          </div>

          <div className="mt-6 space-y-4">
            {Object.entries(data)
              .filter(([_, value]) => typeof value === "object" && value !== null)
              .map(([key, value]) => (
                <details
                  key={key}
                  open={["metrics", "profile", "feature_importance"].includes(key)}
                  className="overflow-hidden rounded-2xl border border-slate-200 bg-white"
                >
                  <summary className="cursor-pointer bg-slate-50 px-4 py-3 text-xs font-bold uppercase tracking-wider text-slate-500">
                    {key.replaceAll("_", " ")}
                  </summary>

                  <pre className="soft-scroll max-h-72 overflow-auto bg-slate-950 p-4 text-xs text-slate-200">
                    {JSON.stringify(value, null, 2)}
                  </pre>
                </details>
              ))}
          </div>
        </div>
      </div>
    </div>
  );
}