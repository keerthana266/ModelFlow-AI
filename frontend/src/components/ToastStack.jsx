import React from "react";
import { CheckCircle2, Info, XCircle } from "lucide-react";

export default function ToastStack({ toasts, onRemove }) {
  return (
    <div className="fixed right-5 top-20 z-[100] flex w-96 flex-col gap-3">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className="rounded-2xl border border-slate-200 bg-white p-4 shadow-[0_20px_50px_rgba(15,23,42,0.16)]"
        >
          <div className="flex items-start gap-3">
            <div
              className={`mt-0.5 grid h-8 w-8 flex-shrink-0 place-items-center rounded-xl ${
                toast.type === "success"
                  ? "bg-emerald-50 text-emerald-600"
                  : toast.type === "error"
                  ? "bg-red-50 text-red-600"
                  : "bg-blue-50 text-blue-600"
              }`}
            >
              {toast.type === "success" ? (
                <CheckCircle2 size={17} />
              ) : toast.type === "error" ? (
                <XCircle size={17} />
              ) : (
                <Info size={17} />
              )}
            </div>

            <div className="min-w-0 flex-1">
              <div className="text-sm font-bold text-slate-900">
                {toast.title}
              </div>
              {toast.message && (
                <div className="mt-1 text-xs leading-relaxed text-slate-500">
                  {toast.message}
                </div>
              )}
            </div>

            <button
              onClick={() => onRemove(toast.id)}
              className="text-slate-400 hover:text-slate-700"
            >
              ×
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}