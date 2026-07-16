import React, { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  ClipboardCheck,
  Loader2,
  RefreshCcw,
  ShieldCheck,
  Sparkles,
  X,
} from "lucide-react";

import { api } from "../api";


function getErrorMessage(error) {
  const detail = error.response?.data?.detail;
  if (typeof detail === "string") return detail;
  return error.message || "Could not check the dataset.";
}


function statusTheme(status) {
  if (status === "ready") {
    return {
      border: "border-emerald-200",
      background: "bg-emerald-50",
      text: "text-emerald-700",
      icon: CheckCircle2,
      label: "Ready",
    };
  }

  if (status === "review") {
    return {
      border: "border-amber-200",
      background: "bg-amber-50",
      text: "text-amber-700",
      icon: AlertTriangle,
      label: "Quick review",
    };
  }

  return {
    border: "border-red-200",
    background: "bg-red-50",
    text: "text-red-700",
    icon: AlertTriangle,
    label: "Needs attention",
  };
}


export default function DatasetReadinessPanel({
  datasetId,
  targetColumn,
  excludedColumns = [],
  removeDuplicates = false,
  onExcludedColumnsChange,
  onRemoveDuplicatesChange,
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [payload, setPayload] = useState(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const loadReadiness = async () => {
    if (!datasetId) {
      setPayload(null);
      setError("");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await api.get(`/datasets/${datasetId}/readiness`, {
        params: targetColumn ? { target_column: targetColumn } : {},
      });
      setPayload(response.data);
    } catch (requestError) {
      setPayload(null);
      setError(getErrorMessage(requestError));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadReadiness();
  }, [datasetId, targetColumn]);

  // A column may have been excluded before the user selected it as the target.
  // Remove that stale exclusion immediately so training can never lose the
  // selected output column.
  useEffect(() => {
    if (!targetColumn || !excludedColumns.includes(targetColumn)) return;

    onExcludedColumnsChange?.(
      excludedColumns.filter((column) => column !== targetColumn)
    );
  }, [targetColumn, excludedColumns, onExcludedColumnsChange]);

  const readiness = payload?.readiness || null;
  const theme = statusTheme(readiness?.status);
  const StatusIcon = theme.icon;

  const excludedSet = useMemo(
    () => new Set(excludedColumns || []),
    [excludedColumns]
  );

  const safeDecisions = useMemo(
    () =>
      (readiness?.decisions || []).filter(
        (decision) =>
          decision.action !== "exclude_column" ||
          decision.column !== targetColumn
      ),
    [readiness, targetColumn]
  );

  const toggleColumn = (column) => {
    if (!column || column === targetColumn) return;

    const next = new Set(excludedColumns || []);
    if (next.has(column)) next.delete(column);
    else next.add(column);
    onExcludedColumnsChange?.([...next]);
  };

  const decisionApplied = (decision) => {
    if (decision.action === "exclude_column") {
      return excludedSet.has(decision.column);
    }
    if (decision.action === "remove_duplicates") {
      return removeDuplicates;
    }
    return false;
  };

  const applyDecision = (decision) => {
    if (decision.action === "exclude_column") {
      toggleColumn(decision.column);
    }
    if (decision.action === "remove_duplicates") {
      onRemoveDuplicatesChange?.(!removeDuplicates);
    }
  };

  return (
    <section className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
      <div className="flex items-center justify-between border-b border-slate-100 px-4 py-4">
        <div className="flex items-center gap-3">
          <div className="grid h-10 w-10 place-items-center rounded-2xl bg-emerald-50 text-emerald-600">
            <ClipboardCheck size={18} />
          </div>
          <div>
            <h2 className="text-sm font-black text-slate-900">
              Dataset Readiness
            </h2>
            <p className="text-xs text-slate-500">
              Simple checks, automatic preparation
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={loadReadiness}
          disabled={!datasetId || loading}
          className="rounded-xl p-2 text-slate-400 transition hover:bg-slate-100 hover:text-emerald-600 disabled:cursor-not-allowed disabled:opacity-40"
          title="Check again"
        >
          <RefreshCcw size={15} className={loading ? "animate-spin" : ""} />
        </button>
      </div>

      <div className="space-y-4 p-4">
        {!datasetId && (
          <div className="rounded-2xl border border-dashed border-slate-200 p-5 text-center text-xs leading-relaxed text-slate-500">
            Select a dataset to see whether it is ready for training.
          </div>
        )}

        {datasetId && loading && (
          <div className="flex items-center justify-center gap-2 rounded-2xl bg-slate-50 p-5 text-xs font-bold text-slate-500">
            <Loader2 size={15} className="animate-spin" />
            Checking the dataset…
          </div>
        )}

        {error && (
          <div className="rounded-2xl border border-red-200 bg-red-50 p-4 text-xs text-red-700">
            {error}
          </div>
        )}

        {readiness && !loading && (
          <>
            <div
              className={`rounded-2xl border p-4 ${theme.border} ${theme.background}`}
            >
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-3">
                  <StatusIcon size={20} className={theme.text} />
                  <div>
                    <div className={`text-sm font-black ${theme.text}`}>
                      {theme.label}
                    </div>
                    <div className="mt-0.5 text-xs text-slate-600">
                      {readiness.summary}
                    </div>
                  </div>
                </div>

                <div className="text-right">
                  <div className={`text-2xl font-black ${theme.text}`}>
                    {readiness.score}
                  </div>
                  <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                    out of 100
                  </div>
                </div>
              </div>
            </div>

            {!targetColumn && (
              <div className="rounded-2xl border border-blue-200 bg-blue-50 p-3 text-xs leading-relaxed text-blue-700">
                Choose the result column in the Data Loader. Then this assistant
                will finish the readiness check.
              </div>
            )}

            {readiness.blockers?.length > 0 && (
              <div className="space-y-2">
                <SectionLabel icon={<AlertTriangle size={13} />}>
                  Fix before training
                </SectionLabel>
                {readiness.blockers.map((item) => (
                  <InfoCard key={item.code} tone="danger" item={item} />
                ))}
              </div>
            )}

            {readiness.auto_actions?.length > 0 && (
              <div className="space-y-2">
                <SectionLabel icon={<Sparkles size={13} />}>
                  Handled automatically
                </SectionLabel>
                {readiness.auto_actions.map((item) => (
                  <div
                    key={item.code}
                    className="flex gap-3 rounded-2xl border border-emerald-100 bg-emerald-50/60 p-3"
                  >
                    <div className="mt-0.5 grid h-5 w-5 flex-shrink-0 place-items-center rounded-full bg-emerald-600 text-white">
                      <Check size={12} strokeWidth={3} />
                    </div>
                    <div>
                      <div className="text-xs font-black text-emerald-800">
                        {item.title}
                      </div>
                      <p className="mt-1 text-xs leading-relaxed text-slate-600">
                        {item.message}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {safeDecisions.length > 0 && (
              <div className="space-y-2">
                <SectionLabel icon={<ShieldCheck size={13} />}>
                  Your decision
                </SectionLabel>
                {safeDecisions.map((decision) => {
                  const applied = decisionApplied(decision);
                  return (
                    <div
                      key={`${decision.code}-${decision.column || "all"}`}
                      className="rounded-2xl border border-amber-200 bg-amber-50/60 p-3"
                    >
                      <div className="text-xs font-black text-slate-800">
                        {decision.title}
                      </div>
                      <p className="mt-1 text-xs leading-relaxed text-slate-600">
                        {decision.message}
                      </p>
                      <button
                        type="button"
                        onClick={() => applyDecision(decision)}
                        className={`mt-3 flex w-full items-center justify-center gap-2 rounded-xl px-3 py-2 text-xs font-black transition ${
                          applied
                            ? "bg-emerald-600 text-white hover:bg-emerald-700"
                            : "bg-slate-950 text-white hover:bg-slate-800"
                        }`}
                      >
                        {applied ? <Check size={13} /> : <X size={13} />}
                        {applied ? "Recommendation applied" : "Apply recommendation"}
                      </button>
                    </div>
                  );
                })}
              </div>
            )}

            {readiness.notes?.length > 0 && (
              <div className="space-y-2">
                {readiness.notes.map((item) => (
                  <InfoCard key={item.code} tone="info" item={item} />
                ))}
              </div>
            )}

            <button
              type="button"
              onClick={() => setShowAdvanced((current) => !current)}
              className="flex w-full items-center justify-between rounded-2xl bg-slate-50 px-3 py-3 text-xs font-bold text-slate-600 transition hover:bg-slate-100"
            >
              <span>Advanced details</span>
              {showAdvanced ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>

            {showAdvanced && (
              <pre className="soft-scroll max-h-72 overflow-auto rounded-2xl bg-slate-950 p-3 text-[10px] leading-relaxed text-slate-200">
                {JSON.stringify(readiness.advanced, null, 2)}
              </pre>
            )}

            <div
              className={`rounded-2xl px-4 py-3 text-center text-xs font-black ${
                readiness.ready_to_train
                  ? "bg-emerald-600 text-white"
                  : "bg-slate-100 text-slate-500"
              }`}
            >
              {readiness.ready_to_train
                ? "Ready to continue to training"
                : "Complete the highlighted step before training"}
            </div>
          </>
        )}
      </div>
    </section>
  );
}


function SectionLabel({ icon, children }) {
  return (
    <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-wider text-slate-400">
      {icon}
      {children}
    </div>
  );
}


function InfoCard({ item, tone }) {
  const className =
    tone === "danger"
      ? "border-red-200 bg-red-50 text-red-700"
      : "border-blue-100 bg-blue-50 text-blue-700";

  return (
    <div className={`rounded-2xl border p-3 ${className}`}>
      <div className="text-xs font-black">{item.title}</div>
      <p className="mt-1 text-xs leading-relaxed opacity-90">{item.message}</p>
    </div>
  );
}
