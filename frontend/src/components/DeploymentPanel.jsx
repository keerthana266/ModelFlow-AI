import React, { useEffect, useMemo, useState } from "react";
import {
  CheckCircle2,
  Code2,
  ExternalLink,
  Loader2,
  Play,
  RefreshCcw,
  Rocket,
  Send,
  Wand2,
  XCircle,
} from "lucide-react";

import { api } from "../api";

export default function DeploymentPanel({
  models = [],
  deployments = [],
  onRefresh,
}) {
  const [selectedModelId, setSelectedModelId] = useState("");
  const [deploymentName, setDeploymentName] = useState("automl-model-api");
  const [environment, setEnvironment] = useState("production");
  const [activate, setActivate] = useState(true);

  const [selectedDeploymentId, setSelectedDeploymentId] = useState("");
  const [predictionText, setPredictionText] = useState("{}");

  const [modelDetails, setModelDetails] = useState(null);
  const [createResult, setCreateResult] = useState(null);
  const [testResult, setTestResult] = useState(null);
  const [error, setError] = useState("");

  const [isCreating, setIsCreating] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [isLoadingSchema, setIsLoadingSchema] = useState(false);

  const activeDeployments = useMemo(() => {
    return deployments.filter((deployment) => deployment.status === "active");
  }, [deployments]);

  const selectedDeployment = useMemo(() => {
    return deployments.find(
      (deployment) => deployment.deployment_id === selectedDeploymentId
    );
  }, [deployments, selectedDeploymentId]);

  useEffect(() => {
    if (!selectedDeploymentId && activeDeployments.length > 0) {
      setSelectedDeploymentId(activeDeployments[0].deployment_id);
    }
  }, [activeDeployments, selectedDeploymentId]);

  useEffect(() => {
    if (!selectedDeployment?.model_id) {
      setModelDetails(null);
      return;
    }

    loadModelSchema(selectedDeployment.model_id, false);
  }, [selectedDeployment?.model_id]);

  const loadModelSchema = async (modelId, shouldAutoFill = true) => {
    if (!modelId) return;

    try {
      setIsLoadingSchema(true);
      setError("");

      const response = await api.get(`/models/${modelId}`);
      const details = response.data;

      setModelDetails(details);

      if (shouldAutoFill) {
        const example =
          details.input_schema?.example ||
          buildFallbackExample(details.feature_columns || []);

        setPredictionText(
          JSON.stringify(
            {
              data: example,
            },
            null,
            2
          )
        );
      }
    } catch (error) {
      setError(getReadableError(error));
    } finally {
      setIsLoadingSchema(false);
    }
  };

  const handleAutoFill = async () => {
    setTestResult(null);

    if (!selectedDeployment) {
      setError("Please select a deployment first.");
      return;
    }

    if (!selectedDeployment.model_id) {
      setError(
        "This deployment does not include model_id. Update the backend /deployments response to include model_id."
      );
      return;
    }

    await loadModelSchema(selectedDeployment.model_id, true);
  };

  const handleCreateDeployment = async () => {
    if (!selectedModelId) {
      setError("Please select a model to deploy.");
      return;
    }

    try {
      setIsCreating(true);
      setError("");
      setCreateResult(null);

      const response = await api.post("/deployments", {
        model_id: selectedModelId,
        deployment_name: sanitizeDeploymentName(deploymentName),
        environment,
        activate,
      });

      setCreateResult(response.data);
      setSelectedDeploymentId(response.data.deployment_id);

      await onRefresh?.();
    } catch (error) {
      setError(getReadableError(error));
    } finally {
      setIsCreating(false);
    }
  };

  const handleTestDeployment = async () => {
    if (!selectedDeployment) {
      setError("Please select a deployment first.");
      return;
    }

    let payload;

    try {
      payload = JSON.parse(predictionText);
    } catch {
      setError("Invalid JSON input.");
      setTestResult(null);
      return;
    }

    try {
      setIsTesting(true);
      setError("");
      setTestResult(null);

      const endpoint = getDeploymentPredictEndpoint(selectedDeployment);

      const response = await api.post(endpoint, payload);

      setTestResult(response.data);
      await onRefresh?.();
    } catch (error) {
      setError(getReadableError(error));
    } finally {
      setIsTesting(false);
    }
  };

  const activeEndpoint = selectedDeployment
    ? selectedDeployment.versioned_endpoint_path ||
      `/deployments/name/${selectedDeployment.deployment_name}/versions/${selectedDeployment.version}/predict`
    : "";

  return (
    <section className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-100 px-4 py-4">
        <div className="flex items-center gap-2">
          <div className="grid h-9 w-9 place-items-center rounded-2xl bg-rose-50 text-rose-600">
            <Rocket size={17} />
          </div>

          <div className="min-w-0 flex-1">
            <h2 className="text-sm font-bold text-slate-900">
              Deployment Center
            </h2>
            <p className="text-xs text-slate-500">
              Deploy and test versioned REST APIs
            </p>
          </div>

          <button
            onClick={onRefresh}
            className="rounded-xl p-2 text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"
          >
            <RefreshCcw size={15} />
          </button>
        </div>
      </div>

      <div className="space-y-5 p-4">
        <div className="rounded-2xl border border-slate-100 bg-slate-50 p-3">
          <div className="mb-3 text-xs font-black uppercase tracking-wider text-slate-500">
            Create Deployment
          </div>

          <div className="space-y-3">
            <div>
              <label className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
                Model
              </label>

              <select
                value={selectedModelId}
                onChange={(event) => setSelectedModelId(event.target.value)}
                className="mt-1 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 outline-none transition focus:border-rose-400 focus:ring-4 focus:ring-rose-100"
              >
                <option value="">Select model</option>

                {models.map((model) => (
                  <option key={model.model_id} value={model.model_id}>
                    {model.model_name || model.model_id}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
                Deployment Name
              </label>

              <input
                value={deploymentName}
                onChange={(event) => setDeploymentName(event.target.value)}
                placeholder="churn-api"
                className="mt-1 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 outline-none transition focus:border-rose-400 focus:ring-4 focus:ring-rose-100"
              />
            </div>

            <div>
              <label className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
                Environment
              </label>

              <select
                value={environment}
                onChange={(event) => setEnvironment(event.target.value)}
                className="mt-1 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 outline-none transition focus:border-rose-400 focus:ring-4 focus:ring-rose-100"
              >
                <option value="development">Development</option>
                <option value="staging">Staging</option>
                <option value="production">Production</option>
              </select>
            </div>

            <label className="flex cursor-pointer items-center gap-2 rounded-2xl border border-slate-200 bg-white p-3 text-xs font-bold text-slate-600">
              <input
                type="checkbox"
                checked={activate}
                onChange={(event) => setActivate(event.target.checked)}
                className="h-4 w-4 accent-rose-600"
              />
              Activate after creation
            </label>

            <button
              onClick={handleCreateDeployment}
              disabled={isCreating || !selectedModelId}
              className={`flex w-full items-center justify-center gap-2 rounded-2xl px-4 py-3 text-sm font-bold text-white transition ${
                isCreating || !selectedModelId
                  ? "cursor-not-allowed bg-slate-400"
                  : "bg-rose-600 hover:bg-rose-700"
              }`}
            >
              {isCreating ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <Send size={16} />
              )}
              {isCreating ? "Creating..." : "Create Deployment"}
            </button>
          </div>
        </div>

        <div>
          <label className="text-xs font-black uppercase tracking-wider text-slate-500">
            Test Deployment
          </label>

          <select
            value={selectedDeploymentId}
            onChange={(event) => {
              setSelectedDeploymentId(event.target.value);
              setPredictionText("{}");
              setTestResult(null);
              setError("");
            }}
            className="mt-1 w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 outline-none transition focus:border-rose-400 focus:bg-white focus:ring-4 focus:ring-rose-100"
          >
            <option value="">Select deployment</option>

            {deployments.map((deployment) => (
              <option
                key={deployment.deployment_id}
                value={deployment.deployment_id}
              >
                {deployment.deployment_name} · v{deployment.version} ·{" "}
                {deployment.status}
              </option>
            ))}
          </select>
        </div>

        {selectedDeployment && (
          <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3">
            <div className="text-xs font-black uppercase tracking-wider text-slate-500">
              Active Endpoint
            </div>

            <div className="soft-scroll mt-2 overflow-auto rounded-2xl bg-white p-3 font-mono text-xs text-slate-700">
              {activeEndpoint}
            </div>

            <div className="mt-2 text-xs text-slate-600">
              Environment:{" "}
              <span className="font-black">
                {selectedDeployment.environment || "-"}
              </span>
            </div>

            {modelDetails && (
              <div className="mt-2 text-xs text-slate-600">
                Schema:{" "}
                <span className="font-black">
                  {modelDetails.feature_columns?.length || 0} features
                </span>
              </div>
            )}
          </div>
        )}

        <div>
          <div className="mb-1 flex items-center justify-between gap-2">
            <label className="flex items-center gap-2 text-xs font-black uppercase tracking-wider text-slate-500">
              <Code2 size={13} />
              Prediction Input JSON
            </label>

            <button
              onClick={handleAutoFill}
              disabled={!selectedDeployment || isLoadingSchema}
              className={`flex items-center gap-1 rounded-xl px-2.5 py-1.5 text-[11px] font-bold transition ${
                !selectedDeployment || isLoadingSchema
                  ? "cursor-not-allowed bg-slate-100 text-slate-300"
                  : "bg-slate-100 text-slate-600 hover:bg-slate-950 hover:text-white"
              }`}
            >
              {isLoadingSchema ? (
                <Loader2 size={12} className="animate-spin" />
              ) : (
                <Wand2 size={12} />
              )}
              Auto-fill
            </button>
          </div>

          <textarea
            value={predictionText}
            onChange={(event) => setPredictionText(event.target.value)}
            rows={8}
            spellCheck={false}
            className="soft-scroll w-full resize-none rounded-2xl border border-slate-200 bg-slate-950 p-3 font-mono text-xs leading-relaxed text-emerald-100 outline-none transition focus:border-rose-400 focus:ring-4 focus:ring-rose-100"
          />
        </div>

        <button
          onClick={handleTestDeployment}
          disabled={isTesting || !selectedDeployment}
          className={`flex w-full items-center justify-center gap-2 rounded-2xl px-4 py-3 text-sm font-bold text-white transition ${
            isTesting || !selectedDeployment
              ? "cursor-not-allowed bg-slate-400"
              : "bg-slate-950 hover:bg-slate-800"
          }`}
        >
          {isTesting ? (
            <Loader2 size={16} className="animate-spin" />
          ) : (
            <Play size={16} />
          )}
          {isTesting ? "Testing..." : "Test Deployment API"}
        </button>

        {error && <ErrorBox message={error} />}

        {createResult && (
          <ResultBox
            title="Deployment Created"
            result={createResult}
            tone="success"
          />
        )}

        {testResult && (
          <ResultBox title="Prediction Result" result={testResult} tone="dark" />
        )}

        <div>
          <div className="mb-2 text-xs font-black uppercase tracking-wider text-slate-500">
            Recent Deployments
          </div>

          <div className="space-y-2">
            {deployments.slice(0, 6).map((deployment) => (
              <button
                key={deployment.deployment_id}
                onClick={() => setSelectedDeploymentId(deployment.deployment_id)}
                className={`w-full rounded-2xl border p-3 text-left transition ${
                  selectedDeploymentId === deployment.deployment_id
                    ? "border-rose-300 bg-rose-50"
                    : "border-slate-200 bg-slate-50 hover:bg-white"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-black text-slate-900">
                      {deployment.deployment_name}
                    </div>

                    <div className="mt-0.5 text-xs text-slate-500">
                      v{deployment.version} · {deployment.environment} ·{" "}
                      {deployment.status}
                    </div>

                    <div className="mt-1 truncate font-mono text-[11px] text-slate-400">
                      {deployment.deployment_id}
                    </div>
                  </div>

                  <ExternalLink size={14} className="text-slate-300" />
                </div>
              </button>
            ))}

            {deployments.length === 0 && (
              <div className="rounded-2xl border border-dashed border-slate-200 p-4 text-center text-xs font-semibold text-slate-400">
                No deployments yet.
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

function ErrorBox({ message }) {
  return (
    <div className="rounded-2xl border border-red-100 bg-red-50 p-3">
      <div className="flex items-center gap-2 text-sm font-black text-red-700">
        <XCircle size={16} />
        Error
      </div>

      <p className="mt-1 whitespace-pre-wrap text-xs leading-relaxed text-red-600">
        {message}
      </p>
    </div>
  );
}

function ResultBox({ title, result, tone }) {
  const isSuccess = tone === "success";

  return (
    <div
      className={`rounded-2xl border p-3 ${
        isSuccess
          ? "border-emerald-100 bg-emerald-50"
          : "border-slate-200 bg-slate-50"
      }`}
    >
      <div
        className={`flex items-center gap-2 text-sm font-black ${
          isSuccess ? "text-emerald-700" : "text-slate-800"
        }`}
      >
        <CheckCircle2 size={16} />
        {title}
      </div>

      <pre className="soft-scroll mt-3 max-h-64 overflow-auto rounded-2xl bg-slate-950 p-3 text-[11px] leading-relaxed text-emerald-100">
        {JSON.stringify(result, null, 2)}
      </pre>
    </div>
  );
}

function getDeploymentPredictEndpoint(deployment) {
  if (deployment.versioned_endpoint_path) {
    return deployment.versioned_endpoint_path;
  }

  if (deployment.deployment_name && deployment.version) {
    return `/deployments/name/${encodeURIComponent(
      deployment.deployment_name
    )}/versions/${deployment.version}/predict`;
  }

  return `/deployments/${deployment.deployment_id}/predict`;
}

function sanitizeDeploymentName(value) {
  const cleaned = String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

  return cleaned || "automl-model-api";
}

function getReadableError(error) {
  const detail = error.response?.data?.detail;

  if (typeof detail === "string") {
    return detail;
  }

  if (detail && typeof detail === "object") {
    const lines = [];

    if (detail.message) {
      lines.push(detail.message);
    }

    if (Array.isArray(detail.errors)) {
      lines.push(...detail.errors);
    }

    if (Array.isArray(detail.warnings) && detail.warnings.length > 0) {
      lines.push("Warnings:");
      lines.push(...detail.warnings);
    }

    return lines.join("\n") || JSON.stringify(detail, null, 2);
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
    } else if (key.includes("area")) {
      example[column] = 1200;
    } else if (key.includes("bedroom")) {
      example[column] = 3;
    } else if (key.includes("bathroom")) {
      example[column] = 2;
    } else if (key.includes("fare")) {
      example[column] = 50;
    } else if (key.includes("plan")) {
      example[column] = "Premium";
    } else if (key.includes("region")) {
      example[column] = "East";
    } else if (key.includes("sex") || key.includes("gender")) {
      example[column] = "female";
    } else if (key.includes("embarked")) {
      example[column] = "S";
    } else if (key.includes("location")) {
      example[column] = "Urban";
    } else {
      example[column] = "sample";
    }
  });

  return example;
}