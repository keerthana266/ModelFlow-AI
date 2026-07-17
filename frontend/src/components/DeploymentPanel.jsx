import React, { useEffect, useMemo, useState } from "react";
import {
  Braces,
  Check,
  CircleStop,
  Clipboard,
  Code2,
  Container,
  ExternalLink,
  FileText,
  HeartPulse,
  Info,
  Lightbulb,
  Minus,
  Loader2,
  Play,
  Plus,
  RefreshCcw,
  RotateCw,
  Send,
  Sparkles,
  TrendingDown,
  TrendingUp,
  Trash2,
  X,
} from "lucide-react";

import { API_BASE_URL, api } from "../api";

const TRANSITIONAL = new Set(["pending", "building", "starting"]);
const CODE_TABS = ["curl", "python", "javascript"];

function errorMessage(error) {
  const detail = error.response?.data?.detail;

  if (typeof detail === "string") return detail;

  if (detail && typeof detail === "object") {
    if (detail.message) {
      const errors = Array.isArray(detail.errors)
        ? ` ${detail.errors.join(" ")}`
        : "";
      return `${detail.message}${errors}`;
    }

    return JSON.stringify(detail);
  }

  return error.message || "Request failed.";
}

function prettifyName(value) {
  return String(value || "")
    .replace(/_/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function inferFieldType(fieldName, modelDetails, property = {}) {
  const numericalColumns = new Set(modelDetails?.numerical_columns || []);
  const categoricalColumns = new Set(modelDetails?.categorical_columns || []);
  const declaredType = property.type || property.data_type || property.dtype;

  if (
    declaredType === "number" ||
    declaredType === "integer" ||
    numericalColumns.has(fieldName)
  ) {
    return "number";
  }

  if (declaredType === "boolean") return "boolean";
  if (categoricalColumns.has(fieldName)) return "text";

  return "text";
}

function normalizeInputFields(modelDetails) {
  const schema = modelDetails?.input_schema || {};

  if (Array.isArray(schema.fields)) {
    return schema.fields.map((field) => ({
      name: field.name || field.column || field.key,
      type: inferFieldType(
        field.name || field.column || field.key,
        modelDetails,
        field
      ),
      required: field.required !== false,
      description: field.description || "",
      options: field.options || field.choices || null,
      example: field.example ?? field.default ?? "",
    }));
  }

  if (schema.properties && typeof schema.properties === "object") {
    const required = new Set(schema.required || []);

    return Object.entries(schema.properties).map(([name, property]) => ({
      name,
      type: inferFieldType(name, modelDetails, property || {}),
      required: required.has(name),
      description: property?.description || "",
      options: property?.enum || property?.options || null,
      example: property?.example ?? property?.default ?? "",
    }));
  }

  const schemaColumns =
    schema.feature_columns || schema.columns || schema.required_columns || [];

  const featureColumns =
    schemaColumns.length > 0
      ? schemaColumns
      : modelDetails?.feature_columns || [];

  return featureColumns.map((name) => ({
    name,
    type: inferFieldType(name, modelDetails),
    required: true,
    description: "",
    options: null,
    example: "",
  }));
}

function createInitialValues(fields) {
  return fields.reduce((values, field) => {
    if (field.example !== "" && field.example !== undefined) {
      values[field.name] = String(field.example);
    } else if (field.type === "boolean") {
      values[field.name] = "false";
    } else {
      values[field.name] = "";
    }

    return values;
  }, {});
}

function coerceValue(rawValue, field) {
  if (field.type === "number") {
    if (rawValue === "") return null;

    const numericValue = Number(rawValue);

    if (!Number.isFinite(numericValue)) {
      throw new Error(`${prettifyName(field.name)} must be a valid number.`);
    }

    return numericValue;
  }

  if (field.type === "boolean") {
    return rawValue === true || rawValue === "true";
  }

  return rawValue;
}

function buildPredictionPayload(fields, values) {
  const row = {};

  fields.forEach((field) => {
    const rawValue = values[field.name];

    if (field.required && (rawValue === "" || rawValue === undefined)) {
      throw new Error(`${prettifyName(field.name)} is required.`);
    }

    if (!field.required && (rawValue === "" || rawValue === undefined)) {
      return;
    }

    row[field.name] = coerceValue(rawValue, field);
  });

  return { data: row };
}

function createExamplePayload(fields, values) {
  const row = {};

  fields.forEach((field) => {
    const value = values[field.name];

    if (value !== "" && value !== undefined) {
      try {
        row[field.name] = coerceValue(value, field);
        return;
      } catch {
        // Fall through to a safe example value.
      }
    }

    if (field.options?.length) {
      row[field.name] = field.options[0];
    } else if (field.type === "number") {
      row[field.name] = 0;
    } else if (field.type === "boolean") {
      row[field.name] = false;
    } else {
      row[field.name] = `<${field.name}>`;
    }
  });

  return { data: row };
}

function integrationSnippet(language, endpointUrl, payload) {
  const jsonPayload = JSON.stringify(payload, null, 2);

  if (language === "python") {
    return `import json\nimport requests\n\nurl = "${endpointUrl}"\nheaders = {\n    "Authorization": "Bearer <ACCESS_TOKEN>",\n    "X-Project-ID": "<PROJECT_ID>",\n    "Content-Type": "application/json",\n}\npayload = json.loads(r'''${jsonPayload}''')\n\nresponse = requests.post(url, headers=headers, json=payload, timeout=30)\nresponse.raise_for_status()\nprint(response.json())`;
  }

  if (language === "javascript") {
    return `const response = await fetch("${endpointUrl}", {\n  method: "POST",\n  headers: {\n    Authorization: "Bearer <ACCESS_TOKEN>",\n    "X-Project-ID": "<PROJECT_ID>",\n    "Content-Type": "application/json",\n  },\n  body: JSON.stringify(${jsonPayload}),\n});\n\nif (!response.ok) {\n  throw new Error(await response.text());\n}\n\nconst result = await response.json();\nconsole.log(result);`;
  }

  const compactPayload = JSON.stringify(payload).replace(/'/g, "'\\''");

  return `curl -X POST "${endpointUrl}" \\\n  -H "Authorization: Bearer <ACCESS_TOKEN>" \\\n  -H "X-Project-ID: <PROJECT_ID>" \\\n  -H "Content-Type: application/json" \\\n  -d '${compactPayload}'`;
}

async function copyText(value) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = value;
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  document.body.removeChild(textarea);
}

export default function DeploymentPanel({ models, deployments, onRefresh }) {
  const [modelId, setModelId] = useState("");
  const [name, setName] = useState("automl-model-api");
  const [environment, setEnvironment] = useState("production");
  const [busy, setBusy] = useState("");
  const [message, setMessage] = useState("");
  const [logs, setLogs] = useState(null);
  const [playground, setPlayground] = useState(null);
  const [integration, setIntegration] = useState(null);

  const hasTransitionalDeployment = useMemo(
    () => deployments.some((item) => TRANSITIONAL.has(item.status)),
    [deployments]
  );

  useEffect(() => {
    if (!hasTransitionalDeployment) return undefined;

    const interval = window.setInterval(() => {
      onRefresh?.();
    }, 2500);

    return () => window.clearInterval(interval);
  }, [hasTransitionalDeployment, onRefresh]);

  useEffect(() => {
    if (!modelId && models[0]) setModelId(models[0].model_id);
  }, [modelId, models]);

  async function createDeployment() {
    if (!modelId) return;
    setBusy("create");
    setMessage("");

    try {
      const response = await api.post("/deployments", {
        model_id: modelId,
        deployment_name: name.trim() || "automl-model-api",
        environment,
        activate: true,
      });

      setMessage(
        `Queued ${response.data.deployment_name} v${response.data.version}.`
      );
      await onRefresh?.();
    } catch (error) {
      setMessage(errorMessage(error));
    } finally {
      setBusy("");
    }
  }

  async function action(deploymentId, actionName, method = "post") {
    setBusy(`${deploymentId}:${actionName}`);
    setMessage("");

    try {
      await api.request({
        method,
        url: `/deployments/${deploymentId}${
          actionName === "delete" ? "" : `/${actionName}`
        }`,
      });
      await onRefresh?.();
    } catch (error) {
      setMessage(errorMessage(error));
    } finally {
      setBusy("");
    }
  }

  async function showLogs(deploymentId) {
    setBusy(`${deploymentId}:logs`);

    try {
      const response = await api.get(`/deployments/${deploymentId}/logs`);
      setLogs({ deploymentId, value: response.data.logs || "No logs." });
    } catch (error) {
      setMessage(errorMessage(error));
    } finally {
      setBusy("");
    }
  }

  async function checkHealth(deploymentId) {
    setBusy(`${deploymentId}:health`);

    try {
      const response = await api.get(`/deployments/${deploymentId}/health`);
      setLogs({
        deploymentId,
        value: JSON.stringify(response.data, null, 2),
      });
    } catch (error) {
      setMessage(errorMessage(error));
    } finally {
      setBusy("");
    }
  }

  async function loadDeploymentUsage(deployment, mode) {
    setBusy(`${deployment.deployment_id}:${mode}`);
    setMessage("");

    try {
      const response = await api.get(`/models/${deployment.model_id}`);
      const modelDetails = response.data;
      const fields = normalizeInputFields(modelDetails);

      if (fields.length === 0) {
        throw new Error("No input fields were found for this model.");
      }

      const state = {
        deployment,
        modelDetails,
        fields,
        values: createInitialValues(fields),
      };

      if (mode === "playground") {
        setPlayground({
          ...state,
          prediction: null,
          error: "",
          submitting: false,
        });
      } else {
        setIntegration({
          ...state,
          language: "curl",
          copied: "",
        });
      }
    } catch (error) {
      setMessage(errorMessage(error));
    } finally {
      setBusy("");
    }
  }

  return (
    <>
      <section className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
        <header className="flex items-center justify-between border-b border-slate-100 px-4 py-4">
          <div className="flex items-center gap-3">
            <div className="grid h-9 w-9 place-items-center rounded-2xl bg-blue-50 text-blue-600">
              <Container size={17} />
            </div>
            <div>
              <h2 className="text-sm font-black text-slate-900">
                Docker Deployments
              </h2>
              <p className="text-xs text-slate-500">
                One isolated container per model
              </p>
            </div>
          </div>
          <button
            onClick={onRefresh}
            className="rounded-xl p-2 text-slate-400 hover:bg-slate-100 hover:text-blue-600"
          >
            <RefreshCcw size={15} />
          </button>
        </header>

        <div className="space-y-3 border-b border-slate-100 p-4">
          <select
            value={modelId}
            onChange={(event) => setModelId(event.target.value)}
            className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs"
          >
            <option value="">Select trained model</option>
            {models.map((model) => (
              <option key={model.model_id} value={model.model_id}>
                {model.model_name}
              </option>
            ))}
          </select>

          <input
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="deployment-name"
            className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs"
          />

          <select
            value={environment}
            onChange={(event) => setEnvironment(event.target.value)}
            className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs"
          >
            <option value="development">Development</option>
            <option value="staging">Staging</option>
            <option value="production">Production</option>
          </select>

          <button
            onClick={createDeployment}
            disabled={!modelId || busy === "create"}
            className="flex w-full items-center justify-center gap-2 rounded-2xl bg-slate-950 px-3 py-2.5 text-xs font-bold text-white disabled:opacity-50"
          >
            {busy === "create" ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <Plus size={14} />
            )}
            Deploy in Docker
          </button>

          {message && (
            <div className="rounded-xl bg-slate-50 p-2 text-[11px] text-slate-600">
              {message}
            </div>
          )}
        </div>

        <div className="soft-scroll max-h-[520px] space-y-3 overflow-auto p-4">
          {deployments.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-slate-200 p-5 text-center text-xs text-slate-400">
              No Docker deployments yet.
            </div>
          ) : (
            deployments.map((deployment) => {
              const prefix = `${deployment.deployment_id}:`;
              const isBusy = busy.startsWith(prefix);
              const canUse = deployment.status === "active";

              return (
                <article
                  key={deployment.deployment_id}
                  className="rounded-2xl border border-slate-200 bg-slate-50 p-3"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <div className="truncate text-xs font-black text-slate-900">
                        {deployment.deployment_name} v{deployment.version}
                      </div>
                      <div className="mt-1 truncate text-[10px] text-slate-500">
                        {deployment.environment} · {deployment.model_id}
                      </div>
                    </div>
                    <StatusBadge status={deployment.status} />
                  </div>

                  <div className="mt-2 rounded-xl bg-white p-2 text-[10px] text-slate-500">
                    {deployment.host_port
                      ? `${deployment.host}:${deployment.host_port}`
                      : "Waiting for container port"}
                  </div>

                  {deployment.error_message && (
                    <div className="mt-2 rounded-xl bg-red-50 p-2 text-[10px] text-red-700">
                      {deployment.error_message}
                    </div>
                  )}

                  <div className="mt-3 grid grid-cols-2 gap-2">
                    <button
                      onClick={() =>
                        loadDeploymentUsage(deployment, "playground")
                      }
                      disabled={isBusy || !canUse}
                      className="flex items-center justify-center gap-1.5 rounded-xl bg-blue-600 px-2 py-2.5 text-[10px] font-black text-white shadow-sm disabled:opacity-35"
                    >
                      <Sparkles size={12} />
                      Use Model
                    </button>

                    <button
                      onClick={() =>
                        loadDeploymentUsage(deployment, "integration")
                      }
                      disabled={isBusy || !canUse}
                      className="flex items-center justify-center gap-1.5 rounded-xl bg-slate-900 px-2 py-2.5 text-[10px] font-black text-white shadow-sm disabled:opacity-35"
                    >
                      <Code2 size={12} />
                      Integrate
                    </button>
                  </div>

                  <div className="mt-2 grid grid-cols-3 gap-1.5">
                    <ActionButton
                      title="Start"
                      icon={<Play size={12} />}
                      disabled={isBusy || deployment.status === "active"}
                      onClick={() => action(deployment.deployment_id, "start")}
                    />
                    <ActionButton
                      title="Stop"
                      icon={<CircleStop size={12} />}
                      disabled={isBusy || !deployment.container_id}
                      onClick={() => action(deployment.deployment_id, "stop")}
                    />
                    <ActionButton
                      title="Restart"
                      icon={<RotateCw size={12} />}
                      disabled={isBusy || !deployment.container_id}
                      onClick={() =>
                        action(deployment.deployment_id, "restart")
                      }
                    />
                    <ActionButton
                      title="Health"
                      icon={<HeartPulse size={12} />}
                      disabled={isBusy}
                      onClick={() => checkHealth(deployment.deployment_id)}
                    />
                    <ActionButton
                      title="Logs"
                      icon={<FileText size={12} />}
                      disabled={isBusy}
                      onClick={() => showLogs(deployment.deployment_id)}
                    />
                    <ActionButton
                      title="Archive"
                      icon={<Trash2 size={12} />}
                      disabled={isBusy || deployment.status === "archived"}
                      onClick={() =>
                        action(deployment.deployment_id, "delete", "delete")
                      }
                    />
                  </div>

                  {isBusy && (
                    <div className="mt-2 flex items-center gap-1 text-[10px] text-blue-600">
                      <Loader2 size={11} className="animate-spin" /> Processing
                    </div>
                  )}
                </article>
              );
            })
          )}
        </div>

        {logs && (
          <div className="border-t border-slate-100 p-4">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
                Runtime output
              </span>
              <button
                onClick={() => setLogs(null)}
                className="text-[10px] font-bold text-slate-500"
              >
                Close
              </button>
            </div>
            <pre className="soft-scroll max-h-48 overflow-auto whitespace-pre-wrap rounded-xl bg-slate-950 p-3 text-[10px] text-slate-200">
              {logs.value}
            </pre>
          </div>
        )}
      </section>

      {playground && (
        <PredictionPlayground
          state={playground}
          setState={setPlayground}
          onClose={() => setPlayground(null)}
          onDone={onRefresh}
        />
      )}

      {integration && (
        <IntegrationModal
          state={integration}
          setState={setIntegration}
          onClose={() => setIntegration(null)}
        />
      )}
    </>
  );
}

function PredictionPlayground({ state, setState, onClose, onDone }) {
  const { deployment, modelDetails, fields, values } = state;

  async function submitPrediction(event) {
    event.preventDefault();

    setState((current) => ({
      ...current,
      submitting: true,
      prediction: null,
      error: "",
    }));

    try {
      const payload = buildPredictionPayload(fields, values);
      const response = await api.post(
        `/deployments/${deployment.deployment_id}/predict`,
        payload
      );

      setState((current) => ({
        ...current,
        submitting: false,
        prediction: response.data,
        error: "",
      }));

      await onDone?.();
    } catch (error) {
      setState((current) => ({
        ...current,
        submitting: false,
        prediction: null,
        error: errorMessage(error),
      }));
    }
  }

  function updateValue(name, value) {
    setState((current) => ({
      ...current,
      values: {
        ...current.values,
        [name]: value,
      },
    }));
  }

  return (
    <ModalShell
      title="Use deployed model"
      subtitle={`${deployment.deployment_name} v${deployment.version}`}
      icon={<Sparkles size={18} />}
      onClose={onClose}
    >
      <div className="grid gap-5 lg:grid-cols-[1.1fr_0.9fr]">
        <form onSubmit={submitPrediction} className="space-y-4">
          <div className="rounded-2xl bg-blue-50 p-4 text-xs text-blue-800">
            Enter real business values. AutoML Forge sends them to the isolated
            Docker model and returns its prediction.
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            {fields.map((field) => (
              <FieldInput
                key={field.name}
                field={field}
                value={values[field.name] ?? ""}
                onChange={(value) => updateValue(field.name, value)}
              />
            ))}
          </div>

          {state.error && (
            <div className="rounded-2xl bg-red-50 p-3 text-xs text-red-700">
              {state.error}
            </div>
          )}

          <button
            type="submit"
            disabled={state.submitting}
            className="flex w-full items-center justify-center gap-2 rounded-2xl bg-blue-600 px-4 py-3 text-sm font-black text-white disabled:opacity-50"
          >
            {state.submitting ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <Send size={16} />
            )}
            {state.submitting ? "Predicting..." : "Get Prediction"}
          </button>
        </form>

        <div className="space-y-4">
          <div className="rounded-2xl border border-slate-200 p-4">
            <div className="text-[10px] font-black uppercase tracking-wider text-slate-400">
              Model
            </div>
            <div className="mt-1 text-sm font-black text-slate-900">
              {modelDetails.model_name}
            </div>
            <div className="mt-2 text-xs text-slate-500">
              {modelDetails.problem_type} · predicts {modelDetails.target_column}
            </div>
          </div>

          {state.prediction ? (
            <PredictionResult
              payload={state.prediction}
              targetColumn={modelDetails.target_column}
              problemType={modelDetails.problem_type}
            />
          ) : (
            <div className="min-h-56 rounded-2xl bg-slate-950 p-4 text-white">
              <div className="flex items-center gap-2 text-xs font-black">
                <Braces size={14} /> Prediction result
              </div>
              <div className="mt-8 text-center text-xs text-slate-400">
                Submit the form to see the prediction and its explanation.
              </div>
            </div>
          )}
        </div>
      </div>
    </ModalShell>
  );
}


function formatResultValue(value) {
  if (typeof value === "number") {
    return new Intl.NumberFormat(undefined, {
      maximumFractionDigits: 4,
    }).format(value);
  }

  if (Array.isArray(value)) {
    return value.length === 1
      ? formatResultValue(value[0])
      : JSON.stringify(value);
  }

  if (value && typeof value === "object") {
    return JSON.stringify(value);
  }

  return String(value ?? "—");
}

function PredictionResult({ payload, targetColumn, problemType }) {
  const explanation = payload?.explanation;
  const factors = explanation?.factors || [];
  const prediction = payload?.prediction;

  return (
    <div className="space-y-4">
      <div className="rounded-2xl bg-slate-950 p-5 text-white">
        <div className="text-[10px] font-black uppercase tracking-[0.18em] text-slate-400">
          Predicted {prettifyName(targetColumn)}
        </div>
        <div className="mt-2 break-words text-3xl font-black tracking-tight text-emerald-300">
          {formatResultValue(prediction)}
        </div>
        {problemType === "classification" &&
          explanation?.prediction_details?.predicted_probability != null && (
            <div className="mt-2 text-xs text-slate-300">
              Confidence for this result: {Math.round(
                explanation.prediction_details.predicted_probability * 1000
              ) / 10}
              %
            </div>
          )}
      </div>

      {explanation?.available ? (
        <div className="rounded-2xl border border-blue-100 bg-blue-50/60 p-4">
          <div className="flex items-start gap-3">
            <div className="grid h-9 w-9 flex-shrink-0 place-items-center rounded-xl bg-blue-600 text-white">
              <Lightbulb size={16} />
            </div>
            <div>
              <div className="text-sm font-black text-slate-900">
                {explanation.title || "Why this result was produced"}
              </div>
              <p className="mt-1 text-xs leading-relaxed text-slate-600">
                {explanation.summary}
              </p>
            </div>
          </div>

          <div className="mt-4 space-y-2.5">
            {factors.map((factor) => {
              const isIncrease = factor.direction === "increase";
              const isDecrease = factor.direction === "decrease";
              const Icon = isIncrease
                ? TrendingUp
                : isDecrease
                ? TrendingDown
                : Minus;
              const iconClass = isIncrease
                ? "bg-emerald-100 text-emerald-700"
                : isDecrease
                ? "bg-amber-100 text-amber-700"
                : "bg-slate-100 text-slate-500";

              return (
                <div
                  key={factor.feature}
                  className="rounded-2xl border border-white bg-white p-3 shadow-sm"
                >
                  <div className="flex items-start gap-3">
                    <div
                      className={`grid h-8 w-8 flex-shrink-0 place-items-center rounded-xl ${iconClass}`}
                    >
                      <Icon size={14} />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center justify-between gap-3">
                        <span className="truncate text-xs font-black text-slate-900">
                          {prettifyName(factor.feature)}
                        </span>
                        <span className="text-[10px] font-bold text-slate-400">
                          entered: {formatResultValue(factor.input_value)}
                        </span>
                      </div>
                      <p className="mt-1 text-[11px] leading-relaxed text-slate-600">
                        {factor.message}
                      </p>
                      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-100">
                        <div
                          className={`h-full rounded-full ${
                            isIncrease
                              ? "bg-emerald-500"
                              : isDecrease
                              ? "bg-amber-500"
                              : "bg-slate-300"
                          }`}
                          style={{
                            width: `${Math.max(
                              4,
                              Math.round((factor.relative_strength || 0) * 100)
                            )}%`,
                          }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          <details className="mt-4 rounded-xl bg-white/70 p-3 text-[10px] text-slate-500">
            <summary className="cursor-pointer font-black text-slate-600">
              How this explanation was calculated
            </summary>
            <p className="mt-2 leading-relaxed">{explanation.note}</p>
            <div className="mt-2">
              Method: {explanation.method} · evaluated {explanation.features_evaluated}
              {explanation.features_total
                ? ` of ${explanation.features_total}`
                : ""} inputs
            </div>
          </details>
        </div>
      ) : (
        <div className="flex items-start gap-2 rounded-2xl border border-slate-200 bg-slate-50 p-4 text-xs text-slate-600">
          <Info size={15} className="mt-0.5 flex-shrink-0" />
          <div>
            <div className="font-black text-slate-800">Prediction completed</div>
            <p className="mt-1 leading-relaxed">
              {explanation?.reason ||
                "An explanation is not available for this saved model."}
            </p>
          </div>
        </div>
      )}

      <details className="rounded-2xl border border-slate-200 bg-white p-3">
        <summary className="cursor-pointer text-[10px] font-black uppercase tracking-wider text-slate-500">
          Technical response
        </summary>
        <pre className="soft-scroll mt-3 max-h-64 overflow-auto whitespace-pre-wrap rounded-xl bg-slate-950 p-3 text-[10px] text-emerald-300">
          {JSON.stringify(payload, null, 2)}
        </pre>
      </details>
    </div>
  );
}

function IntegrationModal({ state, setState, onClose }) {
  const { deployment, fields, values, language, copied } = state;
  const endpointPath =
    deployment.endpoint_path ||
    `/deployments/${deployment.deployment_id}/predict`;
  const endpointUrl = `${API_BASE_URL}${endpointPath}`;
  const payload = createExamplePayload(fields, values);
  const snippet = integrationSnippet(language, endpointUrl, payload);

  async function copyValue(key, value) {
    try {
      await copyText(value);
      setState((current) => ({ ...current, copied: key }));
      window.setTimeout(() => {
        setState((current) =>
          current ? { ...current, copied: "" } : current
        );
      }, 1800);
    } catch {
      setState((current) => ({ ...current, copied: "" }));
    }
  }

  return (
    <ModalShell
      title="Integrate deployment"
      subtitle={`${deployment.deployment_name} v${deployment.version}`}
      icon={<Code2 size={18} />}
      onClose={onClose}
    >
      <div className="space-y-5">
        <div className="rounded-2xl border border-blue-200 bg-blue-50 p-4">
          <div className="text-[10px] font-black uppercase tracking-wider text-blue-500">
            Recommended authenticated endpoint
          </div>
          <div className="mt-2 flex items-center gap-2">
            <code className="min-w-0 flex-1 break-all rounded-xl bg-white px-3 py-2 text-xs text-slate-700">
              {endpointUrl}
            </code>
            <button
              onClick={() => copyValue("endpoint", endpointUrl)}
              className="grid h-10 w-10 flex-shrink-0 place-items-center rounded-xl bg-blue-600 text-white"
            >
              {copied === "endpoint" ? (
                <Check size={15} />
              ) : (
                <Clipboard size={15} />
              )}
            </button>
          </div>
          <p className="mt-2 text-xs leading-relaxed text-blue-800">
            Use this stable AutoML Forge endpoint. It verifies the access token
            and project, proxies the request to the model container, and stores
            the prediction history.
          </p>
        </div>

        <div className="grid gap-4 lg:grid-cols-[0.75fr_1.25fr]">
          <div className="space-y-4">
            <div className="rounded-2xl border border-slate-200 p-4">
              <div className="text-xs font-black text-slate-900">
                Required request headers
              </div>
              <div className="mt-3 space-y-2 text-xs text-slate-600">
                <code className="block rounded-xl bg-slate-50 p-2">
                  Authorization: Bearer &lt;ACCESS_TOKEN&gt;
                </code>
                <code className="block rounded-xl bg-slate-50 p-2">
                  X-Project-ID: &lt;PROJECT_ID&gt;
                </code>
                <code className="block rounded-xl bg-slate-50 p-2">
                  Content-Type: application/json
                </code>
              </div>
            </div>

            <div className="rounded-2xl border border-slate-200 p-4">
              <div className="text-xs font-black text-slate-900">
                Input schema
              </div>
              <div className="mt-3 space-y-2">
                {fields.map((field) => (
                  <div
                    key={field.name}
                    className="flex items-center justify-between gap-3 rounded-xl bg-slate-50 px-3 py-2 text-xs"
                  >
                    <span className="font-bold text-slate-700">
                      {field.name}
                    </span>
                    <span className="text-[10px] uppercase text-slate-400">
                      {field.type}
                      {field.required ? " · required" : ""}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <a
              href={`${API_BASE_URL}/docs`}
              target="_blank"
              rel="noreferrer"
              className="flex items-center justify-center gap-2 rounded-2xl border border-slate-200 px-4 py-3 text-xs font-black text-slate-700 hover:bg-slate-50"
            >
              <ExternalLink size={14} />
              Open API Documentation
            </a>
          </div>

          <div className="overflow-hidden rounded-2xl bg-slate-950">
            <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
              <div className="flex gap-1">
                {CODE_TABS.map((tab) => (
                  <button
                    key={tab}
                    onClick={() =>
                      setState((current) => ({
                        ...current,
                        language: tab,
                        copied: "",
                      }))
                    }
                    className={`rounded-lg px-3 py-1.5 text-[10px] font-black uppercase ${
                      language === tab
                        ? "bg-white text-slate-950"
                        : "text-slate-400 hover:text-white"
                    }`}
                  >
                    {tab}
                  </button>
                ))}
              </div>

              <button
                onClick={() => copyValue("snippet", snippet)}
                className="flex items-center gap-1.5 rounded-lg bg-white/10 px-3 py-1.5 text-[10px] font-bold text-white"
              >
                {copied === "snippet" ? (
                  <Check size={12} />
                ) : (
                  <Clipboard size={12} />
                )}
                {copied === "snippet" ? "Copied" : "Copy"}
              </button>
            </div>

            <pre className="soft-scroll max-h-[560px] overflow-auto whitespace-pre-wrap p-4 text-xs leading-relaxed text-emerald-300">
              {snippet}
            </pre>
          </div>
        </div>
      </div>
    </ModalShell>
  );
}

function FieldInput({ field, value, onChange }) {
  return (
    <label className="block">
      <span className="mb-1.5 flex items-center gap-1 text-[11px] font-black text-slate-600">
        {prettifyName(field.name)}
        {field.required && <span className="text-red-500">*</span>}
      </span>

      {field.options?.length ? (
        <select
          value={value}
          onChange={(event) => onChange(event.target.value)}
          className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm outline-none focus:border-blue-400 focus:bg-white focus:ring-4 focus:ring-blue-100"
        >
          <option value="">Select value</option>
          {field.options.map((option) => (
            <option key={String(option)} value={String(option)}>
              {String(option)}
            </option>
          ))}
        </select>
      ) : field.type === "boolean" ? (
        <select
          value={value}
          onChange={(event) => onChange(event.target.value)}
          className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm outline-none focus:border-blue-400 focus:bg-white focus:ring-4 focus:ring-blue-100"
        >
          <option value="false">False</option>
          <option value="true">True</option>
        </select>
      ) : (
        <input
          type={field.type === "number" ? "number" : "text"}
          step={field.type === "number" ? "any" : undefined}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={field.description || `Enter ${field.name}`}
          className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm outline-none focus:border-blue-400 focus:bg-white focus:ring-4 focus:ring-blue-100"
        />
      )}

      {field.description && (
        <span className="mt-1 block text-[10px] text-slate-400">
          {field.description}
        </span>
      )}
    </label>
  );
}

function ModalShell({ title, subtitle, icon, onClose, children }) {
  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-slate-950/55 p-4 backdrop-blur-sm">
      <div className="soft-scroll max-h-[92vh] w-full max-w-5xl overflow-auto rounded-3xl border border-white/30 bg-white shadow-2xl">
        <header className="sticky top-0 z-10 flex items-center justify-between border-b border-slate-100 bg-white/95 px-5 py-4 backdrop-blur">
          <div className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-2xl bg-blue-50 text-blue-600">
              {icon}
            </div>
            <div>
              <h3 className="text-base font-black text-slate-950">{title}</h3>
              <p className="text-xs text-slate-500">{subtitle}</p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="grid h-9 w-9 place-items-center rounded-xl bg-slate-100 text-slate-500 hover:bg-slate-200"
          >
            <X size={16} />
          </button>
        </header>

        <div className="p-5">{children}</div>
      </div>
    </div>
  );
}

function StatusBadge({ status }) {
  const styles =
    status === "active"
      ? "bg-emerald-100 text-emerald-700"
      : status === "failed"
      ? "bg-red-100 text-red-700"
      : TRANSITIONAL.has(status)
      ? "bg-blue-100 text-blue-700"
      : "bg-slate-200 text-slate-600";

  return (
    <span
      className={`rounded-full px-2 py-1 text-[9px] font-black uppercase ${styles}`}
    >
      {status}
    </span>
  );
}

function ActionButton({ title, icon, onClick, disabled }) {
  return (
    <button
      title={title}
      onClick={onClick}
      disabled={disabled}
      className="flex items-center justify-center gap-1 rounded-xl bg-white px-2 py-2 text-[10px] font-bold text-slate-600 shadow-sm disabled:opacity-35"
    >
      {icon}
      {title}
    </button>
  );
}
