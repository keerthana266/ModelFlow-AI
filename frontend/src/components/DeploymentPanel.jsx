import React, { useEffect, useMemo, useState } from "react";
import {
  Boxes,
  CheckCircle2,
  Code2,
  ExternalLink,
  PlayCircle,
  RefreshCcw,
  Rocket,
} from "lucide-react";
import { api } from "../api";

export default function DeploymentPanel({ models, deployments, onRefresh }) {
  const [selectedModelId, setSelectedModelId] = useState("");
  const [selectedDeploymentId, setSelectedDeploymentId] = useState("");
  const [deploymentName, setDeploymentName] = useState("house-price-api");
  const [environment, setEnvironment] = useState("production");
  const [activate, setActivate] = useState(true);

  const [modelDetails, setModelDetails] = useState(null);
  const [inputJson, setInputJson] = useState("{}");
  const [createResult, setCreateResult] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);

  const [isCreating, setIsCreating] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [error, setError] = useState("");

  const selectedDeployment = useMemo(() => {
    return deployments.find(
      (deployment) => deployment.deployment_id === selectedDeploymentId
    );
  }, [deployments, selectedDeploymentId]);

  useEffect(() => {
    if (!selectedModelId) {
      setModelDetails(null);
      setInputJson("{}");
      return;
    }

    loadModelDetails(selectedModelId);
  }, [selectedModelId]);

  const loadModelDetails = async (modelId) => {
    try {
      setError("");
      const response = await api.get(`/models/${modelId}`);
      const details = response.data;

      setModelDetails(details);
      setInputJson(JSON.stringify(generateExampleInput(details), null, 2));

      if (details.target_column) {
        const safeName = String(details.target_column)
          .toLowerCase()
          .replaceAll("_", "-")
          .replaceAll(" ", "-");

        setDeploymentName(`${safeName}-api`);
      }
    } catch (error) {
      setError(error.response?.data?.detail || error.message);
    }
  };

  const createDeployment = async () => {
    try {
      setError("");
      setCreateResult(null);

      if (!selectedModelId) {
        setError("Select a model first.");
        return;
      }

      if (!deploymentName.trim()) {
        setError("Deployment name is required.");
        return;
      }

      setIsCreating(true);

      const response = await api.post("/deployments", {
        model_id: selectedModelId,
        deployment_name: deploymentName.trim(),
        environment,
        activate,
      });

      setCreateResult(response.data);
      setSelectedDeploymentId(response.data.deployment_id);

      await onRefresh?.();
    } catch (error) {
      setError(error.response?.data?.detail || error.message);
    } finally {
      setIsCreating(false);
    }
  };

  const runDeploymentPrediction = async () => {
    try {
      setError("");
      setPredictionResult(null);

      if (!selectedDeploymentId) {
        setError("Select a deployment first.");
        return;
      }

      const parsedInput = JSON.parse(inputJson);

      setIsPredicting(true);

      const response = await api.post(
        `/deployments/${selectedDeploymentId}/predict`,
        {
          data: parsedInput,
        }
      );

      setPredictionResult(response.data);
      await onRefresh?.();
    } catch (error) {
      if (error instanceof SyntaxError) {
        setError("Invalid JSON format.");
      } else {
        setError(error.response?.data?.detail || error.message);
      }
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <section className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
      <div className="flex items-center justify-between border-b border-slate-100 px-4 py-4">
        <div className="flex items-center gap-2">
          <div className="grid h-9 w-9 place-items-center rounded-2xl bg-violet-50 text-violet-600">
            <Rocket size={17} />
          </div>

          <div>
            <h2 className="text-sm font-bold text-slate-900">
              Deployment Center
            </h2>
            <p className="text-xs text-slate-500">
              Deploy models as versioned REST APIs
            </p>
          </div>
        </div>

        <button
          onClick={onRefresh}
          className="rounded-xl p-2 text-slate-400 transition hover:bg-slate-100 hover:text-violet-600"
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
            className="mt-1 w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 outline-none focus:border-violet-400 focus:ring-4 focus:ring-violet-100"
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
            Deployment Name
          </label>

          <input
            value={deploymentName}
            onChange={(e) => setDeploymentName(e.target.value)}
            placeholder="house-price-api"
            className="mt-1 w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 outline-none focus:border-violet-400 focus:ring-4 focus:ring-violet-100"
          />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
              Environment
            </label>

            <select
              value={environment}
              onChange={(e) => setEnvironment(e.target.value)}
              className="mt-1 w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 outline-none focus:border-violet-400 focus:ring-4 focus:ring-violet-100"
            >
              <option value="development">development</option>
              <option value="staging">staging</option>
              <option value="production">production</option>
            </select>
          </div>

          <label className="mt-6 flex cursor-pointer items-center gap-2 rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-semibold text-slate-700">
            <input
              type="checkbox"
              checked={activate}
              onChange={(e) => setActivate(e.target.checked)}
              className="h-4 w-4"
            />
            Active
          </label>
        </div>

        <button
          onClick={createDeployment}
          disabled={isCreating || !selectedModelId}
          className="flex w-full items-center justify-center gap-2 rounded-2xl bg-violet-600 px-4 py-3 text-sm font-bold text-white shadow-sm transition hover:bg-violet-700 disabled:cursor-not-allowed disabled:bg-slate-400"
        >
          <Rocket size={17} />
          {isCreating ? "Creating Deployment..." : "Create Deployment"}
        </button>

        {createResult && (
          <div className="rounded-2xl border border-violet-100 bg-violet-50 p-3">
            <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-violet-700">
              <CheckCircle2 size={14} />
              Deployment Created
            </div>

            <div className="mt-2 space-y-1 text-xs text-slate-700">
              <div>
                Name:{" "}
                <span className="font-bold">
                  {createResult.deployment_name}
                </span>
              </div>
              <div>
                Version:{" "}
                <span className="font-bold">v{createResult.version}</span>
              </div>
              <div>
                Status:{" "}
                <span className="font-bold">{createResult.status}</span>
              </div>
            </div>

            <div className="mt-3 rounded-xl bg-white p-2 font-mono text-[11px] text-slate-600">
              {createResult.versioned_endpoint_path}
            </div>
          </div>
        )}

        <div className="border-t border-slate-100 pt-4">
          <label className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
            Test Deployment
          </label>

          <select
            value={selectedDeploymentId}
            onChange={(e) => setSelectedDeploymentId(e.target.value)}
            className="mt-1 w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 outline-none focus:border-violet-400 focus:ring-4 focus:ring-violet-100"
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
            <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-500">
              <Boxes size={13} />
              Active Endpoint
            </div>

            <div className="mt-2 rounded-xl bg-white p-2 font-mono text-[11px] text-slate-700">
              {selectedDeployment.versioned_endpoint_path}
            </div>

            <div className="mt-2 text-xs text-slate-500">
              Environment:{" "}
              <span className="font-bold text-slate-700">
                {selectedDeployment.environment}
              </span>
            </div>
          </div>
        )}

        <div>
          <div className="mb-1 flex items-center justify-between">
            <label className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
              Prediction Input JSON
            </label>

            <button
              onClick={() => {
                if (modelDetails) {
                  setInputJson(
                    JSON.stringify(generateExampleInput(modelDetails), null, 2)
                  );
                }
              }}
              disabled={!modelDetails}
              className="flex items-center gap-1 rounded-xl px-2 py-1 text-[11px] font-bold text-slate-500 transition hover:bg-slate-100 hover:text-violet-600 disabled:opacity-40"
            >
              <Code2 size={12} />
              Auto-fill
            </button>
          </div>

          <textarea
            value={inputJson}
            onChange={(e) => setInputJson(e.target.value)}
            className="soft-scroll h-36 w-full rounded-2xl border border-slate-200 bg-slate-950 px-3 py-3 font-mono text-xs text-slate-200 outline-none focus:border-violet-400 focus:ring-4 focus:ring-violet-100"
          />
        </div>

        <button
          onClick={runDeploymentPrediction}
          disabled={isPredicting || !selectedDeploymentId}
          className="flex w-full items-center justify-center gap-2 rounded-2xl bg-slate-950 px-4 py-3 text-sm font-bold text-white shadow-sm transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
        >
          <PlayCircle size={17} />
          {isPredicting ? "Predicting..." : "Test Deployment API"}
        </button>

        {error && (
          <div className="rounded-2xl border border-red-100 bg-red-50 p-3 text-xs text-red-600">
            {error}
          </div>
        )}

        {predictionResult && (
          <div className="rounded-2xl border border-emerald-100 bg-emerald-50 p-3">
            <div className="text-xs font-bold uppercase tracking-wider text-emerald-700">
              Deployment Prediction
            </div>

            <pre className="soft-scroll mt-2 max-h-44 overflow-auto text-xs text-slate-800">
              {JSON.stringify(predictionResult, null, 2)}
            </pre>
          </div>
        )}

        <div>
          <div className="mb-2 text-[11px] font-bold uppercase tracking-wider text-slate-500">
            Recent Deployments
          </div>

          <div className="soft-scroll max-h-52 space-y-2 overflow-auto pr-1">
            {deployments.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-slate-200 p-4 text-center text-xs text-slate-400">
                No deployments yet.
              </div>
            ) : (
              deployments.map((deployment) => (
                <button
                  key={deployment.deployment_id}
                  onClick={() =>
                    setSelectedDeploymentId(deployment.deployment_id)
                  }
                  className="w-full rounded-2xl border border-slate-200 bg-slate-50 p-3 text-left transition hover:border-violet-300 hover:bg-violet-50"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <div className="truncate text-xs font-bold text-slate-800">
                        {deployment.deployment_name}
                      </div>

                      <div className="mt-1 text-xs text-slate-500">
                        v{deployment.version} · {deployment.environment} ·{" "}
                        {deployment.status}
                      </div>

                      <div className="mt-1 truncate text-[10px] text-slate-400">
                        {deployment.deployment_id}
                      </div>
                    </div>

                    <ExternalLink size={13} className="text-slate-300" />
                  </div>
                </button>
              ))
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

function generateExampleInput(details) {
  const featureColumns = details.feature_columns || [];
  const numericalColumns = new Set(details.numerical_columns || []);
  const categoricalColumns = new Set(details.categorical_columns || []);

  const example = {};

  featureColumns.forEach((column) => {
    if (numericalColumns.has(column)) {
      example[column] = getSmartNumericExample(column);
    } else if (categoricalColumns.has(column)) {
      example[column] = getSmartCategoricalExample(column);
    } else {
      example[column] = getFallbackExample(column);
    }
  });

  return example;
}

function getSmartNumericExample(column) {
  const key = column.toLowerCase();

  if (key.includes("age")) return 10;
  if (key.includes("area") || key.includes("sqft") || key.includes("size")) return 1200;
  if (key.includes("bedroom") || key.includes("bed")) return 3;
  if (key.includes("bathroom") || key.includes("bath")) return 2;
  if (key.includes("fare")) return 50;
  if (key.includes("salary") || key.includes("income")) return 50000;
  if (key.includes("year")) return 2024;
  if (key.includes("distance")) return 10;
  if (key.includes("rating")) return 4.5;
  if (key.includes("count") || key.includes("quantity")) return 1;

  return 0;
}

function getSmartCategoricalExample(column) {
  const key = column.toLowerCase();

  if (key.includes("sex") || key.includes("gender")) return "female";
  if (key.includes("location") || key.includes("city")) return "Urban";
  if (key.includes("embarked")) return "S";
  if (key.includes("class")) return "First";
  if (key.includes("type")) return "standard";
  if (key.includes("category")) return "A";
  if (key.includes("status")) return "active";

  return "sample";
}

function getFallbackExample(column) {
  const key = column.toLowerCase();

  if (
    key.includes("id") ||
    key.includes("number") ||
    key.includes("amount") ||
    key.includes("score")
  ) {
    return 1;
  }

  return "sample";
}