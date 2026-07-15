import React, { useMemo, useState } from "react";
import {
  Boxes,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Loader2,
  Pencil,
  Plus,
  RefreshCcw,
  ShieldCheck,
  Trash2,
  XCircle,
} from "lucide-react";

import { modelZooApi } from "../api";

const EMPTY_FORM = {
  name: "",
  description: "",
  task_type: "classification",
  class_path: "sklearn.ensemble.RandomForestClassifier",
  default_parameters: JSON.stringify(
    {
      n_estimators: 100,
      random_state: 42,
      n_jobs: -1,
    },
    null,
    2
  ),
  hyperparameter_space: JSON.stringify(
    {
      n_estimators: {
        type: "integer",
        low: 50,
        high: 300,
        step: 50,
      },
      max_depth: {
        type: "integer",
        low: 2,
        high: 20,
      },
    },
    null,
    2
  ),
};

function readApiError(error) {
  const detail = error.response?.data?.detail;

  if (typeof detail === "string") {
    return detail;
  }

  if (detail) {
    return JSON.stringify(detail);
  }

  return error.message || "Model Zoo request failed.";
}

function parseObject(value, label) {
  const parsed = JSON.parse(value || "{}");

  if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
    throw new Error(`${label} must be a JSON object.`);
  }

  return parsed;
}

export default function ModelZooPanel({
  models,
  onRefresh,
  onChanged,
}) {
  const [expanded, setExpanded] = useState(false);
  const [form, setForm] = useState(EMPTY_FORM);
  const [editingId, setEditingId] = useState("");
  const [busyId, setBusyId] = useState("");
  const [saving, setSaving] = useState(false);
  const [notice, setNotice] = useState(null);

  const validCount = useMemo(
    () => models.filter((model) => model.status === "valid").length,
    [models]
  );

  const setField = (field, value) => {
    setForm((current) => ({
      ...current,
      [field]: value,
    }));
  };

  const resetForm = () => {
    setForm(EMPTY_FORM);
    setEditingId("");
    setNotice(null);
  };

  const beginEdit = (model) => {
    setEditingId(model.model_zoo_id);
    setExpanded(true);
    setNotice(null);
    setForm({
      name: model.name || "",
      description: model.description || "",
      task_type: model.task_type || "classification",
      class_path: model.class_path || "",
      default_parameters: JSON.stringify(
        model.default_parameters || {},
        null,
        2
      ),
      hyperparameter_space: JSON.stringify(
        model.hyperparameter_space || {},
        null,
        2
      ),
    });
  };

  const submit = async (event) => {
    event.preventDefault();
    setSaving(true);
    setNotice(null);

    try {
      const payload = {
        name: form.name.trim(),
        description: form.description.trim() || null,
        framework: "scikit-learn",
        task_type: form.task_type,
        class_path: form.class_path.trim(),
        default_parameters: parseObject(
          form.default_parameters,
          "Default parameters"
        ),
        hyperparameter_space: parseObject(
          form.hyperparameter_space,
          "Hyperparameter space"
        ),
      };

      let modelZooId = editingId;

      if (editingId) {
        await modelZooApi.update(editingId, payload);
      } else {
        const createResponse = await modelZooApi.create(payload);
        modelZooId = createResponse.data.model_zoo_id;
      }

      const validationResponse = await modelZooApi.validate(modelZooId);
      const validated = validationResponse.data;

      setNotice({
        type: validated.status === "valid" ? "success" : "error",
        text:
          validated.status === "valid"
            ? "Model definition saved and validated."
            : validated.validation_error || "Model validation failed.",
      });

      await onChanged();
      setEditingId("");

      if (validated.status === "valid") {
        setForm(EMPTY_FORM);
      }
    } catch (error) {
      setNotice({
        type: "error",
        text: readApiError(error),
      });
    } finally {
      setSaving(false);
    }
  };

  const validateModel = async (modelZooId) => {
    setBusyId(modelZooId);
    setNotice(null);

    try {
      const response = await modelZooApi.validate(modelZooId);
      const model = response.data;

      setNotice({
        type: model.status === "valid" ? "success" : "error",
        text:
          model.status === "valid"
            ? `${model.name} is valid and can be selected in AutoML.`
            : model.validation_error || "Validation failed.",
      });

      await onChanged();
    } catch (error) {
      setNotice({
        type: "error",
        text: readApiError(error),
      });
    } finally {
      setBusyId("");
    }
  };

  const deleteModel = async (model) => {
    const accepted = window.confirm(
      `Delete Model Zoo definition "${model.name}"?`
    );

    if (!accepted) return;

    setBusyId(model.model_zoo_id);
    setNotice(null);

    try {
      await modelZooApi.remove(model.model_zoo_id);
      await onChanged();

      if (editingId === model.model_zoo_id) {
        resetForm();
      }

      setNotice({
        type: "success",
        text: "Model definition deleted.",
      });
    } catch (error) {
      setNotice({
        type: "error",
        text: readApiError(error),
      });
    } finally {
      setBusyId("");
    }
  };

  return (
    <section className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
      <div className="flex items-center justify-between border-b border-slate-100 px-4 py-4">
        <div className="flex items-center gap-2">
          <div className="grid h-9 w-9 place-items-center rounded-2xl bg-violet-50 text-violet-600">
            <Boxes size={17} />
          </div>

          <div>
            <h2 className="text-sm font-bold text-slate-900">Custom Model Zoo</h2>
            <p className="text-xs text-slate-500">
              {validCount}/{models.length} validated
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={onRefresh}
            className="rounded-xl p-2 text-slate-400 transition hover:bg-slate-100 hover:text-violet-600"
            title="Refresh Model Zoo"
          >
            <RefreshCcw size={15} />
          </button>

          <button
            onClick={() => setExpanded((current) => !current)}
            className="rounded-xl p-2 text-slate-400 transition hover:bg-slate-100 hover:text-violet-600"
            title="Create Model Zoo definition"
          >
            {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
        </div>
      </div>

      {expanded && (
        <form onSubmit={submit} className="space-y-3 border-b border-slate-100 p-4">
          <div className="flex items-center justify-between">
            <div className="text-xs font-black uppercase tracking-wider text-slate-500">
              {editingId ? "Edit definition" : "Register definition"}
            </div>

            {editingId && (
              <button
                type="button"
                onClick={resetForm}
                className="text-[11px] font-bold text-slate-400 hover:text-slate-700"
              >
                Cancel edit
              </button>
            )}
          </div>

          <Field label="Display name">
            <input
              required
              value={form.name}
              onChange={(event) => setField("name", event.target.value)}
              placeholder="Custom Random Forest"
              className="w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs outline-none focus:border-violet-400 focus:bg-white"
            />
          </Field>

          <Field label="Task type">
            <select
              value={form.task_type}
              onChange={(event) => setField("task_type", event.target.value)}
              className="w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs outline-none focus:border-violet-400 focus:bg-white"
            >
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
              <option value="both">Both</option>
            </select>
          </Field>

          <Field label="Estimator class path">
            <input
              required
              value={form.class_path}
              onChange={(event) => setField("class_path", event.target.value)}
              placeholder="sklearn.ensemble.RandomForestClassifier"
              className="w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 font-mono text-[11px] outline-none focus:border-violet-400 focus:bg-white"
            />
          </Field>

          <Field label="Description">
            <textarea
              rows={2}
              value={form.description}
              onChange={(event) => setField("description", event.target.value)}
              placeholder="Why this model exists..."
              className="soft-scroll w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs outline-none focus:border-violet-400 focus:bg-white"
            />
          </Field>

          <Field label="Default parameters (JSON)">
            <textarea
              rows={5}
              value={form.default_parameters}
              onChange={(event) =>
                setField("default_parameters", event.target.value)
              }
              className="soft-scroll w-full rounded-xl border border-slate-200 bg-slate-950 px-3 py-2 font-mono text-[10px] text-slate-100 outline-none focus:border-violet-400"
            />
          </Field>

          <Field label="Optuna search space (JSON)">
            <textarea
              rows={7}
              value={form.hyperparameter_space}
              onChange={(event) =>
                setField("hyperparameter_space", event.target.value)
              }
              className="soft-scroll w-full rounded-xl border border-slate-200 bg-slate-950 px-3 py-2 font-mono text-[10px] text-slate-100 outline-none focus:border-violet-400"
            />
          </Field>

          <button
            type="submit"
            disabled={saving}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-violet-600 px-3 py-2.5 text-xs font-bold text-white transition hover:bg-violet-700 disabled:cursor-not-allowed disabled:bg-slate-400"
          >
            {saving ? (
              <Loader2 size={14} className="animate-spin" />
            ) : editingId ? (
              <Pencil size={14} />
            ) : (
              <Plus size={14} />
            )}
            {saving
              ? "Saving..."
              : editingId
              ? "Update and validate"
              : "Create and validate"}
          </button>
        </form>
      )}

      {notice && (
        <div
          className={`mx-4 mt-4 rounded-2xl border px-3 py-2 text-xs ${
            notice.type === "success"
              ? "border-emerald-100 bg-emerald-50 text-emerald-700"
              : "border-red-100 bg-red-50 text-red-700"
          }`}
        >
          {notice.text}
        </div>
      )}

      <div className="soft-scroll max-h-72 space-y-3 overflow-auto p-4">
        {models.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-slate-200 p-5 text-center text-xs text-slate-400">
            No Model Zoo definitions yet.
          </div>
        ) : (
          models.map((model) => (
            <div
              key={model.model_zoo_id}
              className="rounded-2xl border border-slate-200 bg-slate-50 p-3"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="truncate text-xs font-black text-slate-800">
                    {model.name}
                  </div>
                  <div className="mt-1 truncate font-mono text-[10px] text-slate-400">
                    {model.class_path}
                  </div>
                </div>

                <StatusBadge status={model.status} />
              </div>

              <div className="mt-2 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                {model.task_type} · {model.framework}
              </div>

              {model.validation_error && (
                <div className="mt-2 rounded-xl bg-red-50 px-2 py-2 text-[10px] text-red-700">
                  {model.validation_error}
                </div>
              )}

              <div className="mt-3 flex items-center gap-2">
                <button
                  onClick={() => validateModel(model.model_zoo_id)}
                  disabled={busyId === model.model_zoo_id}
                  className="flex items-center gap-1 rounded-lg bg-white px-2 py-1.5 text-[10px] font-bold text-emerald-700 shadow-sm disabled:opacity-50"
                >
                  {busyId === model.model_zoo_id ? (
                    <Loader2 size={12} className="animate-spin" />
                  ) : (
                    <ShieldCheck size={12} />
                  )}
                  Validate
                </button>

                <button
                  onClick={() => beginEdit(model)}
                  className="flex items-center gap-1 rounded-lg bg-white px-2 py-1.5 text-[10px] font-bold text-blue-700 shadow-sm"
                >
                  <Pencil size={12} />
                  Edit
                </button>

                <button
                  onClick={() => deleteModel(model)}
                  disabled={busyId === model.model_zoo_id}
                  className="ml-auto rounded-lg bg-white p-1.5 text-red-500 shadow-sm disabled:opacity-50"
                  title="Delete definition"
                >
                  <Trash2 size={13} />
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </section>
  );
}

function Field({ label, children }) {
  return (
    <label className="block">
      <span className="mb-1 block text-[10px] font-bold uppercase tracking-wider text-slate-400">
        {label}
      </span>
      {children}
    </label>
  );
}

function StatusBadge({ status }) {
  if (status === "valid") {
    return (
      <span className="flex items-center gap-1 rounded-full bg-emerald-100 px-2 py-1 text-[9px] font-black uppercase text-emerald-700">
        <CheckCircle2 size={10} />
        Valid
      </span>
    );
  }

  if (status === "invalid") {
    return (
      <span className="flex items-center gap-1 rounded-full bg-red-100 px-2 py-1 text-[9px] font-black uppercase text-red-700">
        <XCircle size={10} />
        Invalid
      </span>
    );
  }

  return (
    <span className="rounded-full bg-slate-200 px-2 py-1 text-[9px] font-black uppercase text-slate-600">
      {status || "draft"}
    </span>
  );
}
