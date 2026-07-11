import React, { useState } from "react";
import { Check, ChevronDown, FolderKanban, Plus, X } from "lucide-react";

import { useProject } from "../context/ProjectContext";

export default function ProjectSelector() {
  const {
    projects,
    activeProject,
    activeProjectId,
    selectProject,
    createProject,
  } = useProject();

  const [open, setOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [error, setError] = useState("");

  const submitProject = async (event) => {
    event.preventDefault();
    setError("");

    try {
      await createProject({ name, description: description || null });
      setName("");
      setDescription("");
      setCreating(false);
      setOpen(false);
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message);
    }
  };

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex min-w-[190px] items-center gap-2 rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-left transition hover:bg-white"
      >
        <div className="grid h-8 w-8 place-items-center rounded-xl bg-violet-100 text-violet-700">
          <FolderKanban size={16} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-[10px] font-black uppercase tracking-wider text-slate-400">
            Active project
          </div>
          <div className="truncate text-xs font-black text-slate-800">
            {activeProject?.name || "Select project"}
          </div>
        </div>
        <ChevronDown size={14} className="text-slate-400" />
      </button>

      {open && (
        <div className="absolute left-0 top-12 z-50 w-80 rounded-3xl border border-slate-200 bg-white p-3 shadow-[0_24px_70px_rgba(15,23,42,0.18)]">
          <div className="mb-2 flex items-center justify-between px-2 py-1">
            <div className="text-xs font-black uppercase tracking-wider text-slate-500">
              Your projects
            </div>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="rounded-lg p-1 text-slate-400 hover:bg-slate-100"
            >
              <X size={14} />
            </button>
          </div>

          <div className="max-h-56 space-y-1 overflow-auto">
            {projects.map((project) => (
              <button
                key={project.project_id}
                type="button"
                onClick={() => {
                  selectProject(project.project_id);
                  setOpen(false);
                }}
                className={`flex w-full items-center gap-3 rounded-2xl px-3 py-3 text-left transition ${
                  activeProjectId === project.project_id
                    ? "bg-violet-50 text-violet-800"
                    : "hover:bg-slate-50"
                }`}
              >
                <FolderKanban size={16} />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-black">{project.name}</div>
                  <div className="truncate text-[11px] text-slate-400">
                    {project.description || project.project_id}
                  </div>
                </div>
                {activeProjectId === project.project_id && <Check size={15} />}
              </button>
            ))}
          </div>

          {!creating ? (
            <button
              type="button"
              onClick={() => setCreating(true)}
              className="mt-3 flex w-full items-center justify-center gap-2 rounded-2xl border border-dashed border-violet-300 bg-violet-50 px-3 py-3 text-xs font-black text-violet-700"
            >
              <Plus size={15} />
              Create project
            </button>
          ) : (
            <form onSubmit={submitProject} className="mt-3 space-y-2 rounded-2xl bg-slate-50 p-3">
              <input
                required
                minLength={2}
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="Project name"
                className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs outline-none focus:border-violet-400"
              />
              <input
                value={description}
                onChange={(event) => setDescription(event.target.value)}
                placeholder="Description — optional"
                className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs outline-none focus:border-violet-400"
              />
              {error && <p className="text-[11px] text-red-600">{error}</p>}
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setCreating(false)}
                  className="flex-1 rounded-xl border border-slate-200 px-3 py-2 text-xs font-bold text-slate-600"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="flex-1 rounded-xl bg-violet-600 px-3 py-2 text-xs font-black text-white"
                >
                  Create
                </button>
              </div>
            </form>
          )}
        </div>
      )}
    </div>
  );
}
