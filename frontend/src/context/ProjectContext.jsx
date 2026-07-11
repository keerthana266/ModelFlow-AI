import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

import { api, PROJECT_STORAGE_KEY } from "../api";

const ProjectContext = createContext(null);

export function ProjectProvider({ children }) {
  const [projects, setProjects] = useState([]);
  const [activeProjectId, setActiveProjectIdState] = useState(
    localStorage.getItem(PROJECT_STORAGE_KEY) || ""
  );
  const [loading, setLoading] = useState(true);

  const selectProject = useCallback((projectId) => {
    if (projectId) {
      localStorage.setItem(PROJECT_STORAGE_KEY, projectId);
    } else {
      localStorage.removeItem(PROJECT_STORAGE_KEY);
    }

    setActiveProjectIdState(projectId);
  }, []);

  const refreshProjects = useCallback(async () => {
    setLoading(true);

    try {
      const response = await api.get("/projects");
      const nextProjects = response.data.projects || [];
      setProjects(nextProjects);

      const storedProjectId = localStorage.getItem(PROJECT_STORAGE_KEY);
      const storedProjectExists = nextProjects.some(
        (project) => project.project_id === storedProjectId
      );

      if (storedProjectExists) {
        setActiveProjectIdState(storedProjectId);
      } else if (nextProjects[0]) {
        selectProject(nextProjects[0].project_id);
      } else {
        selectProject("");
      }
    } finally {
      setLoading(false);
    }
  }, [selectProject]);

  const createProject = useCallback(
    async ({ name, description }) => {
      const response = await api.post("/projects", { name, description });
      const project = response.data;
      setProjects((current) => [project, ...current]);
      selectProject(project.project_id);
      return project;
    },
    [selectProject]
  );

  const updateProject = useCallback(async (projectId, changes) => {
    const response = await api.patch(`/projects/${projectId}`, changes);
    const updated = response.data;

    setProjects((current) =>
      current.map((project) =>
        project.project_id === projectId ? updated : project
      )
    );

    return updated;
  }, []);

  useEffect(() => {
    refreshProjects().catch(() => setLoading(false));
  }, [refreshProjects]);

  const activeProject = useMemo(
    () =>
      projects.find((project) => project.project_id === activeProjectId) || null,
    [projects, activeProjectId]
  );

  const value = useMemo(
    () => ({
      projects,
      activeProject,
      activeProjectId,
      loading,
      selectProject,
      createProject,
      updateProject,
      refreshProjects,
    }),
    [
      projects,
      activeProject,
      activeProjectId,
      loading,
      selectProject,
      createProject,
      updateProject,
      refreshProjects,
    ]
  );

  return (
    <ProjectContext.Provider value={value}>
      {children}
    </ProjectContext.Provider>
  );
}

export function useProject() {
  const context = useContext(ProjectContext);

  if (!context) {
    throw new Error("useProject must be used inside ProjectProvider.");
  }

  return context;
}
