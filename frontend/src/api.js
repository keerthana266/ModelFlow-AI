import axios from "axios";

export const API_BASE_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export const WS_BASE_URL = API_BASE_URL.replace(/^http/, "ws");

export const TOKEN_STORAGE_KEY = "automl_access_token";
export const PROJECT_STORAGE_KEY = "automl_project_id";

export const api = axios.create({
  baseURL: API_BASE_URL,
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem(TOKEN_STORAGE_KEY);
  const projectId = localStorage.getItem(PROJECT_STORAGE_KEY);

  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }

  if (projectId) {
    config.headers["X-Project-ID"] = projectId;
  }

  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem(TOKEN_STORAGE_KEY);
      localStorage.removeItem(PROJECT_STORAGE_KEY);
      window.dispatchEvent(new Event("automl:unauthorized"));
    }

    return Promise.reject(error);
  }
);
