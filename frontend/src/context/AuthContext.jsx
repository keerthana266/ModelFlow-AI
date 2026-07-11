import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

import {
  api,
  PROJECT_STORAGE_KEY,
  TOKEN_STORAGE_KEY,
} from "../api";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  const clearSession = useCallback(() => {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
    localStorage.removeItem(PROJECT_STORAGE_KEY);
    setUser(null);
  }, []);

  const acceptSession = useCallback((payload) => {
    localStorage.setItem(TOKEN_STORAGE_KEY, payload.access_token);

    if (payload.default_project_id) {
      localStorage.setItem(PROJECT_STORAGE_KEY, payload.default_project_id);
    }

    setUser(payload.user);
    return payload;
  }, []);

  const login = useCallback(
    async (credentials) => {
      const response = await api.post("/auth/login", credentials);
      return acceptSession(response.data);
    },
    [acceptSession]
  );

  const register = useCallback(
    async (details) => {
      const response = await api.post("/auth/register", details);
      return acceptSession(response.data);
    },
    [acceptSession]
  );

  const logout = useCallback(() => {
    clearSession();
  }, [clearSession]);

  useEffect(() => {
    const restoreSession = async () => {
      const token = localStorage.getItem(TOKEN_STORAGE_KEY);

      if (!token) {
        setLoading(false);
        return;
      }

      try {
        const response = await api.get("/auth/me");
        setUser(response.data);
      } catch {
        clearSession();
      } finally {
        setLoading(false);
      }
    };

    restoreSession();
  }, [clearSession]);

  useEffect(() => {
    const handleUnauthorized = () => clearSession();
    window.addEventListener("automl:unauthorized", handleUnauthorized);

    return () => {
      window.removeEventListener("automl:unauthorized", handleUnauthorized);
    };
  }, [clearSession]);

  const value = useMemo(
    () => ({ user, loading, login, register, logout }),
    [user, loading, login, register, logout]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);

  if (!context) {
    throw new Error("useAuth must be used inside AuthProvider.");
  }

  return context;
}
