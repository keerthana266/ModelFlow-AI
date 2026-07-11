import React, { StrictMode, useState } from "react";
import { createRoot } from "react-dom/client";
import { Loader2 } from "lucide-react";

import "reactflow/dist/style.css";
import "./index.css";

import App from "./App.jsx";
import { AuthProvider, useAuth } from "./context/AuthContext.jsx";
import { ProjectProvider, useProject } from "./context/ProjectContext.jsx";
import LoginPage from "./pages/LoginPage.jsx";
import RegisterPage from "./pages/RegisterPage.jsx";

function Root() {
  const { user, loading } = useAuth();
  const [screen, setScreen] = useState("login");

  if (loading) {
    return <LoadingScreen label="Restoring session..." />;
  }

  if (!user) {
    return screen === "register" ? (
      <RegisterPage onShowLogin={() => setScreen("login")} />
    ) : (
      <LoginPage onShowRegister={() => setScreen("register")} />
    );
  }

  return (
    <ProjectProvider>
      <ProjectGate />
    </ProjectProvider>
  );
}

function ProjectGate() {
  const { loading, activeProjectId } = useProject();

  if (loading) {
    return <LoadingScreen label="Loading projects..." />;
  }

  if (!activeProjectId) {
    return <LoadingScreen label="Create a project from the project selector." />;
  }

  return <App key={activeProjectId} />;
}

function LoadingScreen({ label }) {
  return (
    <div className="grid min-h-screen place-items-center bg-slate-950 text-white">
      <div className="flex items-center gap-3 rounded-3xl border border-slate-800 bg-slate-900 px-6 py-4 text-sm font-black">
        <Loader2 size={18} className="animate-spin text-blue-400" />
        {label}
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <AuthProvider>
      <Root />
    </AuthProvider>
  </StrictMode>
);
