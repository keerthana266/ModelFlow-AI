import React, { useState } from "react";
import { BrainCircuit, Loader2, LockKeyhole, Mail } from "lucide-react";

import { useAuth } from "../context/AuthContext";

export default function LoginPage({ onShowRegister }) {
  const { login } = useAuth();
  const [form, setForm] = useState({ email: "", password: "" });
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const submit = async (event) => {
    event.preventDefault();
    setSubmitting(true);
    setError("");

    try {
      await login(form);
    } catch (requestError) {
      setError(
        requestError.response?.data?.detail ||
          requestError.message ||
          "Login failed."
      );
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AuthLayout
      title="Welcome back"
      subtitle="Sign in to continue building ML pipelines."
    >
      <form onSubmit={submit} className="space-y-4">
        <AuthInput
          icon={<Mail size={17} />}
          type="email"
          placeholder="Email address"
          value={form.email}
          onChange={(value) => setForm((current) => ({ ...current, email: value }))}
        />
        <AuthInput
          icon={<LockKeyhole size={17} />}
          type="password"
          placeholder="Password"
          value={form.password}
          onChange={(value) =>
            setForm((current) => ({ ...current, password: value }))
          }
        />

        {error && (
          <div className="rounded-2xl border border-red-100 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={submitting}
          className="flex w-full items-center justify-center gap-2 rounded-2xl bg-blue-600 px-4 py-3.5 text-sm font-black text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-400"
        >
          {submitting && <Loader2 size={17} className="animate-spin" />}
          {submitting ? "Signing in..." : "Sign in"}
        </button>
      </form>

      <p className="mt-5 text-center text-sm text-slate-500">
        New to AutoML Forge?{" "}
        <button
          type="button"
          onClick={onShowRegister}
          className="font-black text-blue-600 hover:text-blue-700"
        >
          Create an account
        </button>
      </p>
    </AuthLayout>
  );
}

export function AuthLayout({ title, subtitle, children }) {
  return (
    <div className="grid min-h-screen place-items-center bg-[#f4f7fb] p-6">
      <div className="w-full max-w-md overflow-hidden rounded-[32px] border border-slate-200 bg-white shadow-[0_30px_80px_rgba(15,23,42,0.12)]">
        <div className="bg-slate-950 px-7 py-7 text-white">
          <div className="mb-6 flex items-center gap-3">
            <div className="grid h-12 w-12 place-items-center rounded-2xl bg-blue-600">
              <BrainCircuit size={24} />
            </div>
            <div>
              <div className="text-xl font-black">
                AutoML <span className="text-blue-400">Forge</span>
              </div>
              <div className="text-xs text-slate-400">
                Project-scoped ML workspace
              </div>
            </div>
          </div>

          <h1 className="text-2xl font-black">{title}</h1>
          <p className="mt-1 text-sm text-slate-400">{subtitle}</p>
        </div>

        <div className="p-7">{children}</div>
      </div>
    </div>
  );
}

function AuthInput({ icon, onChange, ...props }) {
  return (
    <label className="flex items-center gap-3 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 transition focus-within:border-blue-400 focus-within:bg-white focus-within:ring-4 focus-within:ring-blue-100">
      <span className="text-slate-400">{icon}</span>
      <input
        {...props}
        required
        onChange={(event) => onChange(event.target.value)}
        className="w-full bg-transparent text-sm text-slate-900 outline-none placeholder:text-slate-400"
      />
    </label>
  );
}
