import React, { useState } from "react";
import { Loader2, LockKeyhole, Mail, UserRound } from "lucide-react";

import { useAuth } from "../context/AuthContext";
import { AuthLayout } from "./LoginPage";

export default function RegisterPage({ onShowLogin }) {
  const { register } = useAuth();
  const [form, setForm] = useState({ name: "", email: "", password: "" });
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const submit = async (event) => {
    event.preventDefault();
    setSubmitting(true);
    setError("");

    try {
      await register(form);
    } catch (requestError) {
      setError(
        requestError.response?.data?.detail ||
          requestError.message ||
          "Registration failed."
      );
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AuthLayout
      title="Create your workspace"
      subtitle="Your account receives a private default project."
    >
      <form onSubmit={submit} className="space-y-4">
        <AuthInput
          icon={<UserRound size={17} />}
          placeholder="Full name"
          value={form.name}
          onChange={(value) => setForm((current) => ({ ...current, name: value }))}
        />
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
          minLength={8}
          placeholder="Password — minimum 8 characters"
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
          {submitting ? "Creating account..." : "Create account"}
        </button>
      </form>

      <p className="mt-5 text-center text-sm text-slate-500">
        Already registered?{" "}
        <button
          type="button"
          onClick={onShowLogin}
          className="font-black text-blue-600 hover:text-blue-700"
        >
          Sign in
        </button>
      </p>
    </AuthLayout>
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
