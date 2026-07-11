import React, { useState } from "react";
import { LogOut, UserRound } from "lucide-react";

import { useAuth } from "../context/AuthContext";

export default function UserMenu() {
  const { user, logout } = useAuth();
  const [open, setOpen] = useState(false);

  const initials = String(user?.name || user?.email || "U")
    .split(/\s+/)
    .map((part) => part[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="grid h-11 w-11 place-items-center rounded-2xl bg-slate-950 text-sm font-black text-white shadow-sm"
        title={user?.email}
      >
        {initials}
      </button>

      {open && (
        <div className="absolute right-0 top-13 z-50 w-64 rounded-3xl border border-slate-200 bg-white p-3 shadow-[0_24px_70px_rgba(15,23,42,0.18)]">
          <div className="flex items-center gap-3 rounded-2xl bg-slate-50 p-3">
            <div className="grid h-10 w-10 place-items-center rounded-2xl bg-slate-950 text-white">
              <UserRound size={17} />
            </div>
            <div className="min-w-0">
              <div className="truncate text-sm font-black text-slate-900">
                {user?.name}
              </div>
              <div className="truncate text-xs text-slate-500">{user?.email}</div>
            </div>
          </div>

          <button
            type="button"
            onClick={logout}
            className="mt-2 flex w-full items-center gap-2 rounded-2xl px-3 py-3 text-sm font-black text-red-600 transition hover:bg-red-50"
          >
            <LogOut size={16} />
            Sign out
          </button>
        </div>
      )}
    </div>
  );
}
