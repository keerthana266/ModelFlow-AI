import { Link, useLocation } from "react-router-dom";

export default function Sidebar() {
  const location = useLocation();

  const nav = [
    { name: "Dashboard", path: "/" },
    { name: "Upload Dataset", path: "/upload" }
  ];

  return (
    <div className="w-64 bg-slate-900 text-white p-6">

      <h1 className="text-xl font-bold text-blue-400 mb-8">
        AutoML Forge
      </h1>

      <div className="space-y-2">
        {nav.map((item) => (
          <Link
            key={item.path}
            to={item.path}
            className={`block px-3 py-2 rounded-lg text-sm transition ${
              location.pathname === item.path
                ? "bg-blue-600"
                : "hover:bg-slate-800"
            }`}
          >
            {item.name}
          </Link>
        ))}
      </div>

    </div>
  );
}