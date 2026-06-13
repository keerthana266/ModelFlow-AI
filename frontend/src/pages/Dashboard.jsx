import { Link } from "react-router-dom";

export default function Dashboard() {
  return (
    <div className="space-y-6">

      {/* HERO */}
      <div className="card p-8 bg-gradient-to-r from-blue-600 to-indigo-600 text-white">
        <h1 className="text-3xl font-bold">AutoML Forge 🚀</h1>
        <p className="opacity-80 mt-2">
          Upload data → Train models → Deploy AI APIs
        </p>
      </div>

      {/* STATS */}
      <div className="grid grid-cols-3 gap-4">

        <div className="card p-6">
          <p className="text-slate-500 text-sm">Datasets</p>
          <h2 className="text-2xl font-bold">--</h2>
        </div>

        <div className="card p-6">
          <p className="text-slate-500 text-sm">Experiments</p>
          <h2 className="text-2xl font-bold">--</h2>
        </div>

        <div className="card p-6">
          <p className="text-slate-500 text-sm">Models</p>
          <h2 className="text-2xl font-bold">--</h2>
        </div>

      </div>

      {/* ACTIONS */}
      <div className="grid grid-cols-2 gap-4">

        <Link to="/upload" className="card p-6 hover:shadow-md transition">
          <h3 className="font-bold">📤 Upload Dataset</h3>
          <p className="text-sm text-slate-500">
            Start ML pipeline
          </p>
        </Link>

        <div className="card p-6">
          <h3 className="font-bold">🤖 AutoML Training</h3>
          <p className="text-sm text-slate-500">
            Compare multiple models automatically
          </p>
        </div>

      </div>

    </div>
  );
}