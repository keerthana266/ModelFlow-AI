import { useParams, useSearchParams, useNavigate } from "react-router-dom";
import api from "../api/client";

export default function Train() {
  const { id } = useParams();
  const [params] = useSearchParams();
  const navigate = useNavigate();

  const target = params.get("target");

  const train = async () => {
    const res = await api.post("/experiments/start", {
      dataset_id: id,
      target_column: target
    });

    navigate(`/leaderboard/${res.data.result.experiment_id}`);
  };

  return (
    <div className="card p-6">
      <h1 className="text-xl font-bold">Training Models 🤖</h1>

      <p className="text-sm text-slate-500 mt-2">
        Target: {target}
      </p>

      <button className="btn-primary mt-4" onClick={train}>
        Start AutoML Training
      </button>
    </div>
  );
}