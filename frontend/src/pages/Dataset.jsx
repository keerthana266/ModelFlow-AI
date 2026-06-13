import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import api from "../api/client";

export default function Dataset() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [data, setData] = useState(null);
  const [target, setTarget] = useState("");

  useEffect(() => {
    api.get(`/datasets/${id}/profile`).then((res) => {
      setData(res.data);
    });
  }, []);

  if (!data) return <div>Loading dataset...</div>;

  return (
    <div className="space-y-6">

      {/* INFO */}
      <div className="card p-6">
        <h1 className="text-xl font-bold mb-4">Dataset Overview</h1>

        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>Rows: {data.rows}</div>
          <div>Columns: {data.columns}</div>
        </div>
      </div>

      {/* TARGET SELECTOR */}
      <div className="card p-6">
        <h2 className="font-bold mb-3">Select Target Column</h2>

        <input
          className="border p-2 rounded w-full"
          placeholder="Enter target column name"
          value={target}
          onChange={(e) => setTarget(e.target.value)}
        />

        <button
          className="btn-primary mt-4"
          onClick={() =>
            navigate(`/train/${id}?target=${target}`)
          }
        >
          Continue to Training
        </button>
      </div>

    </div>
  );
}