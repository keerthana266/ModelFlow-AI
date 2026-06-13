import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import api from "../api/client";

export default function Leaderboard() {
  const { id } = useParams();
  const [data, setData] = useState(null);

  useEffect(() => {
    api.get(`/experiments/${id}/leaderboard`).then(res => setData(res.data));
  }, []);

  if (!data) return <div>Loading...</div>;

  return (
    <div className="space-y-3">
      <h1 className="font-bold">Leaderboard</h1>

      {data.leaderboard.map((m, i) => (
        <div key={i} className="bg-white p-3 rounded shadow">
          {m.model_name} → {m.score}
        </div>
      ))}
    </div>
  );
}