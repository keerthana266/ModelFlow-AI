import { useState } from "react";
import { useParams } from "react-router-dom";
import api from "../api/client";

export default function Predict() {
  const { modelId } = useParams();
  const [res, setRes] = useState(null);

  const run = async () => {
    const r = await api.post(`/models/${modelId}/predict`, {
      data: { Age: 22 }
    });

    setRes(r.data);
  };

  return (
    <div>
      <button
        onClick={run}
        className="bg-blue-600 text-white px-4 py-2"
      >
        Predict
      </button>

      <pre>{JSON.stringify(res, null, 2)}</pre>
    </div>
  );
}