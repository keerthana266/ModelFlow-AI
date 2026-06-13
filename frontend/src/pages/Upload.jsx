import { useState } from "react";
import api from "../api/client";
import { useNavigate } from "react-router-dom";

export default function Upload() {
  const [file, setFile] = useState(null);
  const navigate = useNavigate();

  const upload = async () => {
    const form = new FormData();
    form.append("file", file);

    const res = await api.post("/datasets/upload", form, {
      headers: { "Content-Type": "multipart/form-data" }
    });

    navigate(`/dataset/${res.data.dataset.dataset_id}`);
  };

  return (
    <div className="flex justify-center">

      <div className="card p-8 w-[520px]">

        <h1 className="text-2xl font-bold mb-6">
          Upload Dataset 📁
        </h1>

        <input
          type="file"
          className="mb-6"
          onChange={(e) => setFile(e.target.files[0])}
        />

        <button className="btn-primary w-full" onClick={upload}>
          Upload & Analyze
        </button>

      </div>

    </div>
  );
}