import { Routes, Route } from "react-router-dom";

import Dashboard from "../pages/Dashboard";
import Upload from "../pages/Upload";
import Dataset from "../pages/Dataset";
import Train from "../pages/Train";
import Leaderboard from "../pages/Leaderboard";
import Predict from "../pages/Predict";

export default function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<Dashboard />} />
      <Route path="/upload" element={<Upload />} />
      <Route path="/dataset/:id" element={<Dataset />} />
      <Route path="/train/:id" element={<Train />} />
      <Route path="/leaderboard/:id" element={<Leaderboard />} />
      <Route path="/predict/:modelId" element={<Predict />} />
    </Routes>
  );
}