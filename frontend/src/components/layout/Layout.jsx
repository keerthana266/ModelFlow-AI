import Sidebar from "./Sidebar";
import Navbar from "./Navbar";

export default function Layout({ children }) {
  return (
    <div className="flex h-screen bg-slate-50">

      <Sidebar />

      <div className="flex flex-col flex-1">

        <Navbar />

        <main className="p-6 overflow-y-auto">
          {children}
        </main>

      </div>

    </div>
  );
}