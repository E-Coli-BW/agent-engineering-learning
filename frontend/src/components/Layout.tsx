import { NavLink, Outlet } from 'react-router-dom'
import {
  MessageSquare,
  LayoutDashboard,
  Bot,
  Database,
  Zap,
} from 'lucide-react'

const NAV = [
  { to: '/chat', label: '💬 Chat', icon: MessageSquare },
  { to: '/dashboard', label: '📊 Dashboard', icon: LayoutDashboard },
  { to: '/agents', label: '🤖 Agents', icon: Bot },
  { to: '/etl', label: '🗄️ ETL', icon: Database },
]

export default function Layout() {
  return (
    <div className="flex h-screen overflow-hidden bg-gray-950 text-gray-100">
      {/* ---- Sidebar ---- */}
      <aside className="w-60 shrink-0 flex flex-col border-r border-gray-800 bg-gray-900/70">
        {/* Logo */}
        <div className="flex items-center gap-2 px-5 py-4 border-b border-gray-800">
          <Zap className="w-6 h-6 text-brand-400" />
          <span className="text-lg font-bold bg-gradient-to-r from-brand-400 to-purple-400 bg-clip-text text-transparent">
            Agent Hub
          </span>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-brand-600/20 text-brand-300'
                    : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
                }`
              }
            >
              <Icon className="w-4.5 h-4.5" />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-gray-800 text-xs text-gray-500">
          Agent Learning Project<br />
          <span className="text-gray-600">RAG · A2A · ReAct</span>
        </div>
      </aside>

      {/* ---- Main ---- */}
      <main className="flex-1 overflow-hidden">
        <Outlet />
      </main>
    </div>
  )
}
