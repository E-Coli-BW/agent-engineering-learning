import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import ChatPage from './pages/ChatPage'
import DashboardPage from './pages/DashboardPage'
import AgentsPage from './pages/AgentsPage'
import ETLPage from './pages/ETLPage'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Navigate to="/chat" replace />} />
        <Route path="chat" element={<ChatPage />} />
        <Route path="dashboard" element={<DashboardPage />} />
        <Route path="agents" element={<AgentsPage />} />
        <Route path="etl" element={<ETLPage />} />
      </Route>
    </Routes>
  )
}
