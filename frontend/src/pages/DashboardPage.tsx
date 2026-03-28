import { useState, useEffect, useCallback } from 'react'
import {
  Activity,
  Server,
  Database,
  Clock,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  RefreshCw,
  Gauge,
} from 'lucide-react'
import { ragHealth, ragMetrics, a2aHealth } from '../api'
import type { HealthResponse, MetricsResponse } from '../api'

// ---------- Types ----------

interface ServiceStatus {
  name: string
  url: string
  status: 'ok' | 'error' | 'loading'
  detail?: string
}

// ---------- Component ----------

export default function DashboardPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null)
  const [services, setServices] = useState<ServiceStatus[]>([
    { name: 'RAG API', url: ':8000', status: 'loading' },
    { name: 'A2A Expert', url: ':5001', status: 'loading' },
    { name: 'ReAct Agent', url: ':5002', status: 'loading' },
  ])
  const [refreshing, setRefreshing] = useState(false)
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null)

  const refresh = useCallback(async () => {
    setRefreshing(true)

    // RAG Health
    const newServices: ServiceStatus[] = [...services]
    try {
      const h = await ragHealth()
      setHealth(h)
      newServices[0] = { ...newServices[0], status: h.status === 'ok' ? 'ok' : 'error', detail: h.ollama }
    } catch {
      newServices[0] = { ...newServices[0], status: 'error', detail: '无法连接' }
    }

    // RAG Metrics
    try {
      const m = await ragMetrics()
      setMetrics(m)
    } catch { /* ignore */ }

    // A2A Health
    try {
      const h = await a2aHealth('a2a')
      newServices[1] = { ...newServices[1], status: 'ok', detail: h.agent }
    } catch {
      newServices[1] = { ...newServices[1], status: 'error', detail: '无法连接' }
    }

    // ReAct Health
    try {
      const h = await a2aHealth('react')
      newServices[2] = { ...newServices[2], status: 'ok', detail: h.agent }
    } catch {
      newServices[2] = { ...newServices[2], status: 'error', detail: '无法连接' }
    }

    setServices(newServices)
    setLastRefresh(new Date())
    setRefreshing(false)
  }, [services])

  useEffect(() => {
    refresh()
    const timer = setInterval(refresh, 30_000) // auto-refresh 30s
    return () => clearInterval(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ---------- Render ----------

  const healthyCount = services.filter((s) => s.status === 'ok').length
  const overallStatus = healthyCount === services.length ? 'ok' : healthyCount > 0 ? 'degraded' : 'down'

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-5xl mx-auto px-6 py-6 space-y-6">
        {/* ---- Header ---- */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Dashboard</h1>
            <p className="text-sm text-gray-400 mt-1">系统状态 & 性能指标</p>
          </div>
          <div className="flex items-center gap-3">
            {lastRefresh && (
              <span className="text-xs text-gray-500">
                上次刷新: {lastRefresh.toLocaleTimeString()}
              </span>
            )}
            <button
              onClick={refresh}
              disabled={refreshing}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-800 border border-gray-700 text-sm text-gray-300 hover:bg-gray-700 disabled:opacity-50 transition-colors"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin' : ''}`} />
              刷新
            </button>
          </div>
        </div>

        {/* ---- Overall Status ---- */}
        <div
          className={`rounded-xl border p-4 flex items-center gap-4 ${
            overallStatus === 'ok'
              ? 'border-green-800/50 bg-green-950/30'
              : overallStatus === 'degraded'
                ? 'border-yellow-800/50 bg-yellow-950/30'
                : 'border-red-800/50 bg-red-950/30'
          }`}
        >
          {overallStatus === 'ok' ? (
            <CheckCircle2 className="w-8 h-8 text-green-400" />
          ) : overallStatus === 'degraded' ? (
            <AlertTriangle className="w-8 h-8 text-yellow-400" />
          ) : (
            <XCircle className="w-8 h-8 text-red-400" />
          )}
          <div>
            <div className="font-semibold">
              {overallStatus === 'ok' ? '所有服务正常运行' : overallStatus === 'degraded' ? '部分服务异常' : '服务不可用'}
            </div>
            <div className="text-sm text-gray-400">
              {healthyCount}/{services.length} 服务在线
            </div>
          </div>
        </div>

        {/* ---- Service Cards ---- */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {services.map((svc) => (
            <div key={svc.name} className="rounded-xl border border-gray-800 bg-gray-900/50 p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Server className="w-4 h-4 text-gray-400" />
                  <span className="font-medium text-sm">{svc.name}</span>
                </div>
                <span
                  className={`w-2.5 h-2.5 rounded-full ${
                    svc.status === 'ok' ? 'bg-green-400' : svc.status === 'error' ? 'bg-red-400' : 'bg-yellow-400 animate-pulse'
                  }`}
                />
              </div>
              <div className="text-xs text-gray-500">{svc.url}</div>
              {svc.detail && <div className="text-xs text-gray-400 mt-1">{svc.detail}</div>}
            </div>
          ))}
        </div>

        {/* ---- Metrics ---- */}
        {(health || metrics) && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <MetricCard
              icon={<Activity className="w-4 h-4 text-brand-400" />}
              label="总请求数"
              value={metrics?.total_requests?.toLocaleString() ?? '—'}
            />
            <MetricCard
              icon={<Clock className="w-4 h-4 text-purple-400" />}
              label="平均延迟"
              value={metrics?.avg_latency_ms != null ? `${metrics.avg_latency_ms}ms` : '—'}
            />
            <MetricCard
              icon={<AlertTriangle className="w-4 h-4 text-yellow-400" />}
              label="错误率"
              value={metrics?.error_rate != null ? `${(metrics.error_rate * 100).toFixed(1)}%` : '—'}
            />
            <MetricCard
              icon={<Database className="w-4 h-4 text-green-400" />}
              label="向量数"
              value={(health?.vector_count ?? metrics?.vector_count)?.toLocaleString() ?? '—'}
            />
          </div>
        )}

        {/* ---- Ollama Models ---- */}
        {health?.models && Object.keys(health.models).length > 0 && (
          <div className="rounded-xl border border-gray-800 bg-gray-900/50 p-4">
            <h3 className="font-medium mb-3 flex items-center gap-2">
              <Gauge className="w-4 h-4 text-gray-400" />
              Ollama 模型
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {Object.entries(health.models).map(([name, size]) => (
                <div key={name} className="bg-gray-800/50 rounded-lg px-3 py-2 text-sm">
                  <div className="font-mono text-brand-300 truncate">{name}</div>
                  <div className="text-xs text-gray-500 mt-0.5">{size}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ---- System Info ---- */}
        {(health || metrics) && (
          <div className="rounded-xl border border-gray-800 bg-gray-900/50 p-4">
            <h3 className="font-medium mb-3">系统信息</h3>
            <div className="grid grid-cols-2 gap-x-8 gap-y-2 text-sm">
              <InfoRow label="Ollama 状态" value={health?.ollama ?? '—'} />
              <InfoRow label="向量库状态" value={health?.vector_store ?? '—'} />
              <InfoRow
                label="运行时间"
                value={metrics?.uptime_seconds != null ? formatUptime(metrics.uptime_seconds) : '—'}
              />
              <InfoRow label="总错误数" value={String(metrics?.total_errors ?? '—')} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ---------- Sub-components ----------

function MetricCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/50 p-4">
      <div className="flex items-center gap-2 text-gray-400 text-xs mb-2">
        {icon}
        {label}
      </div>
      <div className="text-2xl font-bold">{value}</div>
    </div>
  )
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-gray-500">{label}</span>
      <span className="text-gray-300 font-mono text-xs">{value}</span>
    </div>
  )
}

function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) return `${h}h ${m}m ${s}s`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}
