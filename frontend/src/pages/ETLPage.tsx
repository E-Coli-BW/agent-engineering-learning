import { useState } from 'react'
import {
  Database,
  Play,
  Loader2,
  CheckCircle2,
  XCircle,
  FolderOpen,
  Plus,
  Trash2,
} from 'lucide-react'
import { ragETLRun } from '../api'

// ---------- Component ----------

export default function ETLPage() {
  const [sources, setSources] = useState<string[]>(['data/shakespeare.txt'])
  const [newSource, setNewSource] = useState('')
  const [incremental, setIncremental] = useState(true)
  const [chunkSize, setChunkSize] = useState(500)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<{ success: boolean; data: Record<string, unknown> } | null>(null)
  const [history, setHistory] = useState<{ time: string; sources: string[]; success: boolean }[]>([])

  const addSource = () => {
    const s = newSource.trim()
    if (s && !sources.includes(s)) {
      setSources([...sources, s])
      setNewSource('')
    }
  }

  const removeSource = (idx: number) => {
    setSources(sources.filter((_, i) => i !== idx))
  }

  const runETL = async () => {
    if (sources.length === 0) return
    setRunning(true)
    setResult(null)

    try {
      const data = await ragETLRun(sources, incremental)
      setResult({ success: true, data })
      setHistory((prev) => [
        { time: new Date().toLocaleTimeString(), sources: [...sources], success: true },
        ...prev,
      ])
    } catch (err) {
      setResult({
        success: false,
        data: { error: err instanceof Error ? err.message : String(err) },
      })
      setHistory((prev) => [
        { time: new Date().toLocaleTimeString(), sources: [...sources], success: false },
        ...prev,
      ])
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-4xl mx-auto px-6 py-6 space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-xl font-bold">ETL Pipeline</h1>
          <p className="text-sm text-gray-400 mt-1">
            数据导入: 文件 → 分块 → 向量化 → 入库
          </p>
        </div>

        {/* Sources */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/50 p-5">
          <h3 className="font-medium mb-3 flex items-center gap-2">
            <FolderOpen className="w-4 h-4 text-gray-400" />
            数据源
          </h3>
          <div className="space-y-2 mb-3">
            {sources.map((src, i) => (
              <div key={i} className="flex items-center gap-2 bg-gray-800/50 rounded-lg px-3 py-2">
                <Database className="w-4 h-4 text-brand-400 shrink-0" />
                <span className="flex-1 text-sm font-mono text-gray-300 truncate">{src}</span>
                <button
                  onClick={() => removeSource(i)}
                  className="p-1 rounded text-gray-500 hover:text-red-400 transition-colors"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>
          <div className="flex gap-2">
            <input
              value={newSource}
              onChange={(e) => setNewSource(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && addSource()}
              placeholder="添加数据源路径..."
              className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-300 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-brand-500/50"
            />
            <button
              onClick={addSource}
              className="px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-gray-300 hover:bg-gray-700 transition-colors"
            >
              <Plus className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Settings */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/50 p-5">
          <h3 className="font-medium mb-3">参数配置</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <label className="flex items-center justify-between bg-gray-800/50 rounded-lg px-4 py-3">
              <span className="text-sm text-gray-400">增量更新</span>
              <button
                onClick={() => setIncremental(!incremental)}
                className={`relative w-10 h-5 rounded-full transition-colors ${
                  incremental ? 'bg-brand-600' : 'bg-gray-700'
                }`}
              >
                <span
                  className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
                    incremental ? 'translate-x-5' : ''
                  }`}
                />
              </button>
            </label>
            <label className="flex items-center justify-between bg-gray-800/50 rounded-lg px-4 py-3">
              <span className="text-sm text-gray-400">Chunk Size</span>
              <div className="flex items-center gap-2">
                <input
                  type="range"
                  min={100}
                  max={2000}
                  step={100}
                  value={chunkSize}
                  onChange={(e) => setChunkSize(parseInt(e.target.value))}
                  className="w-24 accent-brand-500"
                />
                <span className="text-sm text-gray-300 w-12 text-right">{chunkSize}</span>
              </div>
            </label>
          </div>
        </div>

        {/* Run Button */}
        <button
          onClick={runETL}
          disabled={running || sources.length === 0}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-brand-600 text-white font-medium hover:bg-brand-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {running ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              ETL 运行中...
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              运行 ETL Pipeline
            </>
          )}
        </button>

        {/* Result */}
        {result && (
          <div
            className={`rounded-xl border p-4 ${
              result.success
                ? 'border-green-800/50 bg-green-950/20'
                : 'border-red-800/50 bg-red-950/20'
            }`}
          >
            <div className="flex items-center gap-2 mb-2">
              {result.success ? (
                <CheckCircle2 className="w-5 h-5 text-green-400" />
              ) : (
                <XCircle className="w-5 h-5 text-red-400" />
              )}
              <span className="font-medium">{result.success ? 'ETL 成功' : 'ETL 失败'}</span>
            </div>
            <pre className="text-xs text-gray-400 bg-gray-900/50 rounded-lg p-3 overflow-x-auto">
              {JSON.stringify(result.data, null, 2)}
            </pre>
          </div>
        )}

        {/* History */}
        {history.length > 0 && (
          <div className="rounded-xl border border-gray-800 bg-gray-900/50 p-5">
            <h3 className="font-medium mb-3">运行历史</h3>
            <div className="space-y-2">
              {history.map((h, i) => (
                <div key={i} className="flex items-center gap-3 text-sm">
                  {h.success ? (
                    <CheckCircle2 className="w-4 h-4 text-green-400 shrink-0" />
                  ) : (
                    <XCircle className="w-4 h-4 text-red-400 shrink-0" />
                  )}
                  <span className="text-gray-500">{h.time}</span>
                  <span className="text-gray-400 truncate">{h.sources.join(', ')}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
