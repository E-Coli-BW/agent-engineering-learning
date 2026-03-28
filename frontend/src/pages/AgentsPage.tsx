import { useState, useEffect } from 'react'
import {
  Bot,
  RefreshCw,
  Globe,
  Zap,
  Tag,
  ExternalLink,
  ChevronRight,
} from 'lucide-react'
import { a2aGetCard, a2aHealth } from '../api'
import type { AgentCard } from '../api'

// ---------- Component ----------

export default function AgentsPage() {
  const [agents, setAgents] = useState<{ backend: 'a2a' | 'react'; card: AgentCard | null; healthy: boolean; loading: boolean }[]>([
    { backend: 'a2a', card: null, healthy: false, loading: true },
    { backend: 'react', card: null, healthy: false, loading: true },
  ])
  const [refreshing, setRefreshing] = useState(false)

  const refresh = async () => {
    setRefreshing(true)
    const results = await Promise.all(
      agents.map(async (a) => {
        try {
          const [card, health] = await Promise.all([
            a2aGetCard(a.backend),
            a2aHealth(a.backend),
          ])
          return { ...a, card, healthy: health.status === 'ok', loading: false }
        } catch {
          return { ...a, card: null, healthy: false, loading: false }
        }
      }),
    )
    setAgents(results)
    setRefreshing(false)
  }

  useEffect(() => {
    refresh()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-5xl mx-auto px-6 py-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Agents</h1>
            <p className="text-sm text-gray-400 mt-1">A2A Agent Card 管理 & 能力查看</p>
          </div>
          <button
            onClick={refresh}
            disabled={refreshing}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-800 border border-gray-700 text-sm text-gray-300 hover:bg-gray-700 disabled:opacity-50 transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin' : ''}`} />
            刷新
          </button>
        </div>

        {/* Agent Cards */}
        {agents.map((agent) => (
          <AgentCardView key={agent.backend} agent={agent} />
        ))}
      </div>
    </div>
  )
}

// ---------- Sub-components ----------

function AgentCardView({
  agent,
}: {
  agent: {
    backend: 'a2a' | 'react'
    card: AgentCard | null
    healthy: boolean
    loading: boolean
  }
}) {
  const [expandedSkill, setExpandedSkill] = useState<string | null>(null)

  if (agent.loading) {
    return (
      <div className="rounded-xl border border-gray-800 bg-gray-900/50 p-6 animate-pulse">
        <div className="h-6 bg-gray-800 rounded w-48 mb-4" />
        <div className="h-4 bg-gray-800 rounded w-72" />
      </div>
    )
  }

  if (!agent.card) {
    return (
      <div className="rounded-xl border border-red-900/50 bg-red-950/20 p-6">
        <div className="flex items-center gap-2 text-red-400">
          <Bot className="w-5 h-5" />
          <span className="font-medium">
            {agent.backend === 'a2a' ? 'A2A Expert Agent' : 'ReAct Agent'} — 无法连接
          </span>
        </div>
        <p className="text-sm text-gray-500 mt-2">
          请确保对应的后端服务已启动:
          <code className="ml-1 text-gray-400">
            python project/{agent.backend === 'a2a' ? 'a2a_agent' : 'react_agent'}.py --serve
          </code>
        </p>
      </div>
    )
  }

  const card = agent.card

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/50 overflow-hidden">
      {/* Header */}
      <div className="p-5 border-b border-gray-800 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div
            className={`w-10 h-10 rounded-xl flex items-center justify-center ${
              agent.backend === 'a2a'
                ? 'bg-brand-600/20 text-brand-400'
                : 'bg-purple-600/20 text-purple-400'
            }`}
          >
            {agent.backend === 'a2a' ? <Bot className="w-5 h-5" /> : <Zap className="w-5 h-5" />}
          </div>
          <div>
            <h2 className="font-semibold">{card.name}</h2>
            <p className="text-sm text-gray-400">{card.description}</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span
            className={`text-xs px-2 py-1 rounded-full ${
              agent.healthy ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'
            }`}
          >
            {agent.healthy ? '● 在线' : '● 离线'}
          </span>
          <span className="text-xs text-gray-500">v{card.version}</span>
        </div>
      </div>

      {/* Meta */}
      <div className="px-5 py-3 border-b border-gray-800 flex items-center gap-6 text-xs text-gray-400">
        <span className="flex items-center gap-1">
          <Globe className="w-3.5 h-3.5" />
          {card.url}
        </span>
        <span>Streaming: {card.capabilities?.streaming ? '✅' : '❌'}</span>
        <span>Input: {card.defaultInputModes?.join(', ')}</span>
        <span>Output: {card.defaultOutputModes?.join(', ')}</span>
      </div>

      {/* Skills */}
      <div className="p-5">
        <h3 className="text-sm font-medium text-gray-300 mb-3">
          Skills ({card.skills?.length ?? 0})
        </h3>
        <div className="space-y-2">
          {card.skills?.map((skill) => (
            <div key={skill.id} className="rounded-lg border border-gray-800 bg-gray-800/30">
              <button
                onClick={() => setExpandedSkill(expandedSkill === skill.id ? null : skill.id)}
                className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-gray-800/50 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <ChevronRight
                    className={`w-4 h-4 text-gray-500 transition-transform ${
                      expandedSkill === skill.id ? 'rotate-90' : ''
                    }`}
                  />
                  <span className="font-medium text-sm">{skill.name}</span>
                  <span className="text-xs text-gray-500 font-mono">{skill.id}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  {skill.tags?.slice(0, 3).map((tag) => (
                    <span key={tag} className="flex items-center gap-1 text-[11px] px-1.5 py-0.5 rounded bg-gray-700/50 text-gray-400">
                      <Tag className="w-3 h-3" />
                      {tag}
                    </span>
                  ))}
                </div>
              </button>
              {expandedSkill === skill.id && (
                <div className="px-4 pb-3 border-t border-gray-800 pt-3 space-y-2">
                  <p className="text-sm text-gray-400">{skill.description}</p>
                  {skill.examples?.length > 0 && (
                    <div>
                      <div className="text-xs text-gray-500 mb-1">示例问题:</div>
                      {skill.examples.map((ex, i) => (
                        <div key={i} className="flex items-center gap-1 text-sm text-brand-300">
                          <ExternalLink className="w-3 h-3" />
                          {ex}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Raw JSON */}
      <details className="px-5 pb-4">
        <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-400">
          查看原始 Agent Card JSON
        </summary>
        <pre className="mt-2 bg-gray-900 rounded-lg p-3 text-xs text-gray-400 overflow-x-auto max-h-60">
          {JSON.stringify(card, null, 2)}
        </pre>
      </details>
    </div>
  )
}
