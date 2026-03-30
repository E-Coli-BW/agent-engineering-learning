import { useState, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import {
  Send,
  Trash2,
  Settings2,
  ChevronDown,
  Loader2,
  BookOpen,
  Sparkles,
} from 'lucide-react'
import type { BackendType, SourceDocument } from '../api'
import {
  ragQueryStream,
  a2aSendSubscribe,
  a2aSendTask,
} from '../api'

// ---------- Types ----------

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: SourceDocument[]
  backend?: BackendType
  timestamp: Date
}

// ---------- Constants ----------

const BACKENDS: { value: BackendType; label: string; desc: string }[] = [
  { value: 'rag', label: 'RAG 知识库', desc: '向量检索 + LLM 生成' },
  { value: 'a2a', label: 'A2A Expert', desc: 'A2A 协议 Expert Agent' },
  { value: 'react', label: 'ReAct Agent', desc: 'ReAct + 工具调用' },
]

const EXAMPLES = [
  'Self-Attention 为什么要除以 √d_k？',
  'LoRA 的低秩分解原理是什么？',
  '如何优化 RAG 的召回率？',
  'ReAct 循环是什么？',
  'KV Cache 如何加速推理？',
  'Graph RAG 比普通 RAG 好在哪？',
]

// ---------- Component ----------

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [backend, setBackend] = useState<BackendType>('rag')
  const [showSettings, setShowSettings] = useState(false)
  const [temperature, setTemperature] = useState(0.1)
  const [topK, setTopK] = useState(5)

  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  // auto-scroll
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  // auto-resize textarea
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
    e.target.style.height = 'auto'
    e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px'
  }

  const sendMessage = useCallback(
    async (text?: string) => {
      const question = (text || input).trim()
      if (!question || loading) return

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        content: question,
        timestamp: new Date(),
      }

      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: '',
        backend,
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, userMsg, assistantMsg])
      setInput('')
      if (inputRef.current) inputRef.current.style.height = 'auto'
      setLoading(true)

      // 取消上一个请求 (防止 StrictMode 双重调用)
      abortRef.current?.abort()
      const controller = new AbortController()
      abortRef.current = controller

      try {
        if (backend === 'rag') {
          await ragQueryStream(
            question,
            (token) => {
              if (controller.signal.aborted) return
              setMessages((prev) => {
                const copy = [...prev]
                const last = copy[copy.length - 1]
                last.content += token
                return copy
              })
            },
            topK,
            temperature,
            controller.signal,
          )
        } else if (backend === 'a2a') {
          // A2A Expert — supports SSE streaming
          await a2aSendSubscribe(
            question,
            (token) => {
              if (controller.signal.aborted) return
              setMessages((prev) => {
                const copy = [...prev]
                const last = copy[copy.length - 1]
                last.content += token
                return copy
              })
            },
            undefined,
            'a2a',
            undefined,
            controller.signal,
          )
        } else {
          // ReAct Agent — no streaming, use sync send
          const task = await a2aSendTask(question, 'react')
          // 回答可能在 status.message / artifacts / history 中，按优先级提取
          const extractText = (parts: unknown) => {
            if (!Array.isArray(parts)) return ''
            return parts.filter((p: { type: string; text: string }) => p.type === 'text')
              .map((p: { type: string; text: string }) => p.text).join('')
          }

          const answer =
            extractText(task.status?.message?.parts) ||
            extractText(task.artifacts?.[0]?.parts) ||
            task.history
              ?.filter((m: { role: string }) => m.role === 'agent')
              .flatMap((m: { role: string; parts: { type: string; text: string }[] }) => m.parts)
              .filter((p: { type: string; text: string }) => p.type === 'text')
              .map((p: { type: string; text: string }) => p.text)
              .join('') ||
            '无回答'
          setMessages((prev) => {
            const copy = [...prev]
            const last = copy[copy.length - 1]
            last.content = answer
            return copy
          })
        }
      } catch (err: unknown) {
        // AbortError 是正常取消（StrictMode remount），不是错误
        if (err instanceof DOMException && err.name === 'AbortError') return
        setMessages((prev) => {
          const copy = [...prev]
          const last = copy[copy.length - 1]
          last.content = `❌ 请求失败: ${err instanceof Error ? err.message : String(err)}\n\n请确保对应的后端服务已启动。`
          return copy
        })
      } finally {
        setLoading(false)
      }
    },
    [input, loading, backend, temperature, topK],
  )

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const clearChat = () => {
    setMessages([])
  }

  // ---------- Render ----------

  return (
    <div className="flex flex-col h-full">
      {/* ---- Header ---- */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-gray-800 bg-gray-900/50 backdrop-blur">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-semibold">Chat</h1>
          {/* Backend selector */}
          <div className="relative">
            <select
              value={backend}
              onChange={(e) => setBackend(e.target.value as BackendType)}
              className="appearance-none bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 pr-8 text-sm text-gray-300 focus:outline-none focus:ring-2 focus:ring-brand-500/50"
            >
              {BACKENDS.map((b) => (
                <option key={b.value} value={b.value}>
                  {b.label}
                </option>
              ))}
            </select>
            <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500 pointer-events-none" />
          </div>
          <span className="text-xs text-gray-500 hidden sm:inline">
            {BACKENDS.find((b) => b.value === backend)?.desc}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-2 rounded-lg text-gray-400 hover:bg-gray-800 hover:text-gray-200 transition-colors"
            title="参数设置"
          >
            <Settings2 className="w-4 h-4" />
          </button>
          <button
            onClick={clearChat}
            className="p-2 rounded-lg text-gray-400 hover:bg-gray-800 hover:text-red-400 transition-colors"
            title="清空聊天"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* ---- Settings Panel ---- */}
      {showSettings && (
        <div className="px-6 py-3 border-b border-gray-800 bg-gray-900/30 flex items-center gap-6 text-sm">
          <label className="flex items-center gap-2">
            <span className="text-gray-400">Temperature:</span>
            <input
              type="range"
              min={0}
              max={2}
              step={0.1}
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-24 accent-brand-500"
            />
            <span className="text-gray-300 w-8">{temperature}</span>
          </label>
          <label className="flex items-center gap-2">
            <span className="text-gray-400">Top K:</span>
            <input
              type="range"
              min={1}
              max={20}
              step={1}
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              className="w-24 accent-brand-500"
            />
            <span className="text-gray-300 w-8">{topK}</span>
          </label>
        </div>
      )}

      {/* ---- Messages ---- */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-4 space-y-6">
        {messages.length === 0 ? (
          <EmptyState onExample={sendMessage} />
        ) : (
          messages.map((msg) => <MessageBubble key={msg.id} message={msg} loading={loading && msg === messages[messages.length - 1] && msg.role === 'assistant'} />)
        )}
      </div>

      {/* ---- Input ---- */}
      <div className="border-t border-gray-800 bg-gray-900/50 px-6 py-4">
        <div className="max-w-3xl mx-auto flex gap-3">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder="输入问题... (Enter 发送, Shift+Enter 换行)"
              rows={1}
              className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 pr-12 text-sm text-gray-100 placeholder-gray-500 resize-none focus:outline-none focus:ring-2 focus:ring-brand-500/50 focus:border-brand-500/50"
            />
          </div>
          <button
            onClick={() => sendMessage()}
            disabled={loading || !input.trim()}
            className="self-end px-4 py-3 rounded-xl bg-brand-600 text-white font-medium text-sm hover:bg-brand-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          </button>
        </div>
      </div>
    </div>
  )
}

// ---------- Sub-components ----------

function EmptyState({ onExample }: { onExample: (text: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center">
      <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-brand-500/20 to-purple-500/20 flex items-center justify-center mb-4">
        <Sparkles className="w-8 h-8 text-brand-400" />
      </div>
      <h2 className="text-xl font-semibold mb-2">Agent Learning Chat</h2>
      <p className="text-gray-400 text-sm mb-8 max-w-md">
        连接到后端 RAG 知识库、A2A Expert Agent 或 ReAct Agent，<br />
        进行智能问答。
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-lg w-full">
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            onClick={() => onExample(ex)}
            className="text-left px-4 py-3 rounded-xl border border-gray-800 bg-gray-900/50 text-sm text-gray-300 hover:bg-gray-800 hover:border-gray-700 transition-colors"
          >
            <BookOpen className="w-3.5 h-3.5 inline mr-2 text-brand-400" />
            {ex}
          </button>
        ))}
      </div>
    </div>
  )
}

function MessageBubble({ message, loading }: { message: ChatMessage; loading: boolean }) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm ${
          isUser
            ? 'bg-brand-600 text-white rounded-tr-md'
            : 'bg-gray-800/80 text-gray-100 rounded-tl-md border border-gray-700/50'
        }`}
      >
        {/* Backend tag */}
        {!isUser && message.backend && (
          <div className="text-[11px] text-gray-500 mb-1 font-medium uppercase tracking-wider">
            {message.backend === 'rag' ? '📚 RAG' : message.backend === 'a2a' ? '🤖 A2A' : '⚡ ReAct'}
          </div>
        )}

        {/* Content */}
        {!isUser && message.content ? (
          <div className="markdown-body">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        ) : (
          <div className="whitespace-pre-wrap">{message.content}</div>
        )}

        {/* Loading dots */}
        {loading && !message.content && (
          <div className="flex gap-1 py-1">
            <span className="typing-dot w-2 h-2 bg-gray-400 rounded-full" />
            <span className="typing-dot w-2 h-2 bg-gray-400 rounded-full" />
            <span className="typing-dot w-2 h-2 bg-gray-400 rounded-full" />
          </div>
        )}

        {/* Sources */}
        {message.sources && message.sources.length > 0 && (
          <details className="mt-2 text-xs">
            <summary className="text-gray-400 cursor-pointer hover:text-gray-300">
              📎 {message.sources.length} 个来源
            </summary>
            <div className="mt-1 space-y-1">
              {message.sources.map((s, i) => (
                <div key={i} className="bg-gray-900/50 rounded p-2">
                  <span className="text-brand-400">[{s.source}]</span> 相关度: {(1 - s.score).toFixed(2)}
                  <p className="text-gray-500 mt-1 line-clamp-2">{s.content}</p>
                </div>
              ))}
            </div>
          </details>
        )}

        {/* Timestamp */}
        <div className={`text-[10px] mt-1 ${isUser ? 'text-brand-200/60' : 'text-gray-600'}`}>
          {message.timestamp.toLocaleTimeString()}
        </div>
      </div>
    </div>
  )
}
