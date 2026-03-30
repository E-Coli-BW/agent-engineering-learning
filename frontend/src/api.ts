// ============================================================
// API 客户端 — 对接后端所有服务
// ============================================================

const RAG_BASE = '/api/rag';   // proxy → localhost:8000
const A2A_BASE = '/api/a2a';   // proxy → localhost:5001
const REACT_BASE = '/api/react'; // proxy → localhost:5002

// ---------- 类型 ----------

export interface SourceDocument {
  content: string;
  source: string;
  score: number;
  metadata: Record<string, string>;
}

export interface QueryResponse {
  answer: string;
  sources: SourceDocument[];
  query_time_ms: number;
  model: string;
  retrieval_count: number;
}

export interface HealthResponse {
  status: string;
  ollama: string;
  vector_store: string;
  vector_count: number;
  models: Record<string, string>;
  timestamp: string;
}

export interface MetricsResponse {
  uptime_seconds: number;
  total_requests: number;
  total_errors: number;
  error_rate: number;
  avg_latency_ms: number;
  vector_count: number;
}

export interface AgentCard {
  name: string;
  description: string;
  url: string;
  version: string;
  capabilities: Record<string, boolean>;
  skills: {
    id: string;
    name: string;
    description: string;
    tags: string[];
    examples: string[];
  }[];
  defaultInputModes: string[];
  defaultOutputModes: string[];
}

export interface A2ATask {
  id: string;
  status: { state: string; message?: Record<string, unknown>; timestamp: string };
  history: { role: string; parts: { type: string; text: string }[] }[];
  artifacts: { parts: { type: string; text: string }[] }[];
  metadata: Record<string, string>;
}

export type BackendType = 'rag' | 'a2a' | 'react';

// ---------- RAG API ----------

export async function ragQuery(question: string, topK = 5, temperature = 0.1): Promise<QueryResponse> {
  const res = await fetch(`${RAG_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK, temperature }),
  });
  if (!res.ok) throw new Error(`RAG query failed: ${res.status}`);
  return res.json();
}

export async function ragQueryStream(
  question: string,
  onToken: (token: string) => void,
  topK = 5,
  temperature = 0.1,
): Promise<void> {
  const res = await fetch(`${RAG_BASE}/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK, temperature, stream: true }),
  });
  if (!res.ok) throw new Error(`RAG stream failed: ${res.status}`);
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buf = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split('\n');
    buf = lines.pop() || '';
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const payload = line.slice(6).trim();
        if (payload === '[DONE]') return;
        try {
          const parsed = JSON.parse(payload);
          if (parsed.token) onToken(parsed.token);
        } catch { /* skip */ }
      }
    }
  }
}

export async function ragHealth(): Promise<HealthResponse> {
  const res = await fetch(`${RAG_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function ragMetrics(): Promise<MetricsResponse> {
  const res = await fetch(`${RAG_BASE}/metrics`);
  if (!res.ok) throw new Error(`Metrics failed: ${res.status}`);
  return res.json();
}

export async function ragETLRun(sources: string[], incremental = true): Promise<Record<string, unknown>> {
  const res = await fetch(`${RAG_BASE}/etl/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sources, incremental }),
  });
  if (!res.ok) throw new Error(`ETL failed: ${res.status}`);
  return res.json();
}

// ---------- A2A Agent ----------

function a2aBase(backend: 'a2a' | 'react') {
  return backend === 'a2a' ? A2A_BASE : REACT_BASE;
}

export async function a2aGetCard(backend: 'a2a' | 'react' = 'a2a'): Promise<AgentCard> {
  const res = await fetch(`${a2aBase(backend)}/.well-known/agent.json`);
  if (!res.ok) throw new Error(`Agent card failed: ${res.status}`);
  return res.json();
}

export async function a2aSendTask(
  question: string,
  backend: 'a2a' | 'react' = 'a2a',
  skill?: string,
): Promise<A2ATask> {
  const res = await fetch(`${a2aBase(backend)}/tasks/send`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      id: `task-${crypto.randomUUID().slice(0, 8)}`,
      message: { role: 'user', parts: [{ type: 'text', text: question }] },
      metadata: skill ? { skill } : {},
    }),
  });
  if (!res.ok) throw new Error(`A2A send failed: ${res.status}`);
  return res.json();
}

export async function a2aSendSubscribe(
  question: string,
  onToken: (token: string) => void,
  onStatus?: (state: string) => void,
  backend: 'a2a' | 'react' = 'a2a',
  skill?: string,
): Promise<void> {
  const res = await fetch(`${a2aBase(backend)}/tasks/sendSubscribe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({
      id: `task-${crypto.randomUUID().slice(0, 8)}`,
      message: { role: 'user', parts: [{ type: 'text', text: question }] },
      metadata: skill ? { skill } : {},
    }),
  });
  if (!res.ok) throw new Error(`A2A stream failed: ${res.status}`);

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  let currentEvent = '';
  let currentData = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split('\n');
    buf = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('event: ')) {
        currentEvent = line.slice(7).trim();
      } else if (line.startsWith('data: ')) {
        currentData = line.slice(6);
      } else if (line === '') {
        // 空行 = SSE 事件结束，分发并重置
        if (currentEvent && currentData) {
          try {
            const data = JSON.parse(currentData);
            if (currentEvent === 'artifact') {
              for (const part of data.parts || []) {
                if (part.type === 'text' && part.text) onToken(part.text);
              }
            } else if (currentEvent === 'status') {
              onStatus?.(data.state);
            }
          } catch { /* skip malformed JSON */ }
        }
        currentEvent = '';
        currentData = '';
      }
    }
  }
}

export async function a2aGetTask(taskId: string, backend: 'a2a' | 'react' = 'a2a'): Promise<A2ATask> {
  const res = await fetch(`${a2aBase(backend)}/tasks/${taskId}`);
  if (!res.ok) throw new Error(`Get task failed: ${res.status}`);
  return res.json();
}

export async function a2aHealth(backend: 'a2a' | 'react' = 'a2a'): Promise<{ status: string; agent: string; model: string }> {
  const res = await fetch(`${a2aBase(backend)}/health`);
  if (!res.ok) throw new Error(`A2A health failed: ${res.status}`);
  return res.json();
}
