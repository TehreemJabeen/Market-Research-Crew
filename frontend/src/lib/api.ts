import type { AppConfig, RunDetail, RunRequest, RunSummary } from './types'

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...init })
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      const detail = body.detail
      if (typeof detail === 'string') message = detail
      else if (Array.isArray(detail)) message = detail.map((d: { msg?: string }) => d.msg).filter(Boolean).join('; ')
    } catch {
      /* keep the status text */
    }
    throw new Error(message)
  }
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T)
}

export const api = {
  config: () => call<AppConfig>('/api/config'),
  listRuns: () => call<RunSummary[]>('/api/runs'),
  getRun: (id: string) => call<RunDetail>(`/api/runs/${id}`),
  createRun: (body: RunRequest) => call<{ run_id: string }>('/api/runs', { method: 'POST', body: JSON.stringify(body) }),
  cancelRun: (id: string) => call<{ cancelled: boolean }>(`/api/runs/${id}/cancel`, { method: 'POST' }),
  deleteRun: (id: string) => call<void>(`/api/runs/${id}`, { method: 'DELETE' }),
}
