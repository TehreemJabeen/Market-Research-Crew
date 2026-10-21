export type Status = 'queued' | 'running' | 'done' | 'failed' | 'stopped' | 'cancelled'
export const TERMINAL: Status[] = ['done', 'failed', 'stopped', 'cancelled']

export interface RunOptions {
  dimensions?: string[]
  sites?: Record<string, string>
  provider?: 'gemini' | 'grok' | null
  community?: boolean
  apps?: 'auto' | 'on' | 'off'
  sentiment_backend?: 'hf' | 'llm' | null
}

export interface RunSummary {
  id: string
  question: string
  status: Status
  created_at: number
  finished_at: number | null
  seconds: number | null
  llm_calls: number | null
  tavily_calls: number | null
}

export interface Usage {
  llm_calls?: number
  prompt_tokens?: number
  completion_tokens?: number
  seconds?: number
  tavily_calls?: number
}

export interface Table {
  columns: string[]
  rows: string[][]
}

export interface Claim {
  id: string
  entity: string
  dimension: string
  text: string
  source_url: string
  supporting_quote: string
  check_note: string
  source_type: 'web' | 'appstore' | string
}

export interface Example {
  url: string
  quote: string
  source: string
}

export interface Theme {
  id: string
  entity: string
  dimension: string
  label: string
  n_threads: number
  n_snippets: number
  stance: 'positive' | 'negative' | 'mixed' | 'neutral'
  confidence: 'low' | 'medium' | 'high'
  examples: Example[]
}

export interface Citation {
  n: number
  claim_id: string
  url: string
  quote: string
}

export interface Plan {
  rewritten_question: string
  audience: string
  decision: string
  entities: string[]
  dimensions: string[]
  is_app: boolean
  official_sites: Record<string, string[]>
}

export interface RunResult {
  report?: string
  final_citations?: Citation[]
  gaps?: string[]
  table?: Table | null
  evidence_table?: Table | null
  sentiment_table?: Table | null
  themes?: Theme[]
  claims?: Claim[]
  plan?: Plan
  sample?: boolean
}

export interface RunDetail extends RunSummary {
  options: RunOptions
  error: string | null
  usage: Usage | null
  result: RunResult | null
}

export interface PlanInfo {
  rewritten_question: string
  audience: string
  entities: string[]
  dimensions: string[]
  is_app: boolean
  n_web: number
  n_community: number
  n_app: number
}

export interface StepEvent {
  node: string
  trace: string[]
  plan?: PlanInfo
  official_sites?: Record<string, string[]>
  n_claims?: number
  n_snippets?: number
}

export interface ServerEvent {
  seq: number
  type: string
  data: Record<string, unknown>
}

export interface AppConfig {
  demo: boolean
  providers: { gemini: boolean; grok: boolean }
  default_provider: string
  tavily: boolean
  local_sentiment_model: boolean
  app_country: string
}

export interface RunRequest extends RunOptions {
  question: string
}
