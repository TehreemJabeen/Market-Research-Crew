import { useMutation, useQuery } from '@tanstack/react-query'
import { FormEvent, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../lib/api'
import { parseSites } from '../lib/format'
import type { RunRequest } from '../lib/types'

const EXAMPLES = [
  'Compare Notion, Coda and ClickUp for small teams',
  'Duolingo vs Babbel for casual language learners',
  'Linear or Jira for a 10-person startup under $100 a month',
]

const field = 'w-full rounded-md border border-rule bg-surface px-3 py-2 text-ink placeholder:text-muted/70'

export default function NewRun() {
  const navigate = useNavigate()
  const { data: config } = useQuery({ queryKey: ['config'], queryFn: api.config })
  const [question, setQuestion] = useState('')
  const [dimensions, setDimensions] = useState('')
  const [sites, setSites] = useState('')
  const [provider, setProvider] = useState('')
  const [apps, setApps] = useState<'auto' | 'on' | 'off'>('auto')
  const [community, setCommunity] = useState(true)
  const [sentiment, setSentiment] = useState('')

  const start = useMutation({
    mutationFn: (body: RunRequest) => api.createRun(body),
    onSuccess: ({ run_id }) => navigate(`/runs/${run_id}`),
  })

  const noKeys = config && !config.demo && !config.providers.gemini && !config.providers.grok
  const submit = (e: FormEvent) => {
    e.preventDefault()
    start.mutate({
      question,
      dimensions: dimensions.split(',').map((d) => d.trim()).filter(Boolean),
      sites: parseSites(sites),
      provider: (provider || null) as RunRequest['provider'],
      apps,
      community,
      sentiment_backend: (sentiment || null) as RunRequest['sentiment_backend'],
    })
  }

  return (
    <div className="max-w-2xl">
      <h1 className="text-3xl font-semibold leading-tight tracking-tight">What do you want to compare?</h1>
      <p className="mt-2 text-muted">
        Ask about products, tools or apps. The crew reads each product’s own website, gathers what people say on Hacker News and
        Reddit, and writes a cited report that keeps facts and opinions apart.
      </p>

      {noKeys && (
        <p role="alert" className="mt-6 rounded-md border border-warn px-4 py-3 text-sm text-warn">
          No model API key found. Add GEMINI_API_KEY (or XAI_API_KEY) to the .env file and restart the server.
        </p>
      )}

      <form onSubmit={submit} className="mt-8 space-y-6">
        <div>
          <label htmlFor="q" className="mb-1.5 block font-medium">
            Research question
          </label>
          <textarea
            id="q"
            required
            minLength={8}
            maxLength={500}
            rows={3}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Compare Notion, Coda and ClickUp for small teams"
            className={field}
          />
          <div className="mt-2 flex flex-wrap gap-2">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                type="button"
                onClick={() => setQuestion(ex)}
                className="rounded-full border border-rule px-3 py-1 text-sm text-muted hover:border-accent hover:text-ink"
              >
                {ex}
              </button>
            ))}
          </div>
        </div>

        <details className="rounded-md border border-rule bg-surface">
          <summary className="cursor-pointer px-4 py-3 font-medium">Options</summary>
          <div className="grid gap-5 border-t border-rule px-4 py-5 sm:grid-cols-2">
            <div className="sm:col-span-2">
              <label htmlFor="dims" className="mb-1 block text-sm font-medium">
                Dimensions you insist on
              </label>
              <input id="dims" value={dimensions} onChange={(e) => setDimensions(e.target.value)} placeholder="pricing, security" className={field} />
              <p className="mt-1 text-sm text-muted">Comma-separated. The planner adds the rest.</p>
            </div>
            <div className="sm:col-span-2">
              <label htmlFor="sites" className="mb-1 block text-sm font-medium">
                Official sites to use instead of the automatic lookup
              </label>
              <input id="sites" value={sites} onChange={(e) => setSites(e.target.value)} placeholder="notion=notion.com, coda=coda.io" className={field} />
              <p className="mt-1 text-sm text-muted">Use this when the run info shows the wrong site for a product.</p>
            </div>
            <div>
              <label htmlFor="prov" className="mb-1 block text-sm font-medium">
                Model
              </label>
              <select id="prov" value={provider} onChange={(e) => setProvider(e.target.value)} className={field}>
                <option value="">Server default{config ? ` (${config.default_provider})` : ''}</option>
                <option value="gemini" disabled={config ? !config.providers.gemini : false}>
                  Gemini{config && !config.providers.gemini ? ' (no key)' : ''}
                </option>
                <option value="grok" disabled={config ? !config.providers.grok : false}>
                  Grok{config && !config.providers.grok ? ' (no key)' : ''}
                </option>
              </select>
            </div>
            <div>
              <label htmlFor="sent" className="mb-1 block text-sm font-medium">
                Sentiment scoring
              </label>
              <select id="sent" value={sentiment} onChange={(e) => setSentiment(e.target.value)} className={field}>
                <option value="">Server default</option>
                <option value="hf" disabled={config ? !config.local_sentiment_model : false}>
                  Local model{config && !config.local_sentiment_model ? ' (not installed)' : ''}
                </option>
                <option value="llm">Language model</option>
              </select>
            </div>
            <fieldset>
              <legend className="mb-1 text-sm font-medium">App Store data</legend>
              <div className="flex gap-4 text-sm">
                {(['auto', 'on', 'off'] as const).map((v) => (
                  <label key={v} className="flex items-center gap-1.5">
                    <input type="radio" name="apps" checked={apps === v} onChange={() => setApps(v)} className="accent-accent" />
                    {v === 'auto' ? 'Automatic' : v === 'on' ? 'Always' : 'Never'}
                  </label>
                ))}
              </div>
              <p className="mt-1 text-sm text-muted">Automatic adds it when the products are apps.</p>
            </fieldset>
            <div className="flex items-start">
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={community} onChange={(e) => setCommunity(e.target.checked)} className="accent-accent" />
                Include what people say on Hacker News and Reddit
              </label>
            </div>
          </div>
        </details>

        {start.error && (
          <p role="alert" className="text-sm text-bad">
            {(start.error as Error).message}
          </p>
        )}
        <div className="flex items-center gap-4">
          <button
            type="submit"
            disabled={start.isPending || !!noKeys}
            className="rounded-md bg-accent px-5 py-2.5 font-medium text-paper hover:opacity-90 disabled:opacity-50"
          >
            {start.isPending ? 'Starting…' : 'Start research'}
          </button>
          <p className="text-sm text-muted">On free API tiers a full run takes several minutes. You can leave this page and come back.</p>
        </div>
      </form>
    </div>
  )
}
