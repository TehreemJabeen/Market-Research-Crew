import type { Table, Theme } from '../lib/types'
import { host } from '../lib/format'
import DataTable from './DataTable'

const STANCE: Record<Theme['stance'], string> = {
  positive: 'text-ok border-ok',
  negative: 'text-bad border-bad',
  mixed: 'text-warn border-warn',
  neutral: 'text-muted border-rule',
}

export default function OpinionsView({ themes, sentiment }: { themes: Theme[]; sentiment?: Table | null }) {
  const byEntity = themes.reduce<Record<string, Theme[]>>((acc, t) => ((acc[t.entity] ||= []).push(t), acc), {})
  return (
    <div className="space-y-8">
      <p className="max-w-prose text-sm text-muted">
        Counted per discussion thread, so one busy thread cannot dominate. This is a small, self-selected sample of Hacker News and
        Reddit, not the whole market.
      </p>
      {sentiment && <DataTable table={sentiment} caption="Sentiment by product and dimension" />}
      {themes.length === 0 && <p className="text-muted">No recurring theme appeared in at least two threads.</p>}
      {Object.entries(byEntity).map(([entity, list]) => (
        <section key={entity}>
          <h3 className="mb-3 text-lg font-semibold">{entity}</h3>
          <ul className="space-y-4">
            {list.map((t) => (
              <li key={t.id} className="rounded-md border border-rule bg-surface p-4">
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                  <span className="font-medium">{t.label}</span>
                  <span className={`rounded-full border px-2 py-0.5 text-xs font-medium ${STANCE[t.stance]}`}>{t.stance}</span>
                  <span className="text-sm text-muted">{t.dimension}</span>
                  <span className="text-sm text-muted">
                    {t.n_threads} threads, {t.confidence} confidence
                  </span>
                </div>
                {t.examples.map((e, i) => (
                  <blockquote key={i} className="mt-3 border-l-2 border-rule pl-3 text-sm">
                    “{e.quote}”{' '}
                    <a href={e.url} target="_blank" rel="noopener noreferrer" className="text-accent underline underline-offset-4">
                      {host(e.url)}
                    </a>
                  </blockquote>
                ))}
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  )
}
