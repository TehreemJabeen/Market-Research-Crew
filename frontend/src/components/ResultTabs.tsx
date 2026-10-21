import { useState } from 'react'
import type { RunDetail } from '../lib/types'
import ClaimsView from './ClaimsView'
import DataTable from './DataTable'
import OpinionsView from './OpinionsView'
import ReportView from './ReportView'
import RunInfo from './RunInfo'

export default function ResultTabs({ run }: { run: RunDetail }) {
  const r = run.result
  const gaps = r?.gaps ?? []
  const tabs = [
    { id: 'report', label: 'Report', show: !!r?.report },
    { id: 'compare', label: 'Comparison', show: !!r?.table },
    { id: 'users', label: 'What users say', show: !!(r?.themes?.length || r?.sentiment_table) },
    { id: 'evidence', label: 'Facts vs user evidence', show: !!r?.evidence_table },
    { id: 'sources', label: 'Sources and facts', show: !!r?.claims?.length || !!r?.final_citations?.length },
    { id: 'gaps', label: `Gaps (${gaps.length})`, show: true },
    { id: 'info', label: 'Run info', show: true },
  ].filter((t) => t.show)
  const [active, setActive] = useState(tabs[0].id)
  const current = tabs.some((t) => t.id === active) ? active : tabs[0].id

  return (
    <div>
      <div role="tablist" aria-label="Results" className="mb-6 flex flex-wrap gap-1 border-b border-rule">
        {tabs.map((t) => (
          <button
            key={t.id}
            role="tab"
            id={`tab-${t.id}`}
            aria-selected={current === t.id}
            aria-controls={`panel-${t.id}`}
            onClick={() => setActive(t.id)}
            className={`-mb-px border-b-2 px-3 py-2 text-sm ${
              current === t.id ? 'border-accent font-medium text-ink' : 'border-transparent text-muted hover:text-ink'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div role="tabpanel" id={`panel-${current}`} aria-labelledby={`tab-${current}`}>
        {current === 'report' && r?.report && <ReportView markdown={r.report} filename={`report-${run.id}.md`} />}
        {current === 'compare' && r?.table && (
          <div className="space-y-3">
            <p className="max-w-prose text-sm text-muted">
              Built from the collected facts. “Not found” means no source gave a fact for that cell.
            </p>
            <DataTable table={r.table} />
          </div>
        )}
        {current === 'users' && <OpinionsView themes={r?.themes ?? []} sentiment={r?.sentiment_table} />}
        {current === 'evidence' && r?.evidence_table && (
          <div className="space-y-3">
            <p className="max-w-prose text-sm text-muted">
              Each fact next to what users say about the same product and dimension: confirmed, qualified, disputed, or no user evidence.
            </p>
            <DataTable table={r.evidence_table} />
          </div>
        )}
        {current === 'sources' && <ClaimsView claims={r?.claims ?? []} citations={r?.final_citations ?? []} />}
        {current === 'gaps' &&
          (gaps.length ? (
            <ul className="max-w-prose list-disc space-y-1.5 pl-5">
              {gaps.map((g, i) => (
                <li key={i}>{g}</li>
              ))}
            </ul>
          ) : (
            <p className="text-muted">The analyst reported no gaps.</p>
          ))}
        {current === 'info' && <RunInfo run={run} />}
      </div>
    </div>
  )
}
