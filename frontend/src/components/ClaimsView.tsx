import type { Citation, Claim } from '../lib/types'
import { host } from '../lib/format'

function Source({ claim }: { claim: Claim }) {
  const app = claim.source_type === 'appstore'
  return (
    <div className="space-y-1">
      <a href={claim.source_url} target="_blank" rel="noopener noreferrer" className="text-accent underline underline-offset-4">
        {host(claim.source_url)}
      </a>
      <div className="flex flex-wrap gap-1.5">
        <span className="rounded-full bg-accent-soft px-2 py-0.5 text-xs">{app ? 'App Store' : 'Official site'}</span>
        {!app && <span className="rounded-full border border-warn px-2 py-0.5 text-xs text-warn">Not fact-checked</span>}
      </div>
    </div>
  )
}

export default function ClaimsView({ claims, citations }: { claims: Claim[]; citations: Citation[] }) {
  const sorted = [...claims].sort((a, b) => a.entity.localeCompare(b.entity) || a.dimension.localeCompare(b.dimension))
  return (
    <div className="space-y-10">
      <section>
        <h3 className="text-lg font-semibold">Cited in the report</h3>
        <p className="mb-3 mt-1 max-w-prose text-sm text-muted">
          The report’s bracketed numbers point here. Each quote comes from the page or post it links to.
        </p>
        {citations.length === 0 ? (
          <p className="text-muted">The report cites no sources.</p>
        ) : (
          <ol className="space-y-3">
            {citations.map((c, i) => (
              <li key={i} className="flex gap-3">
                <span className="w-7 shrink-0 text-right font-medium text-muted">{c.n}</span>
                <div className="min-w-0 text-sm">
                  <span>“{c.quote}”</span>{' '}
                  <a href={c.url} target="_blank" rel="noopener noreferrer" className="break-all text-accent underline underline-offset-4">
                    {host(c.url)}
                  </a>
                </div>
              </li>
            ))}
          </ol>
        )}
      </section>
      <section>
        <h3 className="mb-3 text-lg font-semibold">All facts the crew collected ({claims.length})</h3>
        <div className="overflow-x-auto rounded-md border border-rule bg-surface">
          <table className="w-full border-collapse text-left text-sm">
            <thead>
              <tr className="bg-accent-soft">
                {['Product', 'Dimension', 'Fact', 'Source'].map((h) => (
                  <th key={h} scope="col" className="border-b border-rule px-3 py-2 font-semibold">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.map((c) => (
                <tr key={c.id} className="align-top">
                  <td className="border-b border-rule px-3 py-2 font-medium">{c.entity}</td>
                  <td className="border-b border-rule px-3 py-2">{c.dimension}</td>
                  <td className="min-w-64 border-b border-rule px-3 py-2">{c.text}</td>
                  <td className="border-b border-rule px-3 py-2">
                    <Source claim={c} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}
