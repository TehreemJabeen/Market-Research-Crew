import { duration } from '../lib/format'
import type { RunDetail } from '../lib/types'

function Item({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid gap-1 border-b border-rule py-3 sm:grid-cols-[12rem_1fr]">
      <dt className="text-muted">{label}</dt>
      <dd className="min-w-0 break-words">{children}</dd>
    </div>
  )
}

export default function RunInfo({ run }: { run: RunDetail }) {
  const u = run.usage
  const plan = run.result?.plan
  const tokens = (u?.prompt_tokens ?? 0) + (u?.completion_tokens ?? 0)
  const opt = run.options
  return (
    <dl className="max-w-3xl text-sm">
      <Item label="Time taken">{duration(u?.seconds ?? run.seconds)}</Item>
      <Item label="Model calls">{u?.llm_calls ?? '–'}</Item>
      <Item label="Tokens">{tokens ? tokens.toLocaleString() : 'Not reported by this model'}</Item>
      <Item label="Web searches">{u?.tavily_calls ?? '–'}</Item>
      {plan && (
        <>
          <Item label="Question as planned">{plan.rewritten_question}</Item>
          <Item label="Audience">{plan.audience || 'Not stated'}</Item>
          <Item label="Products">{plan.entities.join(', ')}</Item>
          <Item label="Dimensions">{plan.dimensions.join(', ')}</Item>
          <Item label="Official sites used">
            {Object.keys(plan.official_sites ?? {}).length === 0 ? (
              'None found'
            ) : (
              <ul className="space-y-0.5">
                {Object.entries(plan.official_sites).map(([entity, domains]) => (
                  <li key={entity}>
                    {entity}: {domains.join(', ')}
                  </li>
                ))}
              </ul>
            )}
            <p className="mt-1 text-muted">Wrong site? Start a new run and set it under Options.</p>
          </Item>
        </>
      )}
      <Item label="Options you set">
        {[
          opt.dimensions?.length ? `dimensions: ${opt.dimensions.join(', ')}` : null,
          opt.sites && Object.keys(opt.sites).length ? `sites: ${Object.entries(opt.sites).map(([k, v]) => `${k}=${v}`).join(', ')}` : null,
          `App Store data: ${opt.apps ?? 'auto'}`,
          opt.community === false ? 'community opinions off' : null,
          opt.provider ? `model: ${opt.provider}` : null,
        ]
          .filter(Boolean)
          .join('; ')}
      </Item>
    </dl>
  )
}
