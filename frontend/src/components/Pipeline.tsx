import type { Pipeline as PipelineData, Stage, StageState } from '../lib/pipeline'

function Marker({ state }: { state: StageState }) {
  const base = 'relative z-10 mt-1 grid h-4 w-4 shrink-0 place-items-center rounded-full border-2 bg-paper'
  if (state === 'done')
    return (
      <span className={`${base} border-ok bg-ok`} aria-hidden>
        <svg viewBox="0 0 12 12" className="h-2.5 w-2.5 text-paper" fill="none" stroke="currentColor" strokeWidth="2.2">
          <path d="M2.5 6.5l2.2 2.2 4.8-5" />
        </svg>
      </span>
    )
  if (state === 'running') return <span className={`${base} running-dot border-accent bg-accent`} aria-hidden />
  if (state === 'stopped') return <span className={`${base} border-warn`} aria-hidden />
  return <span className={`${base} border-rule`} aria-hidden />
}

const STATE_TEXT: Record<StageState, string> = { waiting: 'Waiting', running: 'Working', done: 'Done', stopped: 'Stopped here' }

function Row({ stage, lane = false }: { stage: Stage; lane?: boolean }) {
  return (
    <li className="relative flex gap-3 pb-5 last:pb-0">
      <Marker state={stage.state} />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-baseline gap-x-2">
          <span className={`font-medium ${stage.state === 'waiting' ? 'text-muted' : 'text-ink'} ${lane ? 'text-[0.95rem]' : ''}`}>
            {stage.title}
          </span>
          {stage.count && (
            <span className="text-sm text-muted">
              {stage.count.done} of {stage.count.total}
            </span>
          )}
          <span className="sr-only">{STATE_TEXT[stage.state]}</span>
        </div>
        <p className="text-sm text-muted">{stage.detail}</p>
        {stage.lines.length > 0 && (
          <details className="mt-1 text-sm">
            <summary className="cursor-pointer text-muted hover:text-ink">What it did ({stage.lines.length})</summary>
            <ul className="mt-1 space-y-1 border-l border-rule pl-3 text-muted">
              {stage.lines.map((line, i) => (
                <li key={i} className="break-words">
                  {line}
                </li>
              ))}
            </ul>
          </details>
        )}
      </div>
    </li>
  )
}

/** A vertical track. The research steps share one bracketed block because they run at the same time. */
export default function Pipeline({ data }: { data: PipelineData }) {
  const [plan, sites, ...rest] = data.stages
  const after = rest
  return (
    <ol className="relative" aria-label="Progress">
      <span className="absolute bottom-3 left-[7px] top-3 w-px bg-rule" aria-hidden />
      <Row stage={plan} />
      <Row stage={sites} />
      <li className="relative pb-5">
        <div className="ml-0 rounded-md border border-dashed border-rule bg-surface/60 py-3 pl-3 pr-3">
          <p className="mb-3 pl-7 text-sm text-muted">Run at the same time</p>
          <ol className="relative">
            {data.lanes.map((lane) => (
              <Row key={lane.id} stage={lane} lane />
            ))}
          </ol>
        </div>
      </li>
      {after.map((stage) => (
        <Row key={stage.id} stage={stage} />
      ))}
    </ol>
  )
}
