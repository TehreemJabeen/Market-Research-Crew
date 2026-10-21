import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo } from 'react'
import { Link, useParams } from 'react-router-dom'
import Pipeline from '../components/Pipeline'
import ResultTabs from '../components/ResultTabs'
import StatusBadge from '../components/StatusBadge'
import { useRunEvents } from '../hooks/useRunEvents'
import { api } from '../lib/api'
import { duration, when } from '../lib/format'
import { derivePipeline } from '../lib/pipeline'
import { TERMINAL } from '../lib/types'

export default function RunPage() {
  const { id = '' } = useParams()
  const qc = useQueryClient()
  const run = useQuery({ queryKey: ['run', id], queryFn: () => api.getRun(id) })
  const { events, live } = useRunEvents(id, () => qc.invalidateQueries({ queryKey: ['run', id] }))
  const cancel = useMutation({ mutationFn: () => api.cancelRun(id) })

  const status = live && !TERMINAL.includes(run.data?.status ?? 'queued') ? live : (run.data?.status ?? live ?? 'queued')
  const pipeline = useMemo(() => derivePipeline(events, status), [events, status])

  if (run.isLoading) return <p className="text-muted">Loading run…</p>
  if (run.error)
    return (
      <div role="alert" className="max-w-xl py-8">
        <h1 className="text-xl font-semibold">Run not found</h1>
        <p className="mt-2 text-muted">It may have been deleted. <Link to="/history" className="text-accent underline underline-offset-4">Back to history</Link></p>
      </div>
    )
  const data = run.data!
  const active = status === 'queued' || status === 'running'
  const plan = pipeline.plan
  const queued = events.filter((e) => e.type === 'status' && e.data.status === 'queued').at(-1)

  return (
    <div>
      <div className="mb-8 max-w-3xl">
        <div className="flex flex-wrap items-center gap-3">
          <StatusBadge status={status} />
          <span className="text-sm text-muted">Started {when(data.created_at)}</span>
          {data.seconds != null && <span className="text-sm text-muted">{duration(data.seconds)}</span>}
        </div>
        <h1 className="mt-3 font-serif text-2xl font-semibold leading-snug">{data.question}</h1>
        {active && (
          <div className="mt-4 flex items-center gap-4">
            <button
              type="button"
              disabled={cancel.isPending || cancel.isSuccess}
              onClick={() => cancel.mutate()}
              className="rounded-md border border-rule px-3 py-1.5 text-sm hover:border-bad hover:text-bad disabled:opacity-60"
            >
              {cancel.isSuccess ? 'Stopping after the current step…' : 'Stop run'}
            </button>
            {status === 'queued' && queued && <span className="text-sm text-muted">Waiting for the run ahead to finish.</span>}
          </div>
        )}
      </div>

      <div className="grid gap-10 lg:grid-cols-[22rem_minmax(0,1fr)]">
        <aside>
          <h2 className="mb-4 text-sm font-semibold text-muted">Progress</h2>
          <Pipeline data={pipeline} />
        </aside>

        <section aria-live="polite" className="min-w-0">
          {data.result?.sample && (
            <p className="mb-6 rounded-md border border-warn px-4 py-3 text-sm text-warn">
              This is recorded sample data from a demo run, not a live result for your question.
            </p>
          )}
          {plan && (
            <div className="mb-8 max-w-prose">
              <h2 className="text-lg font-semibold">What the crew is comparing</h2>
              <p className="mt-1">
                {plan.entities.join(', ')} on {plan.dimensions.join(', ')}
                {plan.audience ? `, for ${plan.audience}` : ''}.
              </p>
            </div>
          )}
          {status === 'done' && data.result && <ResultTabs run={data} />}
          {active && !plan && <p className="text-muted">{status === 'queued' ? 'Queued.' : 'Planning the research…'}</p>}
          {active && plan && <p className="text-muted">The report appears here when the run finishes.</p>}
          {(status === 'failed' || status === 'stopped') && (
            <div role="alert" className="max-w-prose rounded-md border border-bad px-4 py-3">
              <p className="font-medium text-bad">{status === 'stopped' ? 'The run hit its call limit.' : 'The run failed.'}</p>
              <p className="mt-1 text-sm break-words">{data.error}</p>
              <p className="mt-2 text-sm text-muted">
                Check the keys in .env and the server log, then{' '}
                <Link to="/" className="text-accent underline underline-offset-4">start a new run</Link>.
              </p>
            </div>
          )}
          {status === 'cancelled' && <p className="text-muted">You stopped this run. Nothing was saved from it.</p>}
        </section>
      </div>
    </div>
  )
}
