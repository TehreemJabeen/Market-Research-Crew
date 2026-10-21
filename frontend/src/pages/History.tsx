import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import StatusBadge from '../components/StatusBadge'
import { api } from '../lib/api'
import { duration, when } from '../lib/format'
import { TERMINAL } from '../lib/types'

export default function History() {
  const qc = useQueryClient()
  const { data, isLoading, error } = useQuery({ queryKey: ['runs'], queryFn: api.listRuns, refetchInterval: 5000 })
  const remove = useMutation({
    mutationFn: api.deleteRun,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['runs'] }),
  })

  if (isLoading) return <p className="text-muted">Loading runs…</p>
  if (error) return <p role="alert" className="text-bad">Could not load runs: {(error as Error).message}</p>
  if (!data?.length)
    return (
      <div className="max-w-xl py-8">
        <h1 className="text-2xl font-semibold">No runs yet</h1>
        <p className="mt-2 text-muted">
          Start with a question such as “Compare Notion, Coda and ClickUp for small teams”.{' '}
          <Link to="/" className="text-accent underline underline-offset-4">
            Start a run
          </Link>
        </p>
      </div>
    )

  return (
    <div>
      <h1 className="text-2xl font-semibold">History</h1>
      <ul className="mt-6 divide-y divide-rule border-y border-rule">
        {data.map((run) => (
          <li key={run.id} className="flex flex-wrap items-center gap-x-4 gap-y-2 py-4">
            <div className="min-w-0 flex-1 basis-80">
              <Link to={`/runs/${run.id}`} className="font-medium hover:text-accent">
                {run.question}
              </Link>
              <p className="mt-0.5 flex flex-wrap gap-x-4 text-sm text-muted">
                <span>{when(run.created_at)}</span>
                {run.seconds != null && <span>{duration(run.seconds)}</span>}
                {run.llm_calls != null && <span>{run.llm_calls} model calls</span>}
                {run.tavily_calls != null && <span>{run.tavily_calls} searches</span>}
              </p>
            </div>
            <StatusBadge status={run.status} />
            {TERMINAL.includes(run.status) && (
              <button
                type="button"
                onClick={() => remove.mutate(run.id)}
                className="text-sm text-muted underline underline-offset-4 hover:text-bad"
                aria-label={`Delete run: ${run.question}`}
              >
                Delete
              </button>
            )}
          </li>
        ))}
      </ul>
    </div>
  )
}
