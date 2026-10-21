import type { Status } from '../lib/types'

const LOOK: Record<Status, { label: string; cls: string }> = {
  queued: { label: 'Queued', cls: 'border-rule text-muted' },
  running: { label: 'Running', cls: 'border-accent text-accent' },
  done: { label: 'Done', cls: 'border-ok text-ok' },
  failed: { label: 'Failed', cls: 'border-bad text-bad' },
  stopped: { label: 'Stopped', cls: 'border-warn text-warn' },
  cancelled: { label: 'Cancelled', cls: 'border-rule text-muted' },
}

export default function StatusBadge({ status }: { status: Status }) {
  const { label, cls } = LOOK[status]
  return <span className={`inline-block rounded-full border px-2.5 py-0.5 text-xs font-medium ${cls}`}>{label}</span>
}
