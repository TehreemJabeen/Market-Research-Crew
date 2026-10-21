import { useEffect, useState } from 'react'
import type { ServerEvent, Status } from '../lib/types'
import { TERMINAL } from '../lib/types'

const NAMES = ['step', 'status', 'done', 'failed', 'stopped', 'cancelled']

/** Subscribes to a run's event stream. A finished run replays its whole log and then the stream closes, so a page
 *  refresh rebuilds the same timeline. The browser reconnects by itself and resumes from the last event id. */
export function useRunEvents(runId: string, onTerminal: () => void) {
  const [events, setEvents] = useState<ServerEvent[]>([])
  const [live, setLive] = useState<Status | null>(null)

  useEffect(() => {
    setEvents([])
    setLive(null)
    const source = new EventSource(`/api/runs/${runId}/events`)
    const handle = (type: string) => (msg: MessageEvent) => {
      const seq = Number(msg.lastEventId)
      let data: Record<string, unknown> = {}
      try {
        data = JSON.parse(msg.data)
      } catch {
        /* ignore malformed data */
      }
      setEvents((prev) => (prev.some((e) => e.seq === seq) ? prev : [...prev, { seq, type, data }]))
      if (type === 'status') setLive(data.status as Status)
      if (TERMINAL.includes(type as Status)) {
        setLive(type as Status)
        source.close()
        onTerminal()
      }
    }
    NAMES.forEach((name) => source.addEventListener(name, handle(name) as EventListener))
    return () => source.close()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId])

  return { events, live }
}
