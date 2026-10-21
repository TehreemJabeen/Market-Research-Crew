import type { PlanInfo, ServerEvent, Status, StepEvent } from './types'

export type StageState = 'waiting' | 'running' | 'done' | 'stopped'

export interface Stage {
  id: string
  title: string
  detail: string
  state: StageState
  count?: { done: number; total: number }
  lines: string[]
}

export interface Pipeline {
  stages: Stage[]
  lanes: Stage[] // research steps that run side by side
  plan: PlanInfo | null
}

const steps = (events: ServerEvent[]): StepEvent[] =>
  events.filter((e) => e.type === 'step').map((e) => e.data as unknown as StepEvent)

/** Turns the raw event log into the stages shown on the run page. Nothing here is stored: it is recomputed from the events. */
export function derivePipeline(events: ServerEvent[], status: Status): Pipeline {
  const all = steps(events)
  const of = (node: string) => all.filter((s) => s.node === node)
  const lines = (node: string) => of(node).flatMap((s) => s.trace)
  const plan = of('planner')[0]?.plan ?? null
  const live = status === 'running' || status === 'queued'
  const finished = status === 'done'

  const make = (id: string, title: string, detail: string, done: boolean, ready: boolean, total?: number): Stage => {
    const doneCount = of(id).length
    const state: StageState = done ? 'done' : !live ? (ready ? 'stopped' : 'waiting') : ready ? 'running' : 'waiting'
    return { id, title, detail, state, lines: lines(id), count: total && total > 1 ? { done: Math.min(doneCount, total), total } : undefined }
  }

  const plannerDone = of('planner').length > 0 || finished
  const sitesDone = of('sites').length > 0 || finished
  const nWeb = plan?.n_web ?? 0
  const nApp = plan?.n_app ?? 0
  const nCommunity = plan?.n_community ?? 0
  const webDone = finished || (plannerDone && of('research_web').length >= nWeb)
  const appDone = finished || nApp === 0 || of('research_app').length >= nApp
  const communityDone = finished || nCommunity === 0 || of('research_community').length >= nCommunity
  const opinionsDone = of('opinions').length > 0 || finished || nCommunity === 0
  const analystDone = of('analyst').length > 0 || finished
  const writerDone = of('writer').length > 0 || finished

  const lanes: Stage[] = [
    make('research_web', 'Official websites', 'Reads each product’s own site', webDone, sitesDone, nWeb),
    ...(nApp > 0 ? [make('research_app', 'App Store', 'Rating, price and last update', appDone, sitesDone, nApp)] : []),
    ...(nCommunity > 0 ? [make('research_community', 'Community posts', 'Hacker News and Reddit', communityDone, sitesDone, nCommunity)] : []),
  ]
  const stages: Stage[] = [
    make('planner', 'Plan', 'Rewrites the question and picks what to compare', plannerDone, true),
    make('sites', 'Find official sites', 'One lookup per product', sitesDone, plannerDone),
    ...(nCommunity > 0
      ? [make('opinions', 'Score opinions', 'Sentiment and recurring themes', opinionsDone, communityDone)]
      : []),
    make('analyst', 'Analyse', 'Compares facts with what users say', analystDone, webDone && appDone && opinionsDone && sitesDone),
    make('writer', 'Write report', 'Drafts from the numbered evidence', writerDone, analystDone),
  ]
  return { stages, lanes, plan }
}
