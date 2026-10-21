import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { api } from '../lib/api'

function ThemeToggle() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains('dark'))
  const toggle = () => {
    const next = !dark
    setDark(next)
    document.documentElement.classList.toggle('dark', next)
    try {
      localStorage.setItem('theme', next ? 'dark' : 'light')
    } catch {
      /* storage can be unavailable */
    }
  }
  return (
    <button
      type="button"
      onClick={toggle}
      aria-pressed={dark}
      className="rounded-md border border-rule px-2.5 py-1 text-sm text-muted hover:text-ink"
    >
      {dark ? 'Light' : 'Dark'} theme
    </button>
  )
}

const link = ({ isActive }: { isActive: boolean }) =>
  `rounded-md px-2.5 py-1 text-sm ${isActive ? 'bg-accent-soft font-medium text-ink' : 'text-muted hover:text-ink'}`

export default function Layout() {
  const { data: config } = useQuery({ queryKey: ['config'], queryFn: api.config })
  return (
    <div className="min-h-screen">
      <header className="border-b border-rule bg-surface">
        <div className="mx-auto flex max-w-6xl items-center gap-4 px-4 py-3 sm:px-6">
          <NavLink to="/" className="mr-2 text-base font-semibold tracking-tight text-ink">
            Market Research Crew
          </NavLink>
          <nav className="flex gap-1" aria-label="Main">
            <NavLink to="/" end className={link}>
              New research
            </NavLink>
            <NavLink to="/history" className={link}>
              History
            </NavLink>
          </nav>
          <div className="ml-auto">
            <ThemeToggle />
          </div>
        </div>
        {config?.demo && (
          <p className="border-t border-rule bg-accent-soft px-4 py-2 text-center text-sm text-ink">
            Demo mode: every run replays one recorded sample, so no API keys are used.
          </p>
        )}
      </header>
      <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
        <Outlet />
      </main>
    </div>
  )
}
