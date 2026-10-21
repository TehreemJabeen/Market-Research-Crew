import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

export default function ReportView({ markdown, filename }: { markdown: string; filename: string }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(markdown)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      /* clipboard can be blocked */
    }
  }
  const download = () => {
    const url = URL.createObjectURL(new Blob([markdown], { type: 'text/markdown' }))
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }
  return (
    <div>
      <div className="mb-6 flex gap-2">
        <button type="button" onClick={copy} className="rounded-md border border-rule px-3 py-1.5 text-sm hover:border-accent">
          {copied ? 'Copied' : 'Copy Markdown'}
        </button>
        <button type="button" onClick={download} className="rounded-md border border-rule px-3 py-1.5 text-sm hover:border-accent">
          Download .md
        </button>
      </div>
      <article className="report">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            a: ({ href, children }) => (
              <a href={href} target="_blank" rel="noopener noreferrer">
                {children}
              </a>
            ),
            table: ({ children }) => (
              <div className="table-wrap">
                <table>{children}</table>
              </div>
            ),
          }}
        >
          {markdown}
        </ReactMarkdown>
      </article>
    </div>
  )
}
