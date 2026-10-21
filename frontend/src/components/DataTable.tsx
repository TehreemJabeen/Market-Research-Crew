import type { Table } from '../lib/types'

export default function DataTable({ table, caption }: { table: Table; caption?: string }) {
  return (
    <div className="overflow-x-auto rounded-md border border-rule bg-surface">
      <table className="w-full border-collapse text-left text-sm leading-relaxed">
        {caption && <caption className="px-3 py-2 text-left text-muted">{caption}</caption>}
        <thead>
          <tr className="bg-accent-soft">
            {table.columns.map((c) => (
              <th key={c} scope="col" className="border-b border-rule px-3 py-2 font-semibold">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, i) => (
            <tr key={i} className="align-top">
              {row.map((cell, j) => (
                <td key={j} className={`min-w-32 border-b border-rule px-3 py-2 ${j === 0 ? 'font-medium' : ''}`}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
