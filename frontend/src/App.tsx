import { Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import History from './pages/History'
import NewRun from './pages/NewRun'
import RunPage from './pages/RunPage'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<NewRun />} />
        <Route path="runs/:id" element={<RunPage />} />
        <Route path="history" element={<History />} />
        <Route path="*" element={<p className="py-16 text-muted">That page does not exist.</p>} />
      </Route>
    </Routes>
  )
}
