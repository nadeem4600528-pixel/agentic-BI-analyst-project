import { reportById } from '../api'

export default function Reports({ payload, datasetId, download, run, result }) {
  const cards = [
    ['profiling', 'Profiling report'],
    ['cleaning', 'Cleaning report'],
    ['analysis', 'Analysis report'],
    ['full', 'Consolidated report'],
  ]

  const openReport = (kind) => {
    if (!datasetId) return
    run(() => reportById({
      dataset_id: datasetId,
      kind,
      date_column: payload?.date_column,
      value_column: payload?.value_column,
      category_column: payload?.category_column,
    }))
  }

  return (
    <section>
      <div className="section-heading">
        <div>
          <span className="tag">DELIVERABLES</span>
          <h2>Reports that move work forward.</h2>
          <p>Generate a consolidated view or download it for sharing.</p>
        </div>
        <button className="btn btn-primary" onClick={() => openReport('full')} disabled={!datasetId}>
          Generate report
        </button>
      </div>

      {!datasetId && payload?.data?.length > 0 && (
        <p className="text-muted">This dataset isn't linked to the server cache yet — re-upload it from the Data upload page to run reports.</p>
      )}

      <div className="report-grid">
        {cards.map(([id, title]) => (
          <div className="report-card" key={id}>
            <span className="report-symbol">▤</span>
            <h3>{title}</h3>
            <p>Quality, findings, insights, and recommendations.</p>
            <button className="btn btn-outline-primary btn-sm" onClick={() => openReport(id)} disabled={!datasetId}>
              Open report
            </button>
          </div>
        ))}
      </div>

      <div className="export-bar">
        <div>
          <strong>Export consolidated report</strong>
          <p>Choose a format for offline sharing.</p>
        </div>
        <div className="d-flex gap-2">
          <button className="btn btn-outline-dark" onClick={() => download('excel')}>Excel</button>
          <button className="btn btn-outline-dark" onClick={() => download('html')}>HTML</button>
          <button className="btn btn-dark" onClick={() => download('pdf')}>PDF ↓</button>
        </div>
      </div>

      {result && <pre className="result-preview large mt-4">{JSON.stringify(result, null, 2)}</pre>}
    </section>
  )
}