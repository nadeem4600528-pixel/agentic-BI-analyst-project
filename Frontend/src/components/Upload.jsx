import { PanelTitle, Table } from './Common'

export default function Upload({ fileName, upload, records, loadDemo }) {
  return (
    <section>
      <div className="section-heading">
        <div>
          <span className="tag">DATA INGESTION</span>
          <h2>Bring your data to life.</h2>
          <p>Upload a file and the analyst builds a Power BI-style dashboard automatically — or try the live demo.</p>
        </div>
      </div>

      <div className="upload-grid">
        <div className="upload-card">
          <div className="upload-icon">↥</div>
          <h3>Drop your dataset here</h3>
          <p>CSV, Excel, JSON, or Parquet · processed locally</p>
          <label className="btn btn-primary">
            Choose file
            <input type="file" hidden accept=".csv,.xlsx,.xls,.json,.parquet" onChange={upload} />
          </label>
          <div className="upload-name">{fileName}</div>
        </div>

        <div className="upload-card demo-card">
          <div className="upload-icon demo-icon">◈</div>
          <h3>Not sure? Try the demo</h3>
          <p>A 4,000-row, 3-year sales dataset with regions, categories, segments &amp; profit.</p>
          <button className="btn btn-outline-primary" onClick={loadDemo}>Load sample dashboard →</button>
          <div className="upload-name">Instant KPIs, trends &amp; insights</div>
        </div>
      </div>

      {records.length > 0 && (
        <div className="panel mt-4">
          <PanelTitle title="Current preview" sub={`${records.length} records loaded`} />
          <Table records={records} />
        </div>
      )}
    </section>
  )
}
