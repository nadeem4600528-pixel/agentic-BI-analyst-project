import { useMemo, useState } from 'react'
import { analyze, dashboard, demoDataset, getApiError, profileUpload, reportExport, transform, workflows } from './api'
import Sidebar from './components/Sidebar'
import Header from './components/Header'
import Dashboard from './components/Dashboard'
import Upload from './components/Upload'
import DataView from './components/DataView'
import Transform from './components/Transform'
import Reports from './components/Reports'
import Jobs from './components/Jobs'

const navigation = [
  ['dashboard', '▦', 'Dashboard'], ['upload', '↥', 'Data upload'],
  ['profiling', '◉', 'Profiling'], ['cleaning', '✦', 'Cleaning'],
  ['transform', '⇄', 'Transformation'], ['analysis', '◌', 'BI analysis'],
  ['reports', '▤', 'Reports'], ['jobs', '✓', 'Workflow status']
]
export default function App() {
  const [page, setPage] = useState('upload')
  const [records, setRecords] = useState([])
  const [datasetId, setDatasetId] = useState(null)
  const [columns, setColumns] = useState([])
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [fileName, setFileName] = useState('No dataset loaded')
  const [dateColumn, setDateColumn] = useState('')
  const [valueColumn, setValueColumn] = useState('')
  const [categoryColumn, setCategoryColumn] = useState('')
  const payload = useMemo(() => ({ data: records, date_column: dateColumn || undefined, value_column: valueColumn || undefined, category_column: categoryColumn || undefined }), [records, dateColumn, valueColumn, categoryColumn])
  const run = async (request, done) => { setLoading(true); setError(''); try { const response = await request(); setResult(response.data); done?.(response.data) } catch (e) { setError(await getApiError(e)) } finally { setLoading(false) } }
  const applyRows = (rows, name) => { if (Array.isArray(rows) && rows.length) { setRecords(rows); setColumns(Object.keys(rows[0])); setResult(null); if (name) setFileName(name) } else { setRecords([]); setColumns([]); setError('The dataset contains no data rows.') } }
  const upload = e => { const file = e.target.files?.[0]; if (!file) return; setFileName(file.name); setDatasetId(null); run(() => profileUpload(file), data => { applyRows(data.data || data.records || data.preview, file.name); setDatasetId(data.dataset_id || null) }) }
  const loadDemo = async () => { setLoading(true); setError(''); setDatasetId(null); try { const res = await demoDataset(); const d = res.data; applyRows(d.data, d.filename || 'Sample sales dataset'); setDatasetId(d.dataset_id || null); setDateColumn(d.suggested?.date_column || ''); setValueColumn(d.suggested?.value_column || ''); setCategoryColumn(d.suggested?.category_column || ''); setPage('dashboard') } catch (e) { setError(await getApiError(e, 'Could not load demo data')) } finally { setLoading(false) } }
  const download = async kind => { if (!records.length) { setError('Upload a dataset before exporting a report.'); return } setLoading(true); setError(''); try { const response = await reportExport(kind, payload); const url = URL.createObjectURL(response.data); const link = document.createElement('a'); link.href = url; link.download = `agentic-report.${kind === 'excel' ? 'xlsx' : kind}`; document.body.appendChild(link); link.click(); link.remove(); window.setTimeout(() => URL.revokeObjectURL(url), 1000) } catch (e) { setError(await getApiError(e, 'Export failed')) } finally { setLoading(false) } }
  const title = navigation.find(item => item[0] === page)?.[2]
  const content = page === 'dashboard' ? <Dashboard records={records} payload={payload} run={run} result={result} datasetId={datasetId} /> : page === 'upload' ? <Upload fileName={fileName} upload={upload} records={records} loadDemo={loadDemo} /> : page === 'transform' ? <Transform records={records} columns={columns} run={run} setRecords={setRecords} setColumns={setColumns} /> : page === 'reports' ? <Reports payload={payload} datasetId={datasetId} download={download} run={run} result={result} /> : page === 'jobs' ? <Jobs run={run} result={result} /> : <DataView title={title} records={records} result={result} action={() => run(page === 'analysis' ? () => analyze(payload) : () => dashboard(payload))} actionLabel={page === 'analysis' ? 'Run analysis' : 'Scan dataset'} />
  return <div className="app-shell"><Sidebar navigation={navigation} page={page} setPage={setPage} /><main className="main-content"><Header title={title} fileName={fileName} setPage={setPage} />{error && <div className="alert alert-danger border-0 shadow-sm">{error}<button className="btn-close float-end" onClick={() => setError('')} /></div>}{loading && <div className="loading-line"><span /></div>}{content}</main></div>
}
