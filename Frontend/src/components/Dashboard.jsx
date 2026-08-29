import { useEffect } from 'react'
import Plot from 'react-plotly.js'
import { dashboard } from '../api'
import { PanelTitle, Stat, Table } from './Common'

const plotLayout = { autosize: true, margin: { l: 48, r: 18, t: 18, b: 48 }, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { family: 'Inter', color: '#718096' } }

function chartFigure(chart) {
  if (chart.type === 'heatmap') return [{ z: chart.data.z, x: chart.data.x, y: chart.data.y, type: 'heatmap', colorscale: 'Purples' }]
  const points = chart.data || []
  return [{ x: points.map(point => point.x), y: points.map(point => point.y), type: chart.type === 'histogram' ? 'bar' : chart.type, mode: chart.type === 'line' ? 'lines+markers' : undefined, marker: { color: '#7167f8' }, line: { color: '#7167f8', width: 3 } }]
}

function AgentCard({ agent }) {
  if (!agent) return null
  const quality = agent.quality || {}
  const summary = agent.business_summary || {}
  const findings = [...(agent.profiling?.critical_issues || []), ...(agent.profiling?.findings || [])]
  const recommendations = agent.recommendations || []
  return <>
    <div className="panel mt-4"><PanelTitle title="BI Agent assessment" sub={`${agent.role} · ${agent.status}`} /><div className="grid-two"><div><span className="eyebrow">EXECUTIVE HEADLINE</span><h3>{summary.headline || 'Analysis is ready for review.'}</h3><p className="text-muted">Risk level: <strong>{summary.risk_level || 'medium'}</strong></p></div><div className="stat-grid compact"><Stat label="Quality score" value={`${Number(quality.score || 0).toFixed(1)}%`} note={quality.level || 'Assessed'} /><Stat label="Quality issues" value={quality.issues || 0} note="Profiling findings" /></div></div></div>
    <div className="grid-two mt-4"><div className="panel"><PanelTitle title="Findings and risks" sub="Evidence from profiling and analysis" />{findings.length ? <ul className="insight-list">{findings.slice(0, 8).map((item, index) => <li key={index}>{item}</li>)}</ul> : <p className="text-muted">No material issues detected.</p>}</div><div className="panel"><PanelTitle title="Recommended actions" sub="Prioritized next steps" />{recommendations.length ? <ol className="insight-list">{recommendations.slice(0, 8).map((item, index) => <li key={index}>{item}</li>)}</ol> : <p className="text-muted">No recommendations are pending.</p>}</div></div>
  </>
}

export default function Dashboard({ records, payload, run, result }) {
  useEffect(() => { if (records.length && !result?.bi_agent) run(() => dashboard(payload)) }, [records.length])
  if (!records.length) return <section><div className="empty-state"><span className="tag">NO DATASET</span><h2>Upload a dataset to begin.</h2><p>The BI Agent will profile, assess, analyze, visualize, and prepare a business report.</p></div></section>
  const fields = Object.keys(records[0] || {})
  const charts = result?.charts || []
  const kpis = result?.kpis || []
  return <><section className="hero"><div><span className="tag">PROFESSIONAL BI AGENT</span><h2>From raw data to<br /><em>business decisions.</em></h2><p>Automated profiling, quality controls, analytical insight, and Power BI-style interactive reporting.</p></div><button className="btn btn-light" onClick={() => run(() => dashboard(payload))}>Refresh analysis ↻</button></section><div className="stat-grid">{(kpis.length ? kpis : [{ title: 'Records', value: records.length }, { title: 'Columns', value: fields.length }]).slice(0, 5).map(kpi => <Stat key={kpi.id || kpi.title} label={kpi.title} value={typeof kpi.value === 'number' ? kpi.value.toLocaleString() : kpi.value} note={kpi.delta == null ? 'Agent-generated KPI' : `${Number(kpi.delta).toFixed(1)}% change`} />)}</div>{!result && <div className="panel mt-4"><p className="text-muted mb-0">Preparing the complete BI assessment…</p></div>}<AgentCard agent={result?.bi_agent} />{charts.length > 0 && <section className="mt-4"><div className="section-heading"><div><span className="tag">INTERACTIVE VISUALS</span><h2>Decision dashboard.</h2><p>Charts selected from detected dates, measures, segments, and relationships.</p></div></div><div className="dashboard-chart-grid">{charts.map(chart => <div className="panel chart-panel" key={chart.id}><PanelTitle title={chart.title} sub={`${chart.x_axis || ''} · ${chart.y_axis || ''}`} /><Plot data={chartFigure(chart)} layout={plotLayout} useResizeHandler style={{ width: '100%', height: 320 }} config={{ displayModeBar: false, responsive: true }} /></div>)}</div></section>}<div className="panel mt-4"><PanelTitle title="Data preview" sub={`${result?.table?.total_rows || records.length} total records · first 100 shown`} /><Table records={result?.table?.rows || records} /></div></>
}
