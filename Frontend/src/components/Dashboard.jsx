import { useEffect, useMemo, useState } from 'react'
import { dashboard } from '../api'
import PlotlyChart from './charts'
import { PanelTitle, Table } from './Common'

// ---------------------------------------------------------------------------
// Small inline sparkline (no extra chart overhead) for KPI tiles
// ---------------------------------------------------------------------------
function Sparkline({ points, up }) {
  if (!points || points.length < 2) return null
  const w = 96, h = 30
  const ys = points.map(p => p.y)
  const min = Math.min(...ys), max = Math.max(...ys)
  const span = max - min || 1
  const step = w / (points.length - 1)
  const path = points.map((p, i) => {
    const x = i * step
    const y = h - ((p.y - min) / span) * (h - 4) - 2
    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
  const color = up ? '#22c55e' : '#ef4444'
  return (
    <svg width={w} height={h} className="kpi-spark" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      <path d={path} fill="none" stroke={color} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

const ICONS = {
  records: '▤', sales: '$', profit: '▲', average: 'x̄', customers: '◍', rate: '%', alert: '!', trend: '↗',
}

function KpiCard({ kpi }) {
  const dir = kpi.trend_direction
  const up = dir === 'up'
  const deltaClass = dir === 'up' ? 'delta-up' : dir === 'down' ? 'delta-down' : 'delta-flat'
  return (
    <div className="kpi-card">
      <div className="kpi-top">
        <span className="kpi-label">{kpi.title}</span>
        <span className="kpi-icon">{ICONS[kpi.icon] || '●'}</span>
      </div>
      <div className="kpi-value">{kpi.display_value}</div>
      <div className="kpi-bottom">
        {kpi.delta != null ? (
          <span className={`kpi-delta ${deltaClass}`}>
            {up ? '▲' : dir === 'down' ? '▼' : '■'} {Math.abs(kpi.delta)}%
            <em>{kpi.delta_label || ''}</em>
          </span>
        ) : (
          <span className="kpi-delta delta-flat"><em>{kpi.subtitle || kpi.delta_label || 'Live metric'}</em></span>
        )}
        <Sparkline points={kpi.sparkline} up={up} />
      </div>
    </div>
  )
}

function InsightCard({ insight }) {
  const tone = { positive: 'insight-good', warning: 'insight-warn', neutral: 'insight-neutral' }[insight.level] || 'insight-neutral'
  const badge = { positive: '▲', warning: '!', neutral: 'i' }[insight.level] || 'i'
  return (
    <div className={`insight-card ${tone}`}>
      <span className="insight-badge">{badge}</span>
      <div>
        <h4>{insight.title}</h4>
        <p>{insight.detail}</p>
      </div>
    </div>
  )
}

function spanClass(chart, spans) {
  const span = spans?.[chart.id] || chart.layout_hint?.span || 'half'
  if (span === 'wide') return 'span-wide'
  if (span === 'third') return 'span-third'
  return 'span-half'
}

export default function Dashboard({ records, payload, run, result }) {
  const [roles, setRoles] = useState({ date: '', value: '', category: '' })
  const [slicers, setSlicers] = useState({})

  const hasData = records.length > 0

  // Build the request payload (filters + role columns). Slicers are applied
  // client-side by sending the filtered rows so no extra endpoint is needed.
  const dashboardPayload = useMemo(() => {
    let rows = records
    const activeSlicers = Object.entries(slicers).filter(([, vals]) => Array.isArray(vals) && vals.length)
    if (activeSlicers.length) {
      rows = records.filter(r => activeSlicers.every(([name, vals]) => {
        const col = name.replace('slicer_', '')
        return vals.includes(String(r[col]))
      }))
    }
    return {
      data: rows,
      date_column: roles.date || payload?.date_column || undefined,
      value_column: roles.value || payload?.value_column || undefined,
      category_column: roles.category || payload?.category_column || undefined,
    }
  }, [records, slicers, roles, payload])

  const matchingRows = dashboardPayload.data.length

  // Auto-run when data first loads or filter/role selection changes.
  useEffect(() => {
    if (hasData && matchingRows > 0) run(() => dashboard(dashboardPayload))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasData, matchingRows, dashboardPayload.date_column, dashboardPayload.value_column, dashboardPayload.category_column,
      JSON.stringify(slicers)])

  if (!hasData) {
    return (
      <section>
        <div className="empty-state">
          <span className="tag">NO DATASET</span>
          <h2>Upload a dataset to begin.</h2>
          <p>The BI Agent profiles, cleans, analyzes and builds a Power BI-style interactive dashboard automatically.</p>
        </div>
      </section>
    )
  }

  const charts = result?.charts || []
  const kpis = result?.kpis || []
  const insights = result?.insights || []
  const filters = result?.filters || []
  const sections = result?.layout?.sections || []
  const spans = result?.layout?.spans || {}
  const summary = result?.summary || {}

  const roleFilters = filters.filter(f => f.type === 'role')
  const slicerFilters = filters.filter(f => f.type === 'slicer')
  const chartsById = Object.fromEntries(charts.map(c => [c.id, c]))

  const setRole = (name, value) => setRoles(prev => ({ ...prev, [name]: value || '' }))
  const toggleSlicer = (name, option) => {
    setSlicers(prev => {
      const current = prev[name] || []
      const next = current.includes(option) ? current.filter(v => v !== option) : [...current, option]
      return { ...prev, [name]: next }
    })
  }

  return (
    <>
      {/* Hero */}
      <section className="hero">
        <div>
          <span className="tag">POWER BI-STYLE DASHBOARD</span>
          <h2>From raw data to<br /><em>business decisions.</em></h2>
          <p>Auto-generated KPIs, trends, breakdowns, distributions and smart insights — refreshed as you filter.</p>
        </div>
        <button className="btn btn-light" onClick={() => run(() => dashboard(dashboardPayload))}>
          Refresh ↻
        </button>
      </section>

      {result?.truncated_note && (
        <div className="alert alert-warning mb-3">{result.truncated_note}</div>
      )}

      {/* Filters / slicers */}
      {(roleFilters.length > 0 || slicerFilters.length > 0) && (
        <div className="filter-bar panel">
          <div className="filter-group">
            {roleFilters.map(f => (
              <label key={f.name} className="filter-select">
                <span>{f.label}</span>
                <select
                  value={f.name === 'date_column' ? roles.date : f.name === 'value_column' ? roles.value : roles.category}
                  onChange={e => setRole(
                    f.name === 'date_column' ? 'date' : f.name === 'value_column' ? 'value' : 'category',
                    e.target.value
                  )}
                >
                  <option value="">Auto</option>
                  {(f.options || []).filter(Boolean).map(o => <option key={o} value={o}>{o}</option>)}
                </select>
              </label>
            ))}
          </div>
          <div className="slicer-group">
            {slicerFilters.map(f => (
              <div key={f.name} className="slicer">
                <span className="slicer-title">{f.label}</span>
                <div className="slicer-chips">
                  {(f.options || []).slice(0, 12).map(opt => {
                    const active = (slicers[f.name] || []).includes(opt)
                    return (
                      <button
                        key={opt}
                        className={`chip ${active ? 'chip-active' : ''}`}
                        onClick={() => toggleSlicer(f.name, opt)}
                      >
                        {opt}
                      </button>
                    )
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {matchingRows === 0 && (
        <div className="panel mt-4" style={{ textAlign: 'center', padding: '40px' }}>
          <h3>No records match the selected filters.</h3>
          <p className="text-muted">Clear one or more slicer chips to bring data back.</p>
        </div>
      )}

      {/* KPI row */}
      {matchingRows > 0 && (
      <div className="kpi-grid">
        {kpis.length ? kpis.map(kpi => <KpiCard key={kpi.id} kpi={kpi} />) : (
          <>
            <div className="kpi-card"><div className="kpi-value">{records.length.toLocaleString()}</div><span className="kpi-label">Records</span></div>
          </>
        )}
      </div>
      )}

      {/* Smart insights */}
      {insights.length > 0 && (
        <section className="mt-4">
          <div className="section-heading">
            <div><span className="tag">SMART INSIGHTS</span><h2>What the data is telling you.</h2><p>Auto-generated findings ranked by business impact.</p></div>
          </div>
          <div className="insight-grid">
            {insights.map((ins, i) => <InsightCard key={i} insight={ins} />)}
          </div>
        </section>
      )}

      {/* Sectioned chart grid */}
      {!charts.length && (
        <div className="panel mt-4"><p className="text-muted mb-0">Preparing your dashboard…</p></div>
      )}
      {sections.map(section => {
        const sectionCharts = section.chart_ids.map(id => chartsById[id]).filter(Boolean)
        if (!sectionCharts.length) return null
        return (
          <section className="mt-4" key={section.id}>
            <div className="section-heading">
              <div><span className="tag">{section.title.toUpperCase()}</span></div>
            </div>
            <div className="chart-grid">
              {sectionCharts.map(chart => (
                <div className={`panel chart-panel ${spanClass(chart, spans)}`} key={chart.id}>
                  <PanelTitle title={chart.title} sub={chart.subtitle || `${chart.x_axis || ''} ${chart.y_axis ? '· ' + chart.y_axis : ''}`} />
                  <PlotlyChart chart={chart} height={chart.type === 'gauge' ? 240 : chart.layout_hint?.span === 'wide' ? 340 : 300} />
                </div>
              ))}
            </div>
          </section>
        )
      })}

      {/* Data preview */}
      <div className="panel mt-4">
        <PanelTitle title="Data preview" sub={`${result?.table?.total_rows || records.length} total records · first 100 shown`} />
        <Table records={result?.table?.rows || records} />
      </div>

      {summary && (
        <p className="text-muted small mt-3 mb-0">
          Date: <b>{summary.date_column || '—'}</b> · Measure: <b>{summary.value_column || '—'}</b> · Dimension: <b>{summary.category_column || '—'}</b>
        </p>
      )}
    </>
  )
}
