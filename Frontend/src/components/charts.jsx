import Plot from 'react-plotly.js'

// Brand-aligned categorical palette
export const PALETTE = ['#6358df', '#22c1c3', '#f59e0b', '#ef4444', '#10b981', '#8b5cf6', '#ec4899', '#0ea5e9', '#f97316', '#64748b']

const baseLayout = (extra = {}) => ({
  autosize: true,
  margin: { l: 56, r: 20, t: 28, b: 48 },
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  font: { family: 'Inter, sans-serif', color: '#64748b', size: 11 },
  legend: { orientation: 'h', y: -0.18, font: { size: 10 } },
  ...extra,
})

// Build the Plotly trace(s) + layout for a given backend chart definition.
export function buildFigure(chart) {
  const d = chart.data || {}
  switch (chart.type) {
    case 'line':
    case 'area': {
      const series = d.series || []
      const traces = series.map((s, i) => ({
        x: d.x, y: s.y, name: s.name,
        mode: chart.type === 'line' ? 'lines+markers' : 'lines',
        type: 'scatter',
        fill: chart.type === 'area' && i === 0 ? 'tozeroy' : series.length > 1 && chart.type === 'area' ? 'tonexty' : 'none',
        fillcolor: i === 0 ? 'rgba(99,88,223,0.18)' : undefined,
        line: { color: PALETTE[i % PALETTE.length], width: chart.type === 'area' ? 2.5 : 3, shape: 'spline' },
        marker: { color: PALETTE[i % PALETTE.length], size: 6 },
        hovertemplate: `%{x}<br>${s.name}: %{y:,.0f}<extra></extra>`,
      }))
      return { data: traces, layout: baseLayout({ hovermode: 'x unified' }) }
    }

    case 'growth': {
      const trace = {
        x: d.x, y: d.y, type: 'bar',
        marker: { color: d.colors || d.y.map(v => (v >= 0 ? '#22c55e' : '#ef4444')) },
        hovertemplate: '%{x}<br>%{y:+.1f}%<extra></extra>',
      }
      return {
        data: [trace],
        layout: baseLayout({
          barmode: 'relative',
          shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0, y1: 0, line: { color: '#94a3b8', width: 1 } }],
        }),
      }
    }

    case 'bar':
    case 'histogram': {
      const trace = {
        x: d.x, y: d.y, type: 'bar',
        marker: { color: chart.type === 'histogram' ? '#8b5cf6' : '#6358df', borderRadius: chart.type === 'histogram' ? 2 : 6 },
        hovertemplate: '%{x}<br>%{y:,.0f}<extra></extra>',
      }
      return { data: [trace], layout: baseLayout({ barmode: 'stack' }) }
    }

    case 'stacked_bar':
    case 'grouped_bar': {
      const series = d.series || []
      const traces = series.map((s, i) => ({
        x: d.x, y: s.y, name: s.name, type: 'bar',
        marker: { color: PALETTE[i % PALETTE.length], borderRadius: 4 },
        hovertemplate: `%{x}<br>${s.name}: %{y:,.0f}<extra></extra>`,
      }))
      return { data: traces, layout: baseLayout({ barmode: chart.type === 'stacked_bar' ? 'stack' : 'group', hovermode: 'x unified' }) }
    }

    case 'donut': {
      const trace = {
        labels: d.labels, values: d.values, type: 'pie',
        hole: 0.62, sort: false,
        marker: { colors: PALETTE, line: { color: '#fff', width: 2 } },
        textinfo: 'percent', textfont: { size: 10, color: '#fff' },
        hovertemplate: '%{label}<br>%{value:,.0f} (%{percent})<extra></extra>',
      }
      return { data: [trace], layout: baseLayout({ margin: { l: 10, r: 10, t: 10, b: 10 }, showlegend: true }) }
    }

    case 'treemap': {
      const trace = {
        type: 'treemap',
        ids: d.ids, labels: d.labels, parents: d.parents, values: d.values,
        branchvalues: 'total',
        marker: { colors: d.values, colorscale: 'Purples', line: { width: 2 } },
        textinfo: 'label+value',
        hovertemplate: '%{label}<br>%{value:,.0f}<extra></extra>',
      }
      return { data: [trace], layout: baseLayout({ margin: { l: 0, r: 0, t: 0, b: 0 } }) }
    }

    case 'pareto': {
      const bars = { x: d.x, y: d.y, type: 'bar', name: chart.y_axis || 'value', marker: { color: '#6358df', borderRadius: 5 }, hovertemplate: '%{x}<br>%{y:,.0f}<extra></extra>' }
      const line = {
        x: d.x, y: d.cumulative, type: 'scatter', mode: 'lines+markers', name: 'Cumulative %',
        yaxis: 'y2', line: { color: '#f59e0b', width: 3 }, marker: { size: 6, color: '#f59e0b' },
        hovertemplate: '%{x}<br>%{y:.0f}%<extra></extra>',
      }
      const ref = { type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 80, y1: 80, yref: 'y2', line: { color: '#ef4444', width: 1, dash: 'dash' } }
      return {
        data: [bars, line],
        layout: baseLayout({
          barmode: 'stack',
          yaxis: { title: { text: chart.y_axis, font: { size: 10 } } },
          yaxis2: { title: { text: 'Cumulative %', font: { size: 10 } }, overlaying: 'y', side: 'right', range: [0, 100], ticksuffix: '%' },
          shapes: [ref],
          legend: { orientation: 'h', y: -0.22 },
        }),
      }
    }

    case 'box': {
      const traces = (d.traces || []).map((t, i) => ({
        type: 'box', name: t.label,
        q1: [t.q1], median: [t.median], q3: [t.q3],
        lowerfence: [t.min], upperfence: [t.max], mean: [t.mean],
        marker: { color: PALETTE[i % PALETTE.length] },
        fillcolor: `rgba(99,88,223,0.15)`,
        orientation: 'v',
        hovertemplate: `${t.label}<br>Median: %{median:,.0f}<extra></extra>`,
      }))
      return { data: traces, layout: baseLayout({ yaxis: { title: { text: chart.y_axis, font: { size: 10 } } } }) }
    }

    case 'scatter': {
      let traces
      if (d.groups) {
        traces = d.groups.map((g, i) => ({
          x: g.x, y: g.y, name: g.name, mode: 'markers', type: 'scatter',
          marker: { color: PALETTE[i % PALETTE.length], size: 7, opacity: 0.65, line: { width: 0.5, color: '#fff' } },
          hovertemplate: `${g.name}<br>${chart.x_axis}: %{x:,.0f}<br>${chart.y_axis}: %{y:,.0f}<extra></extra>`,
        }))
      } else {
        traces = [{
          x: d.x, y: d.y, mode: 'markers', type: 'scatter', name: 'records',
          marker: { color: 'rgba(99,88,223,0.55)', size: 7, line: { width: 0.5, color: '#fff' } },
          hovertemplate: `${chart.x_axis}: %{x:,.0f}<br>${chart.y_axis}: %{y:,.0f}<extra></extra>`,
        }]
      }
      return { data: traces, layout: baseLayout({ xaxis: { title: { text: chart.x_axis, font: { size: 10 } } }, yaxis: { title: { text: chart.y_axis, font: { size: 10 } } }, legend: { orientation: 'h', y: -0.2 } }) }
    }

    case 'heatmap': {
      const trace = {
        z: d.z, x: d.x, y: d.y, type: 'heatmap',
        colorscale: [['0', '#eef2ff'], ['0.5', '#8b5cf6'], ['1', '#4c1d95']],
        zmin: -1, zmax: 1,
        hovertemplate: '%{y} vs %{x}<br>r = %{z:.2f}<extra></extra>',
        colorbar: { thickness: 10, len: 0.9 },
      }
      return { data: [trace], layout: baseLayout({ margin: { l: 90, r: 10, t: 20, b: 70 }, xaxis: { tickangle: -35 } }) }
    }

    case 'gauge': {
      const trace = {
        type: 'indicator', mode: 'gauge+number',
        value: d.value, number: { suffix: '%', font: { size: 30, color: '#1d2433' } },
        gauge: {
          axis: { range: [0, Math.max(120, d.value)], ticksuffix: '%' },
          bar: { color: d.value >= 100 ? '#22c55e' : d.value >= 75 ? '#6358df' : '#f59e0b' },
          steps: [
            { range: [0, 50], color: '#fee2e2' },
            { range: [50, 85], color: '#fef3c7' },
            { range: [85, 100], color: '#dcfce7' },
          ],
          threshold: { line: { color: '#ef4444', width: 3 }, thickness: 0.9, value: 100 },
        },
      }
      return { data: [trace], layout: baseLayout({ margin: { l: 30, r: 30, t: 30, b: 10 } }) }
    }

    default:
      return null
  }
}

export default function PlotlyChart({ chart, height = 320 }) {
  const figure = buildFigure(chart)
  if (!figure) return null
  return (
    <Plot
      data={figure.data}
      layout={figure.layout}
      useResizeHandler
      style={{ width: '100%', height }}
      config={{ displayModeBar: false, responsive: true }}
    />
  )
}
