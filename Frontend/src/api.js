import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 300000
})

export const profileUpload = (file, onUploadProgress) => {
  const form = new FormData()
  form.append('file', file)
  return api.post('/upload/profile', form, { onUploadProgress })
}
export const dashboard = (payload) => api.post('/dashboard/', payload)
export const transform = (payload) => api.post('/transformation/', payload)
export const analyze = (payload) => api.post('/analysis/', payload)
export const report = (payload) => api.post('/report/', payload)
export const reportExport = (kind, payload) => api.post(`/report/export/${kind}`, payload, { responseType: 'blob' })

export async function getApiError(error, fallback = 'Request failed') {
  const response = error?.response
  if (!response) return error?.code === 'ERR_NETWORK' ? 'Cannot connect to FastAPI. Start the backend at http://localhost:8000.' : error?.message || fallback
  if (response.data instanceof Blob) {
    try {
      const text = await response.data.text()
      const parsed = JSON.parse(text)
      return parsed.detail || text || fallback
    } catch {
      return response.data.size ? `${fallback} (HTTP ${response.status})` : fallback
    }
  }
  return response.data?.detail || response.data?.message || fallback
}

export const workflows = () => api.get('/workflow/jobs')
export default api
