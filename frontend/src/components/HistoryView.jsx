import React, { useState, useEffect } from 'react'
import { History, Loader2, AlertCircle, Trash2, RefreshCw, Filter } from 'lucide-react'
import { getHistory, clearHistory } from '../services/api'

function HistoryView() {
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [filter, setFilter] = useState('')
  const [limit, setLimit] = useState(50)

  const engines = ['', 'math', 'vision', 'cs', 'poker', 'chemistry', 'biology', 'statistics']

  const fetchHistory = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await getHistory(limit, filter || null)
      setHistory(response.data.history || [])
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to load history')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchHistory()
  }, [filter, limit])

  const handleClear = async () => {
    if (!confirm('Are you sure you want to clear all history?')) return
    try {
      await clearHistory()
      setHistory([])
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to clear history')
    }
  }

  const formatDate = (timestamp) => {
    return new Date(timestamp).toLocaleString()
  }

  const getEngineColor = (engine) => {
    const colors = {
      math: 'bg-blue-100 text-blue-700',
      vision: 'bg-purple-100 text-purple-700',
      cs: 'bg-green-100 text-green-700',
      poker: 'bg-red-100 text-red-700',
      chemistry: 'bg-orange-100 text-orange-700',
      biology: 'bg-pink-100 text-pink-700',
      statistics: 'bg-cyan-100 text-cyan-700'
    }
    return colors[engine] || 'bg-gray-100 text-gray-700'
  }

  const truncateQuery = (query, maxLen = 100) => {
    if (typeof query === 'object') {
      query = JSON.stringify(query)
    }
    if (query.length > maxLen) {
      return query.substring(0, maxLen) + '...'
    }
    return query
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="p-3 bg-gray-100 rounded-xl">
          <History className="w-6 h-6 text-gray-600" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Query History</h2>
          <p className="text-gray-600">View past API queries and results</p>
        </div>
      </div>

      {/* Controls */}
      <div className="card">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-gray-500" />
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="input-field w-40"
            >
              <option value="">All Engines</option>
              {engines.slice(1).map((eng) => (
                <option key={eng} value={eng}>
                  {eng.charAt(0).toUpperCase() + eng.slice(1)}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600">Show:</span>
            <select
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
              className="input-field w-24"
            >
              <option value={25}>25</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </select>
          </div>

          <div className="flex-1" />

          <button
            onClick={fetchHistory}
            disabled={loading}
            className="btn-secondary flex items-center gap-2"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>

          <button
            onClick={handleClear}
            className="px-4 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors flex items-center gap-2"
          >
            <Trash2 className="w-4 h-4" />
            Clear
          </button>
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="result-box result-error flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-medium text-red-800">Error</p>
            <p className="text-red-700">{error}</p>
          </div>
        </div>
      )}

      {/* Loading state */}
      {loading && (
        <div className="card flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin text-gray-400" />
        </div>
      )}

      {/* History list */}
      {!loading && history.length === 0 && (
        <div className="card text-center py-12">
          <History className="w-12 h-12 text-gray-300 mx-auto mb-3" />
          <p className="text-gray-500">No history yet</p>
          <p className="text-sm text-gray-400">Make some API calls to see them here</p>
        </div>
      )}

      {!loading && history.length > 0 && (
        <div className="space-y-3">
          {history.map((item) => (
            <div key={item.id} className="card hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${getEngineColor(item.engine)}`}>
                      {item.engine}
                    </span>
                    <span className="text-xs text-gray-500">
                      {formatDate(item.timestamp)}
                    </span>
                  </div>

                  <p className="text-sm text-gray-900 font-mono truncate">
                    {truncateQuery(item.query)}
                  </p>

                  {item.result && (
                    <details className="mt-2">
                      <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">
                        View result
                      </summary>
                      <pre className="mt-2 p-2 bg-gray-50 rounded text-xs overflow-x-auto max-h-40">
                        {typeof item.result === 'object'
                          ? JSON.stringify(item.result, null, 2)
                          : item.result}
                      </pre>
                    </details>
                  )}
                </div>

                <div className="text-right text-xs text-gray-500 flex-shrink-0">
                  <p>{item.duration_ms}ms</p>
                  {item.cached && (
                    <span className="px-1.5 py-0.5 bg-green-100 text-green-700 rounded text-xs">
                      cached
                    </span>
                  )}
                  {item.cost > 0 && (
                    <p className="text-yellow-600">${item.cost.toFixed(4)}</p>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Stats summary */}
      {!loading && history.length > 0 && (
        <div className="card bg-gray-50">
          <p className="text-sm text-gray-600">
            Showing {history.length} queries
            {filter && ` for ${filter}`}
          </p>
        </div>
      )}
    </div>
  )
}

export default HistoryView
