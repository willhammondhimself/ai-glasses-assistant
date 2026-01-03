import React, { useState, useEffect } from 'react'
import { DollarSign, Loader2, AlertCircle, RefreshCw, Trash2, Database } from 'lucide-react'
import { Doughnut, Bar } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  ArcElement,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js'
import { getStats, getCacheStats, clearCache } from '../services/api'

ChartJS.register(
  ArcElement,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
)

function CostTracker() {
  const [stats, setStats] = useState(null)
  const [cacheStats, setCacheStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchStats = async () => {
    setLoading(true)
    setError(null)
    try {
      const [statsRes, cacheRes] = await Promise.all([
        getStats(),
        getCacheStats()
      ])
      setStats(statsRes.data)
      setCacheStats(cacheRes.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to load stats')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchStats()
  }, [])

  const handleClearCache = async () => {
    if (!confirm('Are you sure you want to clear the cache?')) return
    try {
      await clearCache()
      fetchStats()
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to clear cache')
    }
  }

  const engineColors = {
    math: '#3B82F6',
    vision: '#8B5CF6',
    cs: '#10B981',
    poker: '#EF4444',
    chemistry: '#F97316',
    biology: '#EC4899',
    statistics: '#06B6D4'
  }

  const getEngineChartData = () => {
    if (!stats?.by_engine) return null

    const labels = Object.keys(stats.by_engine)
    const data = labels.map((eng) => stats.by_engine[eng].queries)
    const colors = labels.map((eng) => engineColors[eng] || '#6B7280')

    return {
      labels: labels.map((l) => l.charAt(0).toUpperCase() + l.slice(1)),
      datasets: [
        {
          data,
          backgroundColor: colors,
          borderWidth: 0
        }
      ]
    }
  }

  const getCostChartData = () => {
    if (!stats?.by_engine) return null

    const labels = Object.keys(stats.by_engine)
    const data = labels.map((eng) => stats.by_engine[eng].cost || 0)
    const colors = labels.map((eng) => engineColors[eng] || '#6B7280')

    return {
      labels: labels.map((l) => l.charAt(0).toUpperCase() + l.slice(1)),
      datasets: [
        {
          label: 'Cost ($)',
          data,
          backgroundColor: colors
        }
      ]
    }
  }

  const formatCost = (cost) => {
    if (cost < 0.01) return `$${(cost * 100).toFixed(2)}¢`
    return `$${cost.toFixed(4)}`
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="p-3 bg-yellow-100 rounded-xl">
          <DollarSign className="w-6 h-6 text-yellow-600" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Cost Tracker</h2>
          <p className="text-gray-600">Monitor API usage and costs</p>
        </div>
      </div>

      {/* Controls */}
      <div className="flex gap-2">
        <button
          onClick={fetchStats}
          disabled={loading}
          className="btn-secondary flex items-center gap-2"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
        <button
          onClick={handleClearCache}
          className="px-4 py-2 bg-orange-100 text-orange-700 rounded-lg hover:bg-orange-200 transition-colors flex items-center gap-2"
        >
          <Trash2 className="w-4 h-4" />
          Clear Cache
        </button>
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

      {!loading && stats && (
        <>
          {/* Summary cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="card">
              <p className="text-sm text-gray-600">Total Queries</p>
              <p className="text-3xl font-bold text-gray-900">{stats.total_queries || 0}</p>
            </div>
            <div className="card">
              <p className="text-sm text-gray-600">Total Cost</p>
              <p className="text-3xl font-bold text-green-600">
                {formatCost(stats.total_cost || 0)}
              </p>
            </div>
            <div className="card">
              <p className="text-sm text-gray-600">Cache Hit Rate</p>
              <p className="text-3xl font-bold text-blue-600">
                {((stats.cache_hit_rate || 0) * 100).toFixed(1)}%
              </p>
            </div>
            <div className="card">
              <p className="text-sm text-gray-600">Engines Used</p>
              <p className="text-3xl font-bold text-purple-600">
                {Object.keys(stats.by_engine || {}).length}
              </p>
            </div>
          </div>

          {/* Charts */}
          <div className="grid md:grid-cols-2 gap-6">
            {/* Queries by engine */}
            <div className="card">
              <h3 className="font-medium text-gray-900 mb-4">Queries by Engine</h3>
              {getEngineChartData() && getEngineChartData().labels.length > 0 ? (
                <div className="h-64 flex items-center justify-center">
                  <Doughnut
                    data={getEngineChartData()}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: {
                          position: 'bottom'
                        }
                      }
                    }}
                  />
                </div>
              ) : (
                <div className="h-64 flex items-center justify-center text-gray-400">
                  No data yet
                </div>
              )}
            </div>

            {/* Cost by engine */}
            <div className="card">
              <h3 className="font-medium text-gray-900 mb-4">Cost by Engine</h3>
              {getCostChartData() && getCostChartData().labels.length > 0 ? (
                <div className="h-64">
                  <Bar
                    data={getCostChartData()}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: {
                          display: false
                        }
                      },
                      scales: {
                        y: {
                          beginAtZero: true,
                          ticks: {
                            callback: (value) => `$${value.toFixed(4)}`
                          }
                        }
                      }
                    }}
                  />
                </div>
              ) : (
                <div className="h-64 flex items-center justify-center text-gray-400">
                  No cost data yet
                </div>
              )}
            </div>
          </div>

          {/* Engine breakdown table */}
          <div className="card">
            <h3 className="font-medium text-gray-900 mb-4">Engine Breakdown</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-2 px-3 font-medium text-gray-600">Engine</th>
                    <th className="text-right py-2 px-3 font-medium text-gray-600">Queries</th>
                    <th className="text-right py-2 px-3 font-medium text-gray-600">Cost</th>
                    <th className="text-right py-2 px-3 font-medium text-gray-600">Avg Time</th>
                    <th className="text-right py-2 px-3 font-medium text-gray-600">Cached</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(stats.by_engine || {}).map(([engine, data]) => (
                    <tr key={engine} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-2 px-3">
                        <span
                          className="px-2 py-0.5 rounded text-xs font-medium"
                          style={{
                            backgroundColor: `${engineColors[engine]}20`,
                            color: engineColors[engine]
                          }}
                        >
                          {engine}
                        </span>
                      </td>
                      <td className="text-right py-2 px-3 font-mono">{data.queries}</td>
                      <td className="text-right py-2 px-3 font-mono text-green-600">
                        {formatCost(data.cost || 0)}
                      </td>
                      <td className="text-right py-2 px-3 font-mono">
                        {data.avg_duration_ms?.toFixed(0) || 0}ms
                      </td>
                      <td className="text-right py-2 px-3 font-mono">
                        {data.cached_queries || 0}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Cache stats */}
          {cacheStats && (
            <div className="card">
              <div className="flex items-center gap-2 mb-4">
                <Database className="w-5 h-5 text-gray-600" />
                <h3 className="font-medium text-gray-900">Cache Statistics</h3>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="p-3 bg-blue-50 rounded-lg">
                  <p className="text-blue-700 text-xs font-medium">Hits</p>
                  <p className="text-xl font-bold text-blue-900">{cacheStats.hits || 0}</p>
                </div>
                <div className="p-3 bg-red-50 rounded-lg">
                  <p className="text-red-700 text-xs font-medium">Misses</p>
                  <p className="text-xl font-bold text-red-900">{cacheStats.misses || 0}</p>
                </div>
                <div className="p-3 bg-green-50 rounded-lg">
                  <p className="text-green-700 text-xs font-medium">Hit Rate</p>
                  <p className="text-xl font-bold text-green-900">
                    {cacheStats.hit_rate !== undefined
                      ? `${(cacheStats.hit_rate * 100).toFixed(1)}%`
                      : 'N/A'}
                  </p>
                </div>
                <div className="p-3 bg-purple-50 rounded-lg">
                  <p className="text-purple-700 text-xs font-medium">Backend</p>
                  <p className="text-xl font-bold text-purple-900">
                    {cacheStats.backend || 'memory'}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Budget projection */}
          <div className="card bg-gradient-to-r from-yellow-50 to-orange-50">
            <h3 className="font-medium text-gray-900 mb-3">Budget Projection</h3>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <p className="text-sm text-gray-600">Daily (est.)</p>
                <p className="text-lg font-bold text-gray-900">
                  {formatCost((stats.total_cost || 0) * 1)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Monthly (est.)</p>
                <p className="text-lg font-bold text-gray-900">
                  {formatCost((stats.total_cost || 0) * 30)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Yearly (est.)</p>
                <p className="text-lg font-bold text-gray-900">
                  {formatCost((stats.total_cost || 0) * 365)}
                </p>
              </div>
            </div>
            <p className="text-xs text-gray-500 mt-3 text-center">
              *Estimates based on current session usage
            </p>
          </div>
        </>
      )}
    </div>
  )
}

export default CostTracker
