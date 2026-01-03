import React, { useState, useCallback } from 'react'
import { BarChart3, Play, Loader2, AlertCircle, TrendingUp, Sigma, TestTube } from 'lucide-react'
import { Line, Bar, Scatter } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js'
import {
  statsDescriptive,
  statsHypothesis,
  statsCorrelation,
  statsRegression,
  statsProbability,
  statsConcept
} from '../services/api'
import { useVoiceInput } from '../hooks/useVoiceInput'
import { MicButton } from './MicButton'
import { VoiceFeedback } from './VoiceFeedback'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
)

function StatsCalculator() {
  const [mode, setMode] = useState('descriptive')
  const [data, setData] = useState('')
  const [dataY, setDataY] = useState('')
  const [testType, setTestType] = useState('t_test')
  const [testData, setTestData] = useState({
    sample1: '',
    sample2: '',
    observed: '',
    expected: ''
  })
  const [problem, setProblem] = useState('')
  const [concept, setConcept] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [duration, setDuration] = useState(null)

  // Voice input handler
  const handleVoiceResult = useCallback((parsed) => {
    if (!parsed.engine || parsed.engine === 'statistics') {
      if (mode === 'probability') setProblem(parsed.query)
      else if (mode === 'concept') setConcept(parsed.query)
      else setData(parsed.query)
    }
  }, [mode])

  const voice = useVoiceInput({ onResult: handleVoiceResult })

  const modes = [
    { id: 'descriptive', label: 'Descriptive Stats', icon: Sigma },
    { id: 'hypothesis', label: 'Hypothesis Test', icon: TestTube },
    { id: 'correlation', label: 'Correlation', icon: TrendingUp },
    { id: 'regression', label: 'Regression', icon: TrendingUp },
    { id: 'probability', label: 'Probability', icon: BarChart3 },
    { id: 'concept', label: 'Explain', icon: BarChart3 }
  ]

  const parseData = (str) => {
    return str.split(/[\s,]+/).filter(Boolean).map(Number).filter(n => !isNaN(n))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      let response
      switch (mode) {
        case 'descriptive':
          const parsedData = parseData(data)
          if (parsedData.length === 0) return
          response = await statsDescriptive(parsedData)
          break
        case 'hypothesis':
          let formattedTestData = {}
          if (testType === 't_test') {
            formattedTestData = {
              sample1: parseData(testData.sample1),
              sample2: parseData(testData.sample2)
            }
          } else if (testType === 'chi_square') {
            formattedTestData = {
              observed: parseData(testData.observed),
              expected: parseData(testData.expected)
            }
          } else if (testType === 'anova') {
            formattedTestData = {
              groups: [
                parseData(testData.sample1),
                parseData(testData.sample2)
              ]
            }
          }
          response = await statsHypothesis(testType, formattedTestData)
          break
        case 'correlation':
          const x = parseData(data)
          const y = parseData(dataY)
          if (x.length === 0 || y.length === 0) return
          response = await statsCorrelation(x, y)
          break
        case 'regression':
          const xReg = parseData(data)
          const yReg = parseData(dataY)
          if (xReg.length === 0 || yReg.length === 0) return
          response = await statsRegression(xReg, yReg)
          break
        case 'probability':
          if (!problem.trim()) return
          response = await statsProbability(problem)
          break
        case 'concept':
          if (!concept.trim()) return
          response = await statsConcept(concept)
          break
        default:
          return
      }
      setResult(response.data)
      setDuration(response.durationMs)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const dataExamples = {
    simple: '12, 15, 18, 22, 25, 28, 30, 35',
    normal: '45, 48, 52, 55, 58, 60, 62, 65, 68, 72',
    bimodal: '10, 12, 11, 45, 47, 46, 48, 13, 11, 44'
  }

  const conceptExamples = [
    'Central limit theorem',
    'P-value interpretation',
    'Type I vs Type II error',
    'Confidence intervals',
    'Standard deviation'
  ]

  const probabilityExamples = [
    'What is the probability of getting exactly 3 heads in 5 coin flips?',
    'In a normal distribution with mean 100 and std 15, what percent is above 130?'
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="p-3 bg-cyan-100 rounded-xl">
          <BarChart3 className="w-6 h-6 text-cyan-600" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Statistics Calculator</h2>
          <p className="text-gray-600">Descriptive stats, hypothesis tests, regression</p>
        </div>
      </div>

      {/* Mode selector */}
      <div className="flex gap-2 flex-wrap">
        {modes.map((m) => (
          <button
            key={m.id}
            onClick={() => setMode(m.id)}
            className={`px-3 py-2 rounded-lg font-medium text-sm flex items-center gap-2 transition-colors ${
              mode === m.id
                ? 'bg-cyan-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <m.icon className="w-4 h-4" />
            {m.label}
          </button>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {mode === 'descriptive' && (
          <div className="card">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Data (comma or space separated)
            </label>
            <textarea
              value={data}
              onChange={(e) => setData(e.target.value)}
              placeholder="12, 15, 18, 22, 25..."
              rows={3}
              className="input-field font-mono resize-none"
            />
            <div className="mt-3 flex flex-wrap gap-2">
              {Object.entries(dataExamples).map(([label, ex]) => (
                <button
                  key={label}
                  type="button"
                  onClick={() => setData(ex)}
                  className="px-2 py-1 bg-gray-100 rounded text-sm hover:bg-gray-200"
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        )}

        {mode === 'hypothesis' && (
          <>
            <div className="card">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Test Type
              </label>
              <select
                value={testType}
                onChange={(e) => setTestType(e.target.value)}
                className="input-field"
              >
                <option value="t_test">Two-Sample T-Test</option>
                <option value="chi_square">Chi-Square Test</option>
                <option value="anova">One-Way ANOVA</option>
              </select>
            </div>

            {testType === 't_test' && (
              <div className="card grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Sample 1
                  </label>
                  <textarea
                    value={testData.sample1}
                    onChange={(e) => setTestData({ ...testData, sample1: e.target.value })}
                    placeholder="23, 25, 28, 30..."
                    rows={2}
                    className="input-field font-mono resize-none"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Sample 2
                  </label>
                  <textarea
                    value={testData.sample2}
                    onChange={(e) => setTestData({ ...testData, sample2: e.target.value })}
                    placeholder="30, 32, 35, 38..."
                    rows={2}
                    className="input-field font-mono resize-none"
                  />
                </div>
              </div>
            )}

            {testType === 'chi_square' && (
              <div className="card grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Observed Frequencies
                  </label>
                  <textarea
                    value={testData.observed}
                    onChange={(e) => setTestData({ ...testData, observed: e.target.value })}
                    placeholder="45, 35, 20..."
                    rows={2}
                    className="input-field font-mono resize-none"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Expected Frequencies
                  </label>
                  <textarea
                    value={testData.expected}
                    onChange={(e) => setTestData({ ...testData, expected: e.target.value })}
                    placeholder="40, 40, 20..."
                    rows={2}
                    className="input-field font-mono resize-none"
                  />
                </div>
              </div>
            )}

            {testType === 'anova' && (
              <div className="card grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Group 1
                  </label>
                  <textarea
                    value={testData.sample1}
                    onChange={(e) => setTestData({ ...testData, sample1: e.target.value })}
                    placeholder="23, 25, 28..."
                    rows={2}
                    className="input-field font-mono resize-none"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Group 2
                  </label>
                  <textarea
                    value={testData.sample2}
                    onChange={(e) => setTestData({ ...testData, sample2: e.target.value })}
                    placeholder="30, 32, 35..."
                    rows={2}
                    className="input-field font-mono resize-none"
                  />
                </div>
              </div>
            )}
          </>
        )}

        {(mode === 'correlation' || mode === 'regression') && (
          <div className="card grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                X Values
              </label>
              <textarea
                value={data}
                onChange={(e) => setData(e.target.value)}
                placeholder="1, 2, 3, 4, 5..."
                rows={3}
                className="input-field font-mono resize-none"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Y Values
              </label>
              <textarea
                value={dataY}
                onChange={(e) => setDataY(e.target.value)}
                placeholder="2, 4, 5, 4, 5..."
                rows={3}
                className="input-field font-mono resize-none"
              />
            </div>
          </div>
        )}

        {mode === 'probability' && (
          <div className="card">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Probability Problem
            </label>
            <div className="relative">
              <textarea
                value={problem}
                onChange={(e) => setProblem(e.target.value)}
                placeholder="Describe the probability problem..."
                rows={3}
                className="input-field resize-none pr-12"
              />
              <div className="absolute top-2 right-2">
                <MicButton
                  isListening={voice.isListening}
                  isSupported={voice.isSupported}
                  onStart={voice.startListening}
                  onStop={voice.stopListening}
                  size="sm"
                />
              </div>
            </div>
            <VoiceFeedback
              transcript={voice.transcript}
              interimTranscript={voice.interimTranscript}
              confidence={voice.confidence}
              isListening={voice.isListening}
              error={voice.error}
            />
            <div className="mt-3 space-y-1">
              {probabilityExamples.map((ex) => (
                <button
                  key={ex}
                  type="button"
                  onClick={() => setProblem(ex)}
                  className="block w-full text-left px-2 py-1 bg-gray-100 rounded text-sm hover:bg-gray-200"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}

        {mode === 'concept' && (
          <div className="card">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Statistics Concept
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={concept}
                onChange={(e) => setConcept(e.target.value)}
                placeholder="e.g., Central limit theorem"
                className="input-field flex-1"
              />
              <MicButton
                isListening={voice.isListening}
                isSupported={voice.isSupported}
                onStart={voice.startListening}
                onStop={voice.stopListening}
              />
            </div>
            <VoiceFeedback
              transcript={voice.transcript}
              interimTranscript={voice.interimTranscript}
              confidence={voice.confidence}
              isListening={voice.isListening}
              error={voice.error}
            />
            <div className="mt-3 flex flex-wrap gap-2">
              {conceptExamples.map((ex) => (
                <button
                  key={ex}
                  type="button"
                  onClick={() => setConcept(ex)}
                  className="px-2 py-1 bg-gray-100 rounded text-sm hover:bg-gray-200"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          className="btn-primary flex items-center gap-2"
        >
          {loading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Play className="w-4 h-4" />
          )}
          {loading ? 'Computing...' : 'Calculate'}
        </button>
      </form>

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

      {/* Result display */}
      {result && (
        <div className="card space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-medium text-gray-900">Result</h3>
            {duration && (
              <span className="text-xs text-gray-500">{duration}ms</span>
            )}
          </div>

          {/* Descriptive stats */}
          {result.mean !== undefined && mode === 'descriptive' && (
            <>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div className="p-3 bg-blue-50 rounded-lg">
                  <p className="text-blue-700 font-medium text-xs">Mean</p>
                  <p className="text-xl font-bold text-blue-900">{result.mean?.toFixed(4)}</p>
                </div>
                <div className="p-3 bg-green-50 rounded-lg">
                  <p className="text-green-700 font-medium text-xs">Median</p>
                  <p className="text-xl font-bold text-green-900">{result.median?.toFixed(4)}</p>
                </div>
                <div className="p-3 bg-purple-50 rounded-lg">
                  <p className="text-purple-700 font-medium text-xs">Std Dev</p>
                  <p className="text-xl font-bold text-purple-900">{result.std_dev?.toFixed(4)}</p>
                </div>
                <div className="p-3 bg-orange-50 rounded-lg">
                  <p className="text-orange-700 font-medium text-xs">Count</p>
                  <p className="text-xl font-bold text-orange-900">{result.count}</p>
                </div>
              </div>

              {result.quartiles && (
                <div className="grid grid-cols-3 gap-4">
                  <div className="p-3 bg-gray-50 rounded-lg text-center">
                    <p className="text-gray-600 text-xs">Q1 (25%)</p>
                    <p className="font-mono">{result.quartiles.q1?.toFixed(2)}</p>
                  </div>
                  <div className="p-3 bg-gray-50 rounded-lg text-center">
                    <p className="text-gray-600 text-xs">Q2 (50%)</p>
                    <p className="font-mono">{result.quartiles.q2?.toFixed(2)}</p>
                  </div>
                  <div className="p-3 bg-gray-50 rounded-lg text-center">
                    <p className="text-gray-600 text-xs">Q3 (75%)</p>
                    <p className="font-mono">{result.quartiles.q3?.toFixed(2)}</p>
                  </div>
                </div>
              )}
            </>
          )}

          {/* Hypothesis test results */}
          {result.test_statistic !== undefined && (
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-blue-50 rounded-lg">
                <p className="text-blue-700 font-medium text-sm">Test Statistic</p>
                <p className="text-2xl font-bold text-blue-900">{result.test_statistic?.toFixed(4)}</p>
              </div>
              <div className={`p-4 rounded-lg ${result.p_value < 0.05 ? 'bg-green-50' : 'bg-yellow-50'}`}>
                <p className={`font-medium text-sm ${result.p_value < 0.05 ? 'text-green-700' : 'text-yellow-700'}`}>
                  P-Value
                </p>
                <p className={`text-2xl font-bold ${result.p_value < 0.05 ? 'text-green-900' : 'text-yellow-900'}`}>
                  {result.p_value?.toFixed(4)}
                </p>
                <p className={`text-xs mt-1 ${result.p_value < 0.05 ? 'text-green-600' : 'text-yellow-600'}`}>
                  {result.p_value < 0.05 ? 'Statistically significant' : 'Not significant at α=0.05'}
                </p>
              </div>
            </div>
          )}

          {/* Correlation results */}
          {result.pearson !== undefined && (
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-blue-50 rounded-lg">
                <p className="text-blue-700 font-medium text-sm">Pearson r</p>
                <p className="text-2xl font-bold text-blue-900">{result.pearson?.coefficient?.toFixed(4)}</p>
                <p className="text-xs text-blue-600">p = {result.pearson?.p_value?.toFixed(4)}</p>
              </div>
              <div className="p-4 bg-purple-50 rounded-lg">
                <p className="text-purple-700 font-medium text-sm">Spearman ρ</p>
                <p className="text-2xl font-bold text-purple-900">{result.spearman?.coefficient?.toFixed(4)}</p>
                <p className="text-xs text-purple-600">p = {result.spearman?.p_value?.toFixed(4)}</p>
              </div>
            </div>
          )}

          {/* Regression results */}
          {result.slope !== undefined && (
            <>
              <div className="p-4 bg-green-50 rounded-lg">
                <p className="text-green-700 font-medium text-sm">Regression Equation</p>
                <p className="text-xl font-mono text-green-900">
                  y = {result.slope?.toFixed(4)}x + {result.intercept?.toFixed(4)}
                </p>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 bg-gray-50 rounded-lg">
                  <p className="text-gray-600 text-xs">R²</p>
                  <p className="font-bold">{result.r_squared?.toFixed(4)}</p>
                </div>
                <div className="p-3 bg-gray-50 rounded-lg">
                  <p className="text-gray-600 text-xs">Std Error</p>
                  <p className="font-bold">{result.std_error?.toFixed(4)}</p>
                </div>
              </div>
            </>
          )}

          {/* Explanation/Answer */}
          {(result.explanation || result.answer || result.solution) && (
            <div className="result-box result-success whitespace-pre-wrap">
              {result.explanation || result.answer || result.solution}
            </div>
          )}

          {result.key_points && result.key_points.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-2">Key Points</h4>
              <ul className="space-y-1">
                {result.key_points.map((point, i) => (
                  <li key={i} className="text-sm text-gray-600 flex items-start gap-2">
                    <span className="text-cyan-500">•</span>
                    {point}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default StatsCalculator
