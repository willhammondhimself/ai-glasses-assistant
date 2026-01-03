import React, { useState, useCallback } from 'react'
import { Calculator, Play, Loader2, AlertCircle } from 'lucide-react'
import { mathSolve, mathCalculus, mathAlgebra } from '../services/api'
import { useVoiceInput } from '../hooks/useVoiceInput'
import { MicButton } from './MicButton'
import { VoiceFeedback } from './VoiceFeedback'

function MathSolver() {
  const [mode, setMode] = useState('solve')
  const [input, setInput] = useState('')
  const [variable, setVariable] = useState('x')
  const [operation, setOperation] = useState('differentiate')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [duration, setDuration] = useState(null)

  // Voice input handler
  const handleVoiceResult = useCallback((parsed) => {
    // Accept if no engine specified or if it's math
    if (!parsed.engine || parsed.engine === 'math') {
      setInput(parsed.query)
    }
  }, [])

  const voice = useVoiceInput({ onResult: handleVoiceResult })

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      let response
      switch (mode) {
        case 'solve':
          response = await mathSolve(input)
          break
        case 'calculus':
          response = await mathCalculus(input, operation, variable)
          break
        case 'algebra':
          response = await mathAlgebra(input, variable)
          break
        default:
          response = await mathSolve(input)
      }
      setResult(response.data)
      setDuration(response.durationMs)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const examples = {
    solve: ['2 + 2', 'sqrt(16) + 5', 'sin(pi/4)', 'integrate(x**2, x)'],
    calculus: ['x**3 + 2*x', 'sin(x)*cos(x)', 'exp(-x**2)', 'ln(x)'],
    algebra: ['x**2 - 5*x + 6 = 0', '2*x + 3 = 7', 'x**3 - 8 = 0']
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="p-3 bg-blue-100 rounded-xl">
          <Calculator className="w-6 h-6 text-blue-600" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Math Solver</h2>
          <p className="text-gray-600">Solve expressions, calculus, and algebra with SymPy</p>
        </div>
      </div>

      {/* Mode selector */}
      <div className="flex gap-2">
        {['solve', 'calculus', 'algebra'].map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              mode === m
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            {m.charAt(0).toUpperCase() + m.slice(1)}
          </button>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="card space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            {mode === 'solve' && 'Expression'}
            {mode === 'calculus' && 'Function'}
            {mode === 'algebra' && 'Equation'}
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={examples[mode][0]}
              className="input-field font-mono flex-1"
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
        </div>

        {mode === 'calculus' && (
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Operation
              </label>
              <select
                value={operation}
                onChange={(e) => setOperation(e.target.value)}
                className="input-field"
              >
                <option value="differentiate">Differentiate</option>
                <option value="integrate">Integrate</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Variable
              </label>
              <input
                type="text"
                value={variable}
                onChange={(e) => setVariable(e.target.value)}
                className="input-field font-mono"
              />
            </div>
          </div>
        )}

        {mode === 'algebra' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Solve for
            </label>
            <input
              type="text"
              value={variable}
              onChange={(e) => setVariable(e.target.value)}
              className="input-field font-mono w-24"
            />
          </div>
        )}

        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="btn-primary flex items-center gap-2"
        >
          {loading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Play className="w-4 h-4" />
          )}
          {loading ? 'Computing...' : 'Solve'}
        </button>
      </form>

      {/* Examples */}
      <div className="card">
        <h3 className="text-sm font-medium text-gray-700 mb-2">Examples</h3>
        <div className="flex flex-wrap gap-2">
          {examples[mode].map((ex) => (
            <button
              key={ex}
              onClick={() => setInput(ex)}
              className="px-3 py-1 bg-gray-100 rounded-lg text-sm font-mono hover:bg-gray-200 transition-colors"
            >
              {ex}
            </button>
          ))}
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

      {/* Result display */}
      {result && (
        <div className="card space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-medium text-gray-900">Result</h3>
            {duration && (
              <span className="text-xs text-gray-500">{duration}ms</span>
            )}
          </div>

          <div className="result-box result-success">
            <p className="text-lg font-mono">
              {result.result || result.solution || result.solutions?.join(', ')}
            </p>
          </div>

          {result.steps && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-2">Steps</h4>
              <div className="space-y-1">
                {result.steps.map((step, i) => (
                  <p key={i} className="text-sm text-gray-600 font-mono">
                    {i + 1}. {step}
                  </p>
                ))}
              </div>
            </div>
          )}

          {result.method && (
            <p className="text-xs text-gray-500">Method: {result.method}</p>
          )}
        </div>
      )}
    </div>
  )
}

export default MathSolver
