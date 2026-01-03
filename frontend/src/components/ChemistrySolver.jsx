import React, { useState, useCallback } from 'react'
import { FlaskConical, Play, Loader2, AlertCircle, Scale, Beaker, Calculator } from 'lucide-react'
import {
  chemistryBalance,
  chemistryMolecularWeight,
  chemistryMolarity,
  chemistryStoichiometry,
  chemistrySolve
} from '../services/api'
import { useVoiceInput } from '../hooks/useVoiceInput'
import { MicButton } from './MicButton'
import { VoiceFeedback } from './VoiceFeedback'

function ChemistrySolver() {
  const [mode, setMode] = useState('balance')
  const [equation, setEquation] = useState('')
  const [formula, setFormula] = useState('')
  const [soluteMoles, setSoluteMoles] = useState('')
  const [volumeLiters, setVolumeLiters] = useState('')
  const [problem, setProblem] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [duration, setDuration] = useState(null)

  // Voice input handler - routes to appropriate field based on mode
  const handleVoiceResult = useCallback((parsed) => {
    if (!parsed.engine || parsed.engine === 'chemistry') {
      if (mode === 'balance') setEquation(parsed.query)
      else if (mode === 'weight') setFormula(parsed.query)
      else setProblem(parsed.query)
    }
  }, [mode])

  const voice = useVoiceInput({ onResult: handleVoiceResult })

  const modes = [
    { id: 'balance', label: 'Balance Equation', icon: Scale },
    { id: 'weight', label: 'Molecular Weight', icon: Calculator },
    { id: 'molarity', label: 'Molarity', icon: Beaker },
    { id: 'stoich', label: 'Stoichiometry', icon: FlaskConical },
    { id: 'solve', label: 'Word Problem', icon: FlaskConical }
  ]

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      let response
      switch (mode) {
        case 'balance':
          if (!equation.trim()) return
          response = await chemistryBalance(equation)
          break
        case 'weight':
          if (!formula.trim()) return
          response = await chemistryMolecularWeight(formula)
          break
        case 'molarity':
          if (!soluteMoles || !volumeLiters) return
          response = await chemistryMolarity(parseFloat(soluteMoles), parseFloat(volumeLiters))
          break
        case 'stoich':
          if (!problem.trim()) return
          response = await chemistryStoichiometry(problem)
          break
        case 'solve':
          if (!problem.trim()) return
          response = await chemistrySolve(problem)
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

  const examples = {
    balance: ['H2 + O2 -> H2O', 'Fe + O2 -> Fe2O3', 'CH4 + O2 -> CO2 + H2O', 'C6H12O6 + O2 -> CO2 + H2O'],
    weight: ['H2O', 'NaCl', 'C6H12O6', 'H2SO4', 'Ca(OH)2'],
    stoich: [
      'How many grams of H2O are produced from 4 mol H2?',
      'If 50g of NaCl is dissolved, how many moles is that?'
    ],
    solve: [
      'Calculate the pH of a 0.1M HCl solution',
      'What is the oxidation state of Mn in KMnO4?'
    ]
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="p-3 bg-orange-100 rounded-xl">
          <FlaskConical className="w-6 h-6 text-orange-600" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Chemistry Solver</h2>
          <p className="text-gray-600">Balance equations, calculate weights, solve problems</p>
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
                ? 'bg-orange-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <m.icon className="w-4 h-4" />
            {m.label}
          </button>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {mode === 'balance' && (
          <div className="card">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Chemical Equation
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={equation}
                onChange={(e) => setEquation(e.target.value)}
                placeholder="H2 + O2 -> H2O"
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
            <div className="mt-3 flex flex-wrap gap-2">
              {examples.balance.map((ex) => (
                <button
                  key={ex}
                  type="button"
                  onClick={() => setEquation(ex)}
                  className="px-2 py-1 bg-gray-100 rounded text-sm font-mono hover:bg-gray-200"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}

        {mode === 'weight' && (
          <div className="card">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Molecular Formula
            </label>
            <input
              type="text"
              value={formula}
              onChange={(e) => setFormula(e.target.value)}
              placeholder="H2O"
              className="input-field font-mono"
            />
            <div className="mt-3 flex flex-wrap gap-2">
              {examples.weight.map((ex) => (
                <button
                  key={ex}
                  type="button"
                  onClick={() => setFormula(ex)}
                  className="px-2 py-1 bg-gray-100 rounded text-sm font-mono hover:bg-gray-200"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}

        {mode === 'molarity' && (
          <div className="card grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Moles of Solute
              </label>
              <input
                type="number"
                step="0.001"
                value={soluteMoles}
                onChange={(e) => setSoluteMoles(e.target.value)}
                placeholder="0.5"
                className="input-field"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Volume (Liters)
              </label>
              <input
                type="number"
                step="0.001"
                value={volumeLiters}
                onChange={(e) => setVolumeLiters(e.target.value)}
                placeholder="1.0"
                className="input-field"
              />
            </div>
          </div>
        )}

        {(mode === 'stoich' || mode === 'solve') && (
          <div className="card">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              {mode === 'stoich' ? 'Stoichiometry Problem' : 'Chemistry Problem'}
            </label>
            <textarea
              value={problem}
              onChange={(e) => setProblem(e.target.value)}
              placeholder="Describe the problem..."
              rows={4}
              className="input-field resize-none"
            />
            <div className="mt-3 space-y-1">
              {examples[mode].map((ex) => (
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

        <button
          type="submit"
          disabled={loading || (
            mode === 'balance' ? !equation.trim() :
            mode === 'weight' ? !formula.trim() :
            mode === 'molarity' ? (!soluteMoles || !volumeLiters) :
            !problem.trim()
          )}
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

          {result.balanced_equation && (
            <div className="p-4 bg-green-50 rounded-lg">
              <p className="text-green-700 font-medium text-sm mb-1">Balanced Equation</p>
              <p className="text-xl font-mono text-green-900">{result.balanced_equation}</p>
            </div>
          )}

          {result.molecular_weight !== undefined && (
            <div className="p-4 bg-blue-50 rounded-lg">
              <p className="text-blue-700 font-medium text-sm mb-1">Molecular Weight</p>
              <p className="text-2xl font-bold text-blue-900">
                {result.molecular_weight.toFixed(2)} g/mol
              </p>
              {result.composition && (
                <div className="mt-3 text-sm">
                  <p className="font-medium text-blue-700 mb-1">Composition:</p>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(result.composition).map(([element, data]) => (
                      <span key={element} className="px-2 py-1 bg-blue-100 rounded">
                        {element}: {data.count} ({data.mass.toFixed(2)}g)
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {result.molarity !== undefined && (
            <div className="p-4 bg-purple-50 rounded-lg">
              <p className="text-purple-700 font-medium text-sm mb-1">Molarity</p>
              <p className="text-2xl font-bold text-purple-900">
                {result.molarity.toFixed(4)} M
              </p>
            </div>
          )}

          {(result.answer || result.solution) && (
            <div className="result-box result-success whitespace-pre-wrap">
              {result.answer || result.solution}
            </div>
          )}

          {result.steps && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-2">Steps</h4>
              <div className="text-sm text-gray-600 whitespace-pre-wrap">
                {result.steps}
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

export default ChemistrySolver
