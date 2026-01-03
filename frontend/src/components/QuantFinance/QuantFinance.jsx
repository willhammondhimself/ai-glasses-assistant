import React, { useState, useEffect, useRef, useCallback } from 'react'
import {
  Brain,
  Dice5,
  TrendingUp,
  Target,
  HelpCircle,
  GraduationCap,
  Play,
  Loader2,
  AlertCircle,
  CheckCircle,
  XCircle,
  Clock,
  Zap,
  BarChart2,
  Award,
  ChevronRight,
  BookOpen,
  Download
} from 'lucide-react'
import { useVoiceInput } from '../../hooks/useVoiceInput'
import { MicButton } from '../MicButton'
import { VoiceFeedback } from '../VoiceFeedback'
import LatexRenderer from '../shared/LatexRenderer'
import { ExportPDF, exportSolutionToPDF } from '../shared/ExportPDF'
import {
  quantMentalMathGenerate,
  quantMentalMathCheck,
  quantMentalMathTypes,
  quantProbabilityGenerate,
  quantProbabilityCheck,
  quantOptionsBlackScholes,
  quantOptionsGreeks,
  quantMarketKelly,
  quantMarketEdge,
  quantFermiGenerate,
  quantFermiHint,
  quantFermiEvaluate,
  quantInterviewStart,
  quantInterviewNext,
  quantInterviewEnd,
  quantInterviewFirms,
  quantProgress,
  flashcardGenerate
} from '../../services/api'

const modes = [
  { id: 'mental_math', label: 'Mental Math', icon: Brain, color: 'blue' },
  { id: 'probability', label: 'Probability', icon: Dice5, color: 'purple' },
  { id: 'options', label: 'Options', icon: TrendingUp, color: 'green' },
  { id: 'market', label: 'Market Making', icon: Target, color: 'orange' },
  { id: 'fermi', label: 'Fermi', icon: HelpCircle, color: 'cyan' },
  { id: 'interview', label: 'Interview Mode', icon: GraduationCap, color: 'red' }
]

function QuantFinance() {
  const [mode, setMode] = useState('mental_math')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="p-3 bg-indigo-100 rounded-xl">
          <Brain className="w-6 h-6 text-indigo-600" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Quant Finance</h2>
          <p className="text-gray-600">Interview prep for Jane Street, Citadel, Two Sigma</p>
        </div>
      </div>

      {/* Mode Tabs */}
      <div className="flex gap-2 flex-wrap">
        {modes.map((m) => (
          <button
            key={m.id}
            onClick={() => setMode(m.id)}
            className={`px-4 py-2 rounded-lg font-medium text-sm flex items-center gap-2 transition-colors ${
              mode === m.id
                ? `bg-${m.color}-600 text-white`
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
            style={mode === m.id ? { backgroundColor: getColor(m.color) } : {}}
          >
            <m.icon className="w-4 h-4" />
            {m.label}
          </button>
        ))}
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-medium text-red-800">Error</p>
            <p className="text-red-700">{error}</p>
          </div>
        </div>
      )}

      {/* Content based on mode */}
      {mode === 'mental_math' && <MentalMathMode setError={setError} />}
      {mode === 'probability' && <ProbabilityMode setError={setError} />}
      {mode === 'options' && <OptionsMode setError={setError} />}
      {mode === 'market' && <MarketMode setError={setError} />}
      {mode === 'fermi' && <FermiMode setError={setError} />}
      {mode === 'interview' && <InterviewMode setError={setError} />}
    </div>
  )
}

function getColor(color) {
  const colors = {
    blue: '#2563eb',
    purple: '#7c3aed',
    green: '#059669',
    orange: '#ea580c',
    cyan: '#0891b2',
    red: '#dc2626'
  }
  return colors[color] || '#6b7280'
}

// ==================== Flashcard Prompt Component ====================

function FlashcardPrompt({ problem, answer, explanation, problemType, onCreated }) {
  const [creating, setCreating] = useState(false)
  const [created, setCreated] = useState(false)

  const createFlashcard = async () => {
    setCreating(true)
    try {
      await flashcardGenerate({
        source_type: 'quant',
        problem_data: {
          problem,
          answer: String(answer),
          explanation,
          problem_type: problemType
        }
      })
      setCreated(true)
      onCreated?.()
    } catch (err) {
      console.error('Failed to create flashcard:', err)
    } finally {
      setCreating(false)
    }
  }

  if (created) {
    return (
      <div className="flex items-center gap-2 text-green-600 text-sm">
        <CheckCircle className="w-4 h-4" />
        Flashcard created!
      </div>
    )
  }

  return (
    <button
      onClick={createFlashcard}
      disabled={creating}
      className="flex items-center gap-2 px-3 py-1.5 bg-violet-100 text-violet-700 rounded-lg hover:bg-violet-200 transition-colors text-sm"
    >
      {creating ? (
        <Loader2 className="w-4 h-4 animate-spin" />
      ) : (
        <BookOpen className="w-4 h-4" />
      )}
      Create Flashcard
    </button>
  )
}

// ==================== Mental Math Mode ====================

function MentalMathMode({ setError }) {
  const [problem, setProblem] = useState(null)
  const [answer, setAnswer] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [difficulty, setDifficulty] = useState(2)
  const [problemType, setProblemType] = useState('')
  const [startTime, setStartTime] = useState(null)
  const [streak, setStreak] = useState(0)
  const inputRef = useRef(null)

  // Voice input for answers
  const handleVoiceResult = useCallback((parsed) => {
    if (!parsed.engine || parsed.engine === 'quant' || parsed.engine === 'math') {
      setAnswer(parsed.query)
    }
  }, [])
  const voice = useVoiceInput({ onResult: handleVoiceResult })

  const generateProblem = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    setAnswer('')
    try {
      const res = await quantMentalMathGenerate(problemType || null, difficulty)
      setProblem(res.data)
      setStartTime(Date.now())
      setTimeout(() => inputRef.current?.focus(), 100)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  const submitAnswer = async (e) => {
    e.preventDefault()
    if (!problem || !answer) return
    setLoading(true)
    const timeMs = Date.now() - startTime
    try {
      const res = await quantMentalMathCheck(problem.problem_id, answer, timeMs)
      setResult(res.data)
      if (res.data.correct) {
        setStreak((s) => s + 1)
      } else {
        setStreak(0)
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="card flex flex-wrap items-center gap-4">
        <div>
          <label className="text-sm text-gray-600 block mb-1">Difficulty</label>
          <select
            value={difficulty}
            onChange={(e) => setDifficulty(Number(e.target.value))}
            className="input-field w-32"
          >
            <option value={1}>1 - Warmup</option>
            <option value={2}>2 - Interview</option>
            <option value={3}>3 - Advanced</option>
            <option value={4}>4 - Expert</option>
          </select>
        </div>
        <div>
          <label className="text-sm text-gray-600 block mb-1">Type</label>
          <select
            value={problemType}
            onChange={(e) => setProblemType(e.target.value)}
            className="input-field w-40"
          >
            <option value="">Random</option>
            <option value="multiplication">Multiplication</option>
            <option value="division">Division</option>
            <option value="percentage">Percentage</option>
            <option value="fraction_decimal">Fractions</option>
            <option value="square_root">Square Roots</option>
          </select>
        </div>
        <div className="flex-1" />
        <div className="flex items-center gap-2">
          <Zap className="w-5 h-5 text-yellow-500" />
          <span className="font-bold text-lg">{streak}</span>
          <span className="text-gray-500 text-sm">streak</span>
        </div>
      </div>

      {/* Problem Display */}
      {!problem && !result && (
        <div className="card text-center py-12">
          <Brain className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <p className="text-gray-500 mb-4">Ready to practice mental math?</p>
          <button onClick={generateProblem} disabled={loading} className="btn-primary">
            {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : 'Start Practice'}
          </button>
        </div>
      )}

      {problem && !result && (
        <div className="card">
          <div className="text-center mb-6">
            <div className="inline-flex items-center gap-2 px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm mb-4">
              <Clock className="w-4 h-4" />
              Target: {problem.time_target_ms / 1000}s
            </div>
            <p className="text-4xl font-bold text-gray-900 font-mono">{problem.problem}</p>
            {problem.hint && (
              <p className="text-sm text-gray-500 mt-3">{problem.hint}</p>
            )}
          </div>
          <form onSubmit={submitAnswer} className="space-y-2">
            <div className="flex gap-3">
              <input
                ref={inputRef}
                type="text"
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
                placeholder="Your answer..."
                className="input-field flex-1 text-center text-2xl font-mono"
                autoComplete="off"
              />
              <MicButton
                isListening={voice.isListening}
                isSupported={voice.isSupported}
                onStart={voice.startListening}
                onStop={voice.stopListening}
              />
              <button type="submit" disabled={loading || !answer} className="btn-primary">
                {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : 'Submit'}
              </button>
            </div>
            <VoiceFeedback
              transcript={voice.transcript}
              interimTranscript={voice.interimTranscript}
              confidence={voice.confidence}
              isListening={voice.isListening}
              error={voice.error}
            />
          </form>
        </div>
      )}

      {result && (
        <div className={`card ${result.correct ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
          <div className="flex items-center gap-4 mb-4">
            {result.correct ? (
              <CheckCircle className="w-12 h-12 text-green-500" />
            ) : (
              <XCircle className="w-12 h-12 text-red-500" />
            )}
            <div>
              <h3 className={`text-2xl font-bold ${result.correct ? 'text-green-700' : 'text-red-700'}`}>
                {result.correct ? 'Correct!' : 'Incorrect'}
              </h3>
              <p className="text-gray-600">{result.feedback}</p>
            </div>
            <div className="ml-auto text-right">
              <p className="text-3xl font-bold text-gray-900">{result.score}</p>
              <p className="text-sm text-gray-500">points</p>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-4 text-center mb-4">
            <div className="p-3 bg-white rounded-lg">
              <p className="text-sm text-gray-500">Your Answer</p>
              <p className="text-xl font-mono font-bold">{result.user_answer}</p>
            </div>
            <div className="p-3 bg-white rounded-lg">
              <p className="text-sm text-gray-500">Correct Answer</p>
              <p className="text-xl font-mono font-bold">{result.correct_answer}</p>
            </div>
            <div className="p-3 bg-white rounded-lg">
              <p className="text-sm text-gray-500">Time</p>
              <p className="text-xl font-mono font-bold">{(result.time_ms / 1000).toFixed(1)}s</p>
            </div>
          </div>
          <div className="flex items-center justify-between">
            <FlashcardPrompt
              problem={problem?.problem}
              answer={result.correct_answer}
              explanation={result.feedback}
              problemType="mental_math"
            />
            <button onClick={generateProblem} className="btn-primary">
              Next Problem
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

// ==================== Probability Mode ====================

function ProbabilityMode({ setError }) {
  const [problem, setProblem] = useState(null)
  const [answer, setAnswer] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [difficulty, setDifficulty] = useState(2)
  const [problemType, setProblemType] = useState('card')

  // Voice input for probability answers
  const handleVoiceResult = useCallback((parsed) => {
    if (!parsed.engine || parsed.engine === 'quant' || parsed.engine === 'statistics') {
      setAnswer(parsed.query)
    }
  }, [])
  const voice = useVoiceInput({ onResult: handleVoiceResult })

  const generateProblem = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    setAnswer('')
    try {
      const res = await quantProbabilityGenerate(problemType, difficulty)
      setProblem(res.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  const submitAnswer = async (e) => {
    e.preventDefault()
    if (!problem || !answer) return
    setLoading(true)
    try {
      const res = await quantProbabilityCheck(problem.problem_id, answer)
      setResult(res.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      <div className="card flex flex-wrap items-center gap-4">
        <div>
          <label className="text-sm text-gray-600 block mb-1">Type</label>
          <select
            value={problemType}
            onChange={(e) => setProblemType(e.target.value)}
            className="input-field w-40"
          >
            <option value="card">Card Problems</option>
            <option value="dice">Dice Problems</option>
            <option value="expected_value">Expected Value</option>
          </select>
        </div>
        <div>
          <label className="text-sm text-gray-600 block mb-1">Difficulty</label>
          <select
            value={difficulty}
            onChange={(e) => setDifficulty(Number(e.target.value))}
            className="input-field w-32"
          >
            <option value={1}>Easy</option>
            <option value={2}>Medium</option>
            <option value={3}>Hard</option>
            <option value={4}>Expert</option>
          </select>
        </div>
        <button onClick={generateProblem} disabled={loading} className="btn-primary ml-auto">
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Generate Problem'}
        </button>
      </div>

      {problem && !result && (
        <div className="card">
          <div className="mb-4">
            <span className="px-2 py-1 bg-purple-100 text-purple-700 rounded text-sm font-medium">
              {problem.problem_type}
            </span>
          </div>
          <p className="text-xl text-gray-900 mb-4">{problem.problem}</p>
          {problem.hint && (
            <p className="text-sm text-gray-500 mb-4">{problem.hint}</p>
          )}
          <form onSubmit={submitAnswer} className="space-y-2">
            <div className="flex gap-3">
              <input
                type="text"
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
                placeholder="Enter probability (decimal or fraction)..."
                className="input-field flex-1"
              />
              <MicButton
                isListening={voice.isListening}
                isSupported={voice.isSupported}
                onStart={voice.startListening}
                onStop={voice.stopListening}
              />
              <button type="submit" disabled={loading || !answer} className="btn-primary">
                Check
              </button>
            </div>
            <VoiceFeedback
              transcript={voice.transcript}
              interimTranscript={voice.interimTranscript}
              confidence={voice.confidence}
              isListening={voice.isListening}
              error={voice.error}
            />
          </form>
        </div>
      )}

      {result && (
        <div className={`card ${result.correct ? 'bg-green-50' : 'bg-red-50'}`}>
          <div className="flex items-center gap-3 mb-4">
            {result.correct ? (
              <CheckCircle className="w-8 h-8 text-green-500" />
            ) : (
              <XCircle className="w-8 h-8 text-red-500" />
            )}
            <h3 className={`text-xl font-bold ${result.correct ? 'text-green-700' : 'text-red-700'}`}>
              {result.correct ? 'Correct!' : 'Incorrect'}
            </h3>
          </div>
          <div className="bg-white p-4 rounded-lg mb-4">
            <p className="text-sm text-gray-500">Correct Answer</p>
            <p className="text-2xl font-mono font-bold">{result.correct_answer}</p>
            <p className="text-gray-600 mt-2">{result.explanation}</p>
          </div>
          <div className="flex items-center justify-between">
            <FlashcardPrompt
              problem={problem?.problem}
              answer={result.correct_answer}
              explanation={result.explanation}
              problemType="probability"
            />
            <button onClick={generateProblem} className="btn-primary">
              Next Problem
            </button>
          </div>
        </div>
      )}

      {!problem && !result && (
        <div className="card text-center py-12">
          <Dice5 className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <p className="text-gray-500">Card problems are the most common in quant interviews</p>
        </div>
      )}
    </div>
  )
}

// ==================== Options Mode ====================

function OptionsMode({ setError }) {
  const [S, setS] = useState(100)
  const [K, setK] = useState(100)
  const [r, setR] = useState(0.05)
  const [sigma, setSigma] = useState(0.2)
  const [T, setT] = useState(1)
  const [optionType, setOptionType] = useState('call')
  const [result, setResult] = useState(null)
  const [greeks, setGreeks] = useState(null)
  const [loading, setLoading] = useState(false)

  const calculate = async () => {
    setLoading(true)
    setError(null)
    try {
      const [bsRes, greeksRes] = await Promise.all([
        quantOptionsBlackScholes(S, K, r, sigma, T, optionType),
        quantOptionsGreeks(S, K, r, sigma, T, optionType)
      ])
      setResult(bsRes.data)
      setGreeks(greeksRes.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      <div className="card">
        <h3 className="font-medium text-gray-900 mb-4">Black-Scholes Calculator</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div>
            <label className="text-sm text-gray-600 block mb-1">Stock Price (S)</label>
            <input
              type="number"
              value={S}
              onChange={(e) => setS(Number(e.target.value))}
              className="input-field"
              step="0.01"
            />
          </div>
          <div>
            <label className="text-sm text-gray-600 block mb-1">Strike (K)</label>
            <input
              type="number"
              value={K}
              onChange={(e) => setK(Number(e.target.value))}
              className="input-field"
              step="0.01"
            />
          </div>
          <div>
            <label className="text-sm text-gray-600 block mb-1">Risk-Free Rate (r)</label>
            <input
              type="number"
              value={r}
              onChange={(e) => setR(Number(e.target.value))}
              className="input-field"
              step="0.01"
            />
          </div>
          <div>
            <label className="text-sm text-gray-600 block mb-1">Volatility (σ)</label>
            <input
              type="number"
              value={sigma}
              onChange={(e) => setSigma(Number(e.target.value))}
              className="input-field"
              step="0.01"
            />
          </div>
          <div>
            <label className="text-sm text-gray-600 block mb-1">Time (T years)</label>
            <input
              type="number"
              value={T}
              onChange={(e) => setT(Number(e.target.value))}
              className="input-field"
              step="0.1"
            />
          </div>
          <div>
            <label className="text-sm text-gray-600 block mb-1">Type</label>
            <select
              value={optionType}
              onChange={(e) => setOptionType(e.target.value)}
              className="input-field"
            >
              <option value="call">Call</option>
              <option value="put">Put</option>
            </select>
          </div>
        </div>
        <button onClick={calculate} disabled={loading} className="btn-primary mt-4">
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Calculate'}
        </button>
      </div>

      {result && (
        <div className="card bg-green-50">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-medium text-gray-900">Option Price</h3>
            <button
              onClick={() => exportSolutionToPDF({
                problem: `Black-Scholes: S=${S}, K=${K}, r=${r}, σ=${sigma}, T=${T}`,
                answer: `$${result.price}`,
                explanation: result.formula
              }, 'Black-Scholes-Calculation')}
              className="flex items-center gap-1 px-2 py-1 bg-white/50 rounded text-sm text-gray-600 hover:bg-white/80"
            >
              <Download className="w-4 h-4" />
              PDF
            </button>
          </div>
          <p className="text-4xl font-bold text-green-700">${result.price}</p>
          <p className="text-sm text-gray-500 mt-2 font-mono">{result.formula}</p>
        </div>
      )}

      {greeks && (
        <div className="card">
          <h3 className="font-medium text-gray-900 mb-4">Greeks</h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {['delta', 'gamma', 'theta', 'vega', 'rho'].map((g) => (
              <div key={g} className="p-3 bg-gray-50 rounded-lg text-center">
                <p className="text-sm text-gray-500 capitalize">{g}</p>
                <p className="text-xl font-mono font-bold">{greeks[g]}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ==================== Market Making Mode ====================

function MarketMode({ setError }) {
  const [tab, setTab] = useState('kelly')
  const [probWin, setProbWin] = useState(0.55)
  const [odds, setOdds] = useState(2)
  const [bankroll, setBankroll] = useState(10000)
  const [payoutWin, setPayoutWin] = useState(100)
  const [payoutLose, setPayoutLose] = useState(50)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const calculateKelly = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await quantMarketKelly(probWin, odds, bankroll)
      setResult(res.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  const calculateEdge = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await quantMarketEdge(probWin, payoutWin, payoutLose)
      setResult(res.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <button
          onClick={() => { setTab('kelly'); setResult(null) }}
          className={`px-4 py-2 rounded-lg ${tab === 'kelly' ? 'bg-orange-600 text-white' : 'bg-gray-100'}`}
        >
          Kelly Criterion
        </button>
        <button
          onClick={() => { setTab('edge'); setResult(null) }}
          className={`px-4 py-2 rounded-lg ${tab === 'edge' ? 'bg-orange-600 text-white' : 'bg-gray-100'}`}
        >
          Edge Calculator
        </button>
      </div>

      {tab === 'kelly' && (
        <div className="card">
          <h3 className="font-medium mb-4">Kelly Criterion: f* = (bp - q) / b</h3>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="text-sm text-gray-600 block mb-1">P(Win)</label>
              <input
                type="number"
                value={probWin}
                onChange={(e) => setProbWin(Number(e.target.value))}
                className="input-field"
                step="0.01"
                min="0"
                max="1"
              />
            </div>
            <div>
              <label className="text-sm text-gray-600 block mb-1">Odds (payout ratio)</label>
              <input
                type="number"
                value={odds}
                onChange={(e) => setOdds(Number(e.target.value))}
                className="input-field"
                step="0.1"
              />
            </div>
            <div>
              <label className="text-sm text-gray-600 block mb-1">Bankroll</label>
              <input
                type="number"
                value={bankroll}
                onChange={(e) => setBankroll(Number(e.target.value))}
                className="input-field"
              />
            </div>
          </div>
          <button onClick={calculateKelly} disabled={loading} className="btn-primary mt-4">
            Calculate Kelly
          </button>
        </div>
      )}

      {tab === 'edge' && (
        <div className="card">
          <h3 className="font-medium mb-4">Edge = P(win) × Payout - P(lose) × Loss</h3>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="text-sm text-gray-600 block mb-1">P(Win)</label>
              <input
                type="number"
                value={probWin}
                onChange={(e) => setProbWin(Number(e.target.value))}
                className="input-field"
                step="0.01"
              />
            </div>
            <div>
              <label className="text-sm text-gray-600 block mb-1">Win Amount</label>
              <input
                type="number"
                value={payoutWin}
                onChange={(e) => setPayoutWin(Number(e.target.value))}
                className="input-field"
              />
            </div>
            <div>
              <label className="text-sm text-gray-600 block mb-1">Lose Amount</label>
              <input
                type="number"
                value={payoutLose}
                onChange={(e) => setPayoutLose(Number(e.target.value))}
                className="input-field"
              />
            </div>
          </div>
          <button onClick={calculateEdge} disabled={loading} className="btn-primary mt-4">
            Calculate Edge
          </button>
        </div>
      )}

      {result && (
        <div className="card bg-orange-50">
          <h3 className="font-medium text-gray-900 mb-4">Result</h3>
          {result.kelly_fraction !== undefined && (
            <>
              <p className="text-3xl font-bold text-orange-700">
                Bet {result.kelly_percent} of bankroll
              </p>
              <p className="text-xl text-gray-700 mt-2">
                Optimal bet: ${result.optimal_bet}
              </p>
              <p className="text-sm text-gray-500 mt-2">{result.formula}</p>
              <p className="text-sm text-gray-500 mt-1">{result.note}</p>
            </>
          )}
          {result.expected_value !== undefined && !result.kelly_fraction && (
            <>
              <p className="text-3xl font-bold text-orange-700">
                EV: ${result.expected_value}
              </p>
              <p className={`text-xl mt-2 ${result.expected_value > 0 ? 'text-green-600' : 'text-red-600'}`}>
                {result.recommendation}
              </p>
              <p className="text-sm text-gray-500 mt-2">{result.formula}</p>
            </>
          )}
        </div>
      )}
    </div>
  )
}

// ==================== Fermi Mode ====================

function FermiMode({ setError }) {
  const [problem, setProblem] = useState(null)
  const [estimate, setEstimate] = useState('')
  const [hints, setHints] = useState([])
  const [hintLevel, setHintLevel] = useState(0)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [category, setCategory] = useState('')

  // Voice input for estimates
  const handleVoiceResult = useCallback((parsed) => {
    if (!parsed.engine || parsed.engine === 'quant') {
      setEstimate(parsed.query)
    }
  }, [])
  const voice = useVoiceInput({ onResult: handleVoiceResult })

  const generateProblem = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    setHints([])
    setHintLevel(0)
    setEstimate('')
    try {
      const res = await quantFermiGenerate(category || null)
      setProblem(res.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  const getHint = async () => {
    if (!problem) return
    try {
      const res = await quantFermiHint(problem.problem_id, hintLevel + 1)
      setHints(res.data.hints)
      setHintLevel(hintLevel + 1)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    }
  }

  const submitEstimate = async (e) => {
    e.preventDefault()
    if (!problem || !estimate) return
    setLoading(true)
    try {
      const res = await quantFermiEvaluate(problem.problem_id, Number(estimate.replace(/,/g, '')))
      setResult(res.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      <div className="card flex items-center gap-4">
        <div>
          <label className="text-sm text-gray-600 block mb-1">Category</label>
          <select
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            className="input-field w-40"
          >
            <option value="">Random</option>
            <option value="market_size">Market Size</option>
            <option value="counting">Counting</option>
            <option value="rates">Rates</option>
            <option value="finance">Finance</option>
          </select>
        </div>
        <button onClick={generateProblem} disabled={loading} className="btn-primary ml-auto">
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Generate Problem'}
        </button>
      </div>

      {problem && !result && (
        <div className="card">
          <span className="px-2 py-1 bg-cyan-100 text-cyan-700 rounded text-sm">
            {problem.category}
          </span>
          <h3 className="text-xl font-bold text-gray-900 mt-3 mb-4">{problem.problem}</h3>

          {hints.length > 0 && (
            <div className="bg-yellow-50 p-4 rounded-lg mb-4">
              <p className="font-medium text-yellow-800 mb-2">Hints:</p>
              <ul className="space-y-1">
                {hints.map((h, i) => (
                  <li key={i} className="text-yellow-700 flex items-start gap-2">
                    <ChevronRight className="w-4 h-4 mt-0.5" />
                    {h}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="flex gap-2 mb-4">
            <button
              onClick={getHint}
              disabled={hintLevel >= problem.hint_count}
              className="px-3 py-1 bg-yellow-100 text-yellow-700 rounded hover:bg-yellow-200"
            >
              Get Hint ({problem.hint_count - hintLevel} left)
            </button>
          </div>

          <form onSubmit={submitEstimate} className="space-y-2">
            <div className="flex gap-3">
              <input
                type="text"
                value={estimate}
                onChange={(e) => setEstimate(e.target.value)}
                placeholder="Your estimate (e.g., 500000)"
                className="input-field flex-1"
              />
              <MicButton
                isListening={voice.isListening}
                isSupported={voice.isSupported}
                onStart={voice.startListening}
                onStop={voice.stopListening}
              />
              <button type="submit" disabled={loading || !estimate} className="btn-primary">
                Submit
              </button>
            </div>
            <VoiceFeedback
              transcript={voice.transcript}
              interimTranscript={voice.interimTranscript}
              confidence={voice.confidence}
              isListening={voice.isListening}
              error={voice.error}
            />
          </form>
        </div>
      )}

      {result && (
        <div className="card">
          <div className="flex items-center gap-3 mb-4">
            <div className={`w-16 h-16 rounded-full flex items-center justify-center ${
              result.score >= 80 ? 'bg-green-100' : result.score >= 40 ? 'bg-yellow-100' : 'bg-red-100'
            }`}>
              <span className={`text-2xl font-bold ${
                result.score >= 80 ? 'text-green-600' : result.score >= 40 ? 'text-yellow-600' : 'text-red-600'
              }`}>
                {result.score}
              </span>
            </div>
            <div>
              <h3 className="text-xl font-bold">{result.feedback}</h3>
              <p className="text-gray-500">
                Order of magnitude difference: {result.order_of_magnitude_diff}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4 mb-4">
            <div className="p-3 bg-gray-50 rounded-lg text-center">
              <p className="text-sm text-gray-500">Your Estimate</p>
              <p className="text-lg font-mono font-bold">{result.estimate.toLocaleString()}</p>
            </div>
            <div className="p-3 bg-gray-50 rounded-lg text-center">
              <p className="text-sm text-gray-500">Expected Low</p>
              <p className="text-lg font-mono font-bold">{result.expected_range.low.toLocaleString()}</p>
            </div>
            <div className="p-3 bg-gray-50 rounded-lg text-center">
              <p className="text-sm text-gray-500">Expected High</p>
              <p className="text-lg font-mono font-bold">{result.expected_range.high.toLocaleString()}</p>
            </div>
          </div>

          <div className="bg-blue-50 p-4 rounded-lg mb-4">
            <p className="font-medium text-blue-800">Solution Approach:</p>
            <p className="text-blue-700 mt-1">{result.solution_approach}</p>
          </div>

          <div className="flex items-center justify-between">
            <FlashcardPrompt
              problem={problem?.problem}
              answer={`${result.expected_range.low.toLocaleString()} - ${result.expected_range.high.toLocaleString()}`}
              explanation={result.solution_approach}
              problemType="fermi"
            />
            <button onClick={generateProblem} className="btn-primary">
              Next Problem
            </button>
          </div>
        </div>
      )}

      {!problem && !result && (
        <div className="card text-center py-12">
          <HelpCircle className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <p className="text-gray-500">"How many piano tuners are in Chicago?"</p>
          <p className="text-sm text-gray-400 mt-2">Practice estimation and problem decomposition</p>
        </div>
      )}
    </div>
  )
}

// ==================== Interview Mode ====================

function InterviewMode({ setError }) {
  const [session, setSession] = useState(null)
  const [currentProblem, setCurrentProblem] = useState(null)
  const [answer, setAnswer] = useState('')
  const [loading, setLoading] = useState(false)
  const [firms, setFirms] = useState({})
  const [selectedFirm, setSelectedFirm] = useState('general')
  const [duration, setDuration] = useState(30)
  const [difficulty, setDifficulty] = useState(2)
  const [startTime, setStartTime] = useState(null)
  const [report, setReport] = useState(null)

  // Voice input for answers
  const handleVoiceResult = useCallback((parsed) => {
    if (!parsed.engine || parsed.engine === 'quant') {
      setAnswer(parsed.query)
    }
  }, [])
  const voice = useVoiceInput({ onResult: handleVoiceResult })

  useEffect(() => {
    quantInterviewFirms().then((res) => setFirms(res.data)).catch(() => {})
  }, [])

  const startSession = async () => {
    setLoading(true)
    setError(null)
    setReport(null)
    try {
      const res = await quantInterviewStart(duration, selectedFirm, difficulty)
      setSession(res.data)
      setCurrentProblem(res.data.first_problem)
      setStartTime(Date.now())
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  const submitAndNext = async (e) => {
    e.preventDefault()
    if (!session) return
    setLoading(true)
    const timeMs = Date.now() - startTime
    try {
      const res = await quantInterviewNext(session.session_id, answer, timeMs)

      if (res.data.overall_accuracy !== undefined) {
        // Session ended
        setReport(res.data)
        setSession(null)
        setCurrentProblem(null)
      } else {
        setCurrentProblem(res.data.next_problem)
        setStartTime(Date.now())
      }
      setAnswer('')
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  const endEarly = async () => {
    if (!session) return
    setLoading(true)
    try {
      const res = await quantInterviewEnd(session.session_id)
      setReport(res.data)
      setSession(null)
      setCurrentProblem(null)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  // Not started
  if (!session && !report) {
    return (
      <div className="space-y-4">
        <div className="card">
          <h3 className="font-medium text-gray-900 mb-4">Start Interview Practice</h3>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="text-sm text-gray-600 block mb-1">Firm Style</label>
              <select
                value={selectedFirm}
                onChange={(e) => setSelectedFirm(e.target.value)}
                className="input-field"
              >
                {Object.keys(firms).map((f) => (
                  <option key={f} value={f}>{f.replace('_', ' ').toUpperCase()}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-sm text-gray-600 block mb-1">Duration (min)</label>
              <select
                value={duration}
                onChange={(e) => setDuration(Number(e.target.value))}
                className="input-field"
              >
                <option value={15}>15 min</option>
                <option value={30}>30 min</option>
                <option value={45}>45 min</option>
                <option value={60}>60 min</option>
              </select>
            </div>
            <div>
              <label className="text-sm text-gray-600 block mb-1">Starting Difficulty</label>
              <select
                value={difficulty}
                onChange={(e) => setDifficulty(Number(e.target.value))}
                className="input-field"
              >
                <option value={1}>Easy</option>
                <option value={2}>Medium</option>
                <option value={3}>Hard</option>
              </select>
            </div>
          </div>
          {firms[selectedFirm] && (
            <p className="text-sm text-gray-500 mt-3">{firms[selectedFirm].description}</p>
          )}
          <button onClick={startSession} disabled={loading} className="btn-primary mt-4 w-full">
            {loading ? <Loader2 className="w-5 h-5 animate-spin mx-auto" /> : 'Start Interview'}
          </button>
        </div>
      </div>
    )
  }

  // Session active
  if (session && currentProblem) {
    return (
      <div className="space-y-4">
        <div className="card flex items-center justify-between">
          <div>
            <span className="text-sm text-gray-500">
              Problem #{(session.problems_completed || 0) + 1}
            </span>
            <span className="mx-2">•</span>
            <span className={`px-2 py-0.5 rounded text-xs font-medium ${
              currentProblem.category === 'mental_math' ? 'bg-blue-100 text-blue-700' :
              currentProblem.category === 'probability' ? 'bg-purple-100 text-purple-700' :
              currentProblem.category === 'expected_value' ? 'bg-green-100 text-green-700' :
              currentProblem.category === 'market_making' ? 'bg-orange-100 text-orange-700' :
              'bg-gray-100 text-gray-700'
            }`}>
              {currentProblem.category?.replace('_', ' ')}
            </span>
          </div>
          <button onClick={endEarly} className="text-red-600 hover:underline text-sm">
            End Session
          </button>
        </div>

        <div className="card">
          <p className="text-xl text-gray-900 mb-4">{currentProblem.problem}</p>
          {currentProblem.hint && (
            <p className="text-sm text-gray-500 mb-4">{currentProblem.hint}</p>
          )}
          <form onSubmit={submitAndNext} className="space-y-2">
            <div className="flex gap-3">
              <input
                type="text"
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
                placeholder="Your answer..."
                className="input-field flex-1"
                autoFocus
              />
              <MicButton
                isListening={voice.isListening}
                isSupported={voice.isSupported}
                onStart={voice.startListening}
                onStop={voice.stopListening}
              />
              <button type="submit" disabled={loading} className="btn-primary">
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Submit & Next'}
              </button>
            </div>
            <VoiceFeedback
              transcript={voice.transcript}
              interimTranscript={voice.interimTranscript}
              confidence={voice.confidence}
              isListening={voice.isListening}
              error={voice.error}
            />
          </form>
        </div>
      </div>
    )
  }

  // Report
  if (report) {
    return (
      <div className="space-y-4">
        <div className="card bg-gradient-to-r from-indigo-50 to-purple-50">
          <div className="flex items-center gap-4">
            <div className={`w-20 h-20 rounded-full flex items-center justify-center ${
              report.interview_ready ? 'bg-green-100' : 'bg-yellow-100'
            }`}>
              <Award className={`w-10 h-10 ${report.interview_ready ? 'text-green-600' : 'text-yellow-600'}`} />
            </div>
            <div>
              <h3 className="text-2xl font-bold">
                {report.interview_ready ? 'Interview Ready!' : 'Keep Practicing'}
              </h3>
              <p className="text-gray-600">
                Readiness Score: <span className="font-bold text-xl">{report.readiness_score}</span>
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <h3 className="font-medium mb-4">Session Summary</h3>
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center p-3 bg-gray-50 rounded-lg">
              <p className="text-3xl font-bold">{report.total_problems}</p>
              <p className="text-sm text-gray-500">Questions</p>
            </div>
            <div className="text-center p-3 bg-gray-50 rounded-lg">
              <p className="text-3xl font-bold">{report.correct}</p>
              <p className="text-sm text-gray-500">Correct</p>
            </div>
            <div className="text-center p-3 bg-gray-50 rounded-lg">
              <p className="text-3xl font-bold">{report.overall_accuracy}%</p>
              <p className="text-sm text-gray-500">Accuracy</p>
            </div>
          </div>
        </div>

        {report.recommendations && report.recommendations.length > 0 && (
          <div className="card">
            <h3 className="font-medium mb-3">Recommendations</h3>
            <ul className="space-y-2">
              {report.recommendations.map((r, i) => (
                <li key={i} className="flex items-start gap-2 text-gray-700">
                  <ChevronRight className="w-4 h-4 mt-1 text-indigo-500" />
                  {r}
                </li>
              ))}
            </ul>
          </div>
        )}

        <button onClick={() => setReport(null)} className="btn-primary w-full">
          Start New Session
        </button>
      </div>
    )
  }

  return null
}

export default QuantFinance
