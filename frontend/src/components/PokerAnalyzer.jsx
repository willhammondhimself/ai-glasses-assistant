import React, { useState, useCallback } from 'react'
import { Spade, Play, Loader2, AlertCircle, Percent, Brain, FileText } from 'lucide-react'
import { pokerOdds, pokerStrategy, pokerHandAnalysis } from '../services/api'
import { useVoiceInput } from '../hooks/useVoiceInput'
import { MicButton } from './MicButton'
import { VoiceFeedback } from './VoiceFeedback'

function PokerAnalyzer() {
  const [mode, setMode] = useState('odds')
  const [holeCards, setHoleCards] = useState('')
  const [communityCards, setCommunityCards] = useState('')
  const [numOpponents, setNumOpponents] = useState(1)
  const [situation, setSituation] = useState('')
  const [hand, setHand] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [duration, setDuration] = useState(null)

  // Voice input handler
  const handleVoiceResult = useCallback((parsed) => {
    if (!parsed.engine || parsed.engine === 'poker') {
      if (mode === 'odds') setHoleCards(parsed.query)
      else if (mode === 'strategy') setSituation(parsed.query)
      else setHand(parsed.query)
    }
  }, [mode])

  const voice = useVoiceInput({ onResult: handleVoiceResult })

  const modes = [
    { id: 'odds', label: 'Odds Calculator', icon: Percent },
    { id: 'strategy', label: 'Strategy', icon: Brain },
    { id: 'analysis', label: 'Hand Analysis', icon: FileText }
  ]

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      let response
      switch (mode) {
        case 'odds':
          if (!holeCards.trim()) return
          const holeArr = holeCards.split(/[\s,]+/).filter(Boolean)
          const commArr = communityCards ? communityCards.split(/[\s,]+/).filter(Boolean) : []
          response = await pokerOdds(holeArr, commArr, numOpponents)
          break
        case 'strategy':
          if (!situation.trim()) return
          response = await pokerStrategy(situation)
          break
        case 'analysis':
          if (!hand.trim()) return
          response = await pokerHandAnalysis(hand)
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

  const cardExamples = {
    hole: ['As Kh', 'Qd Qc', '7s 7h', 'Jc Tc'],
    community: ['Ah Kd 7c', 'Qs Jh Td 2c', '9s 8s 7d 6c 5h']
  }

  const situationExamples = [
    'UTG raises 3bb, I have AKs in the BB',
    'Button 3-bets my CO open, I have JJ',
    'SB vs BB, I have 20bb and ATs',
    'Final table bubble with 15bb'
  ]

  const handExamples = [
    'Preflop: I raise AKo from MP, BB calls. Flop Kh 7d 2c. BB checks, I bet half pot, BB raises.',
    'Tournament hand: 25bb effective. UTG opens, I 3bet with QQ from BTN, UTG 4bets.'
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="p-3 bg-red-100 rounded-xl">
          <Spade className="w-6 h-6 text-red-600" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Poker Analyzer</h2>
          <p className="text-gray-600">Calculate odds, get strategy advice, analyze hands</p>
        </div>
      </div>

      {/* Mode selector */}
      <div className="flex gap-2 flex-wrap">
        {modes.map((m) => (
          <button
            key={m.id}
            onClick={() => setMode(m.id)}
            className={`px-4 py-2 rounded-lg font-medium flex items-center gap-2 transition-colors ${
              mode === m.id
                ? 'bg-red-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <m.icon className="w-4 h-4" />
            {m.label}
          </button>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {mode === 'odds' && (
          <>
            <div className="card">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Hole Cards (e.g., "As Kh")
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={holeCards}
                  onChange={(e) => setHoleCards(e.target.value)}
                  placeholder="As Kh"
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
              <div className="mt-2 flex flex-wrap gap-2">
                {cardExamples.hole.map((ex) => (
                  <button
                    key={ex}
                    type="button"
                    onClick={() => setHoleCards(ex)}
                    className="px-2 py-1 bg-gray-100 rounded text-sm font-mono hover:bg-gray-200"
                  >
                    {ex}
                  </button>
                ))}
              </div>
            </div>

            <div className="card">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Community Cards (optional)
              </label>
              <input
                type="text"
                value={communityCards}
                onChange={(e) => setCommunityCards(e.target.value)}
                placeholder="Ah Kd 7c"
                className="input-field font-mono"
              />
              <div className="mt-2 flex flex-wrap gap-2">
                {cardExamples.community.map((ex) => (
                  <button
                    key={ex}
                    type="button"
                    onClick={() => setCommunityCards(ex)}
                    className="px-2 py-1 bg-gray-100 rounded text-sm font-mono hover:bg-gray-200"
                  >
                    {ex}
                  </button>
                ))}
              </div>
            </div>

            <div className="card">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Number of Opponents
              </label>
              <input
                type="number"
                min="1"
                max="9"
                value={numOpponents}
                onChange={(e) => setNumOpponents(parseInt(e.target.value) || 1)}
                className="input-field w-24"
              />
            </div>
          </>
        )}

        {mode === 'strategy' && (
          <div className="card">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Describe the Situation
            </label>
            <div className="relative">
              <textarea
                value={situation}
                onChange={(e) => setSituation(e.target.value)}
                placeholder="Describe the poker situation..."
                rows={4}
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
            <div className="mt-3">
              <p className="text-xs text-gray-500 mb-2">Examples:</p>
              <div className="space-y-1">
                {situationExamples.map((ex) => (
                  <button
                    key={ex}
                    type="button"
                    onClick={() => setSituation(ex)}
                    className="block w-full text-left px-2 py-1 bg-gray-100 rounded text-sm hover:bg-gray-200"
                  >
                    {ex}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {mode === 'analysis' && (
          <div className="card">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Hand History
            </label>
            <div className="relative">
              <textarea
                value={hand}
                onChange={(e) => setHand(e.target.value)}
                placeholder="Describe the hand..."
                rows={6}
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
            <div className="mt-3">
              <p className="text-xs text-gray-500 mb-2">Examples:</p>
              <div className="space-y-1">
                {handExamples.map((ex) => (
                  <button
                    key={ex}
                    type="button"
                    onClick={() => setHand(ex)}
                    className="block w-full text-left px-2 py-1 bg-gray-100 rounded text-xs hover:bg-gray-200"
                  >
                    {ex.substring(0, 80)}...
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        <button
          type="submit"
          disabled={loading || (
            mode === 'odds' ? !holeCards.trim() :
            mode === 'strategy' ? !situation.trim() :
            !hand.trim()
          )}
          className="btn-primary flex items-center gap-2"
        >
          {loading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Play className="w-4 h-4" />
          )}
          {loading ? 'Analyzing...' : 'Analyze'}
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

          {mode === 'odds' && result.win_probability !== undefined && (
            <div className="grid grid-cols-3 gap-4">
              <div className="p-4 bg-green-50 rounded-lg text-center">
                <p className="text-green-700 font-medium text-sm">Win</p>
                <p className="text-2xl font-bold text-green-600">
                  {(result.win_probability * 100).toFixed(1)}%
                </p>
              </div>
              <div className="p-4 bg-yellow-50 rounded-lg text-center">
                <p className="text-yellow-700 font-medium text-sm">Tie</p>
                <p className="text-2xl font-bold text-yellow-600">
                  {(result.tie_probability * 100).toFixed(1)}%
                </p>
              </div>
              <div className="p-4 bg-red-50 rounded-lg text-center">
                <p className="text-red-700 font-medium text-sm">Lose</p>
                <p className="text-2xl font-bold text-red-600">
                  {(result.lose_probability * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          )}

          {(result.strategy || result.analysis || result.advice) && (
            <div className="result-box result-success whitespace-pre-wrap">
              {result.strategy || result.analysis || result.advice}
            </div>
          )}

          {result.recommendation && (
            <div className="p-4 bg-blue-50 rounded-lg">
              <p className="text-blue-700 font-medium mb-1">Recommendation</p>
              <p className="text-blue-900">{result.recommendation}</p>
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

export default PokerAnalyzer
