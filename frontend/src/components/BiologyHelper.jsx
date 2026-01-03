import React, { useState, useCallback } from 'react'
import { Dna, Play, Loader2, AlertCircle, Grid3X3, Percent, BookOpen, FileQuestion } from 'lucide-react'
import {
  biologyPunnett,
  biologyProbability,
  biologyConcept,
  biologyGenetics
} from '../services/api'
import { useVoiceInput } from '../hooks/useVoiceInput'
import { MicButton } from './MicButton'
import { VoiceFeedback } from './VoiceFeedback'

function BiologyHelper() {
  const [mode, setMode] = useState('punnett')
  const [parent1, setParent1] = useState('')
  const [parent2, setParent2] = useState('')
  const [targetGenotype, setTargetGenotype] = useState('')
  const [concept, setConcept] = useState('')
  const [problem, setProblem] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [duration, setDuration] = useState(null)

  // Voice input handler
  const handleVoiceResult = useCallback((parsed) => {
    if (!parsed.engine || parsed.engine === 'biology') {
      if (mode === 'concept') setConcept(parsed.query)
      else if (mode === 'genetics') setProblem(parsed.query)
      else setParent1(parsed.query)
    }
  }, [mode])

  const voice = useVoiceInput({ onResult: handleVoiceResult })

  const modes = [
    { id: 'punnett', label: 'Punnett Square', icon: Grid3X3 },
    { id: 'probability', label: 'Probability', icon: Percent },
    { id: 'concept', label: 'Explain Concept', icon: BookOpen },
    { id: 'genetics', label: 'Genetics Problem', icon: FileQuestion }
  ]

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      let response
      switch (mode) {
        case 'punnett':
          if (!parent1.trim() || !parent2.trim()) return
          response = await biologyPunnett(parent1, parent2)
          break
        case 'probability':
          if (!targetGenotype.trim() || !parent1.trim() || !parent2.trim()) return
          response = await biologyProbability(targetGenotype, parent1, parent2)
          break
        case 'concept':
          if (!concept.trim()) return
          response = await biologyConcept(concept)
          break
        case 'genetics':
          if (!problem.trim()) return
          response = await biologyGenetics(problem)
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

  const genotypeExamples = [
    { p1: 'Aa', p2: 'Aa', label: 'Monohybrid (Aa × Aa)' },
    { p1: 'Aa', p2: 'aa', label: 'Test cross (Aa × aa)' },
    { p1: 'AaBb', p2: 'AaBb', label: 'Dihybrid (AaBb × AaBb)' },
    { p1: 'AA', p2: 'aa', label: 'F1 cross (AA × aa)' }
  ]

  const conceptExamples = [
    'Mitosis vs Meiosis',
    'DNA replication',
    'Natural selection',
    'Gene expression',
    'Photosynthesis',
    'Cell respiration'
  ]

  const problemExamples = [
    'In pea plants, tall (T) is dominant over short (t). Cross two heterozygous tall plants.',
    'If a colorblind man marries a carrier woman, what are the chances of their children being colorblind?',
    'Blood type A father and type B mother. What blood types are possible for children?'
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="p-3 bg-pink-100 rounded-xl">
          <Dna className="w-6 h-6 text-pink-600" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Biology Helper</h2>
          <p className="text-gray-600">Punnett squares, genetics, concept explanations</p>
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
                ? 'bg-pink-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <m.icon className="w-4 h-4" />
            {m.label}
          </button>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {(mode === 'punnett' || mode === 'probability') && (
          <>
            <div className="card grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Parent 1 Genotype
                </label>
                <input
                  type="text"
                  value={parent1}
                  onChange={(e) => setParent1(e.target.value)}
                  placeholder="Aa"
                  className="input-field font-mono"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Parent 2 Genotype
                </label>
                <input
                  type="text"
                  value={parent2}
                  onChange={(e) => setParent2(e.target.value)}
                  placeholder="Aa"
                  className="input-field font-mono"
                />
              </div>
            </div>

            <div className="card">
              <p className="text-sm text-gray-600 mb-2">Quick examples:</p>
              <div className="flex flex-wrap gap-2">
                {genotypeExamples.map((ex) => (
                  <button
                    key={ex.label}
                    type="button"
                    onClick={() => {
                      setParent1(ex.p1)
                      setParent2(ex.p2)
                    }}
                    className="px-2 py-1 bg-gray-100 rounded text-sm hover:bg-gray-200"
                  >
                    {ex.label}
                  </button>
                ))}
              </div>
            </div>

            {mode === 'probability' && (
              <div className="card">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Target Genotype
                </label>
                <input
                  type="text"
                  value={targetGenotype}
                  onChange={(e) => setTargetGenotype(e.target.value)}
                  placeholder="aa"
                  className="input-field font-mono w-32"
                />
              </div>
            )}
          </>
        )}

        {mode === 'concept' && (
          <div className="card">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Biology Concept
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={concept}
                onChange={(e) => setConcept(e.target.value)}
                placeholder="e.g., Mitosis"
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

        {mode === 'genetics' && (
          <div className="card">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Genetics Problem
            </label>
            <div className="relative">
              <textarea
                value={problem}
                onChange={(e) => setProblem(e.target.value)}
                placeholder="Describe the genetics problem..."
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
            <div className="mt-3 space-y-1">
              {problemExamples.map((ex) => (
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
            (mode === 'punnett' || mode === 'probability') ? (!parent1.trim() || !parent2.trim()) :
            mode === 'concept' ? !concept.trim() :
            !problem.trim()
          ) || (mode === 'probability' && !targetGenotype.trim())}
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

          {/* Punnett Square Grid */}
          {result.grid && (
            <div className="overflow-x-auto">
              <table className="mx-auto border-collapse">
                <thead>
                  <tr>
                    <th className="w-16 h-12"></th>
                    {result.gametes1?.map((g, i) => (
                      <th key={i} className="w-16 h-12 bg-pink-100 border border-pink-200 font-mono text-pink-700">
                        {g}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.grid.map((row, i) => (
                    <tr key={i}>
                      <th className="w-16 h-12 bg-pink-100 border border-pink-200 font-mono text-pink-700">
                        {result.gametes2?.[i]}
                      </th>
                      {row.map((cell, j) => (
                        <td key={j} className="w-16 h-12 border border-gray-200 text-center font-mono bg-white">
                          {cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Genotype ratios */}
          {result.genotype_ratios && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-2">Genotype Ratios</h4>
              <div className="flex flex-wrap gap-2">
                {Object.entries(result.genotype_ratios).map(([genotype, ratio]) => (
                  <span key={genotype} className="px-3 py-1 bg-blue-50 rounded-lg font-mono text-sm">
                    {genotype}: {ratio}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Phenotype ratios */}
          {result.phenotype_ratios && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-2">Phenotype Ratios</h4>
              <div className="flex flex-wrap gap-2">
                {Object.entries(result.phenotype_ratios).map(([phenotype, ratio]) => (
                  <span key={phenotype} className="px-3 py-1 bg-green-50 rounded-lg text-sm">
                    {phenotype}: {ratio}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Probability result */}
          {result.probability !== undefined && (
            <div className="p-4 bg-purple-50 rounded-lg">
              <p className="text-purple-700 font-medium text-sm mb-1">
                Probability of {result.target_genotype || targetGenotype}
              </p>
              <p className="text-3xl font-bold text-purple-900">
                {result.percentage || `${(result.probability * 100).toFixed(1)}%`}
              </p>
              {result.ratio && (
                <p className="text-sm text-purple-600 mt-1">Ratio: {result.ratio}</p>
              )}
            </div>
          )}

          {/* Concept explanation */}
          {result.explanation && (
            <div className="result-box result-success">
              <p className="whitespace-pre-wrap">{result.explanation}</p>
            </div>
          )}

          {result.key_points && result.key_points.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-2">Key Points</h4>
              <ul className="space-y-1">
                {result.key_points.map((point, i) => (
                  <li key={i} className="text-sm text-gray-600 flex items-start gap-2">
                    <span className="text-pink-500">•</span>
                    {point}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {result.example && (
            <div className="p-3 bg-yellow-50 rounded-lg">
              <p className="text-yellow-700 font-medium text-sm mb-1">Example</p>
              <p className="text-sm text-yellow-800">{result.example}</p>
            </div>
          )}

          {result.answer && (
            <div className="result-box result-success whitespace-pre-wrap">
              {result.answer}
            </div>
          )}

          {result.punnett_square && (
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-gray-700 font-medium text-sm mb-1">Punnett Square</p>
              <pre className="text-sm font-mono whitespace-pre-wrap">{result.punnett_square}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default BiologyHelper
