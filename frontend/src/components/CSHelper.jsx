import React, { useState, useCallback } from 'react'
import { Code, Play, Loader2, AlertCircle, BookOpen, Bug, Gauge, FileSearch } from 'lucide-react'
import Editor from '@monaco-editor/react'
import { csExplain, csDebug, csComplexity, csReview } from '../services/api'
import { useVoiceInput } from '../hooks/useVoiceInput'
import { MicButton } from './MicButton'
import { VoiceFeedback } from './VoiceFeedback'

function CSHelper() {
  const [mode, setMode] = useState('explain')
  const [topic, setTopic] = useState('')
  const [code, setCode] = useState('')
  const [language, setLanguage] = useState('python')
  const [errorMsg, setErrorMsg] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [duration, setDuration] = useState(null)

  // Voice input handler
  const handleVoiceResult = useCallback((parsed) => {
    if (!parsed.engine || parsed.engine === 'cs') {
      setTopic(parsed.query)
    }
  }, [])

  const voice = useVoiceInput({ onResult: handleVoiceResult })

  const modes = [
    { id: 'explain', label: 'Explain', icon: BookOpen },
    { id: 'debug', label: 'Debug', icon: Bug },
    { id: 'complexity', label: 'Complexity', icon: Gauge },
    { id: 'review', label: 'Review', icon: FileSearch }
  ]

  const languages = ['python', 'javascript', 'typescript', 'java', 'cpp', 'go', 'rust']

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      let response
      switch (mode) {
        case 'explain':
          if (!topic.trim()) return
          response = await csExplain(topic)
          break
        case 'debug':
          if (!code.trim()) return
          response = await csDebug(code, language, errorMsg || null)
          break
        case 'complexity':
          if (!code.trim()) return
          response = await csComplexity(code, language)
          break
        case 'review':
          if (!code.trim()) return
          response = await csReview(code, language)
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

  const topicExamples = [
    'Binary search',
    'Hash tables',
    'Recursion',
    'Big O notation',
    'Dynamic programming',
    'Graph traversal'
  ]

  const codeExamples = {
    python: `def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)`,
    javascript: `function quickSort(arr) {
  if (arr.length <= 1) return arr;
  const pivot = arr[0];
  const left = arr.slice(1).filter(x => x < pivot);
  const right = arr.slice(1).filter(x => x >= pivot);
  return [...quickSort(left), pivot, ...quickSort(right)];
}`
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="p-3 bg-green-100 rounded-xl">
          <Code className="w-6 h-6 text-green-600" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900">CS Helper</h2>
          <p className="text-gray-600">Explain concepts, debug code, analyze complexity</p>
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
                ? 'bg-green-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <m.icon className="w-4 h-4" />
            {m.label}
          </button>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {mode === 'explain' ? (
          <div className="card">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Topic to Explain
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="e.g., Binary search trees"
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
              {topicExamples.map((ex) => (
                <button
                  key={ex}
                  type="button"
                  onClick={() => setTopic(ex)}
                  className="px-2 py-1 bg-gray-100 rounded text-sm hover:bg-gray-200 transition-colors"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            <div className="card">
              <div className="flex items-center justify-between mb-2">
                <label className="block text-sm font-medium text-gray-700">
                  Code
                </label>
                <select
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                  className="text-sm border border-gray-300 rounded-lg px-2 py-1"
                >
                  {languages.map((lang) => (
                    <option key={lang} value={lang}>
                      {lang}
                    </option>
                  ))}
                </select>
              </div>
              <div className="monaco-editor-container h-64">
                <Editor
                  height="100%"
                  language={language}
                  value={code}
                  onChange={(value) => setCode(value || '')}
                  theme="vs-light"
                  options={{
                    minimap: { enabled: false },
                    fontSize: 14,
                    scrollBeyondLastLine: false,
                    lineNumbers: 'on',
                    automaticLayout: true
                  }}
                />
              </div>
              <div className="mt-2 flex gap-2">
                {Object.entries(codeExamples).map(([lang, ex]) => (
                  <button
                    key={lang}
                    type="button"
                    onClick={() => {
                      setCode(ex)
                      setLanguage(lang)
                    }}
                    className="px-2 py-1 bg-gray-100 rounded text-xs hover:bg-gray-200 transition-colors"
                  >
                    Example ({lang})
                  </button>
                ))}
              </div>
            </div>

            {mode === 'debug' && (
              <div className="card">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Error Message (optional)
                </label>
                <input
                  type="text"
                  value={errorMsg}
                  onChange={(e) => setErrorMsg(e.target.value)}
                  placeholder="Paste error message here..."
                  className="input-field font-mono text-sm"
                />
              </div>
            )}
          </>
        )}

        <button
          type="submit"
          disabled={loading || (mode === 'explain' ? !topic.trim() : !code.trim())}
          className="btn-primary flex items-center gap-2"
        >
          {loading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Play className="w-4 h-4" />
          )}
          {loading ? 'Processing...' : 'Submit'}
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

          <div className="result-box result-success whitespace-pre-wrap">
            {result.explanation || result.analysis || result.review || JSON.stringify(result, null, 2)}
          </div>

          {result.time_complexity && (
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="p-3 bg-blue-50 rounded-lg">
                <p className="text-blue-700 font-medium">Time Complexity</p>
                <p className="font-mono">{result.time_complexity}</p>
              </div>
              <div className="p-3 bg-purple-50 rounded-lg">
                <p className="text-purple-700 font-medium">Space Complexity</p>
                <p className="font-mono">{result.space_complexity}</p>
              </div>
            </div>
          )}

          {result.suggestions && result.suggestions.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-2">Suggestions</h4>
              <ul className="space-y-1">
                {result.suggestions.map((s, i) => (
                  <li key={i} className="text-sm text-gray-600 flex items-start gap-2">
                    <span className="text-green-500">•</span>
                    {s}
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

export default CSHelper
