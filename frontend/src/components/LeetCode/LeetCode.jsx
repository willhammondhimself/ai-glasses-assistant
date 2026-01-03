/**
 * LeetCode - Coding Practice Interface with Jane Street Mode
 *
 * Features:
 * - Problem search and filtering
 * - Jane Street mode (quant-relevant problems)
 * - Monaco code editor with syntax highlighting
 * - Auto-starting timer
 * - Progressive hint system (3 levels)
 * - Test cases for manual verification
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import Editor from '@monaco-editor/react'
import {
  Code, Search, Clock, Lightbulb, Play, CheckCircle,
  XCircle, ChevronRight, Filter, Loader2, AlertCircle,
  RefreshCw, Trophy, Target, Flame, BookOpen
} from 'lucide-react'
import {
  leetcodeGetProblems,
  leetcodeGetProblem,
  leetcodeGetHint,
  leetcodeSaveSolution,
  leetcodeGetProgress
} from '../../services/api'

// Language options for the editor
const LANGUAGES = [
  { value: 'python', label: 'Python 3', monacoLang: 'python' },
  { value: 'javascript', label: 'JavaScript', monacoLang: 'javascript' },
  { value: 'typescript', label: 'TypeScript', monacoLang: 'typescript' },
  { value: 'java', label: 'Java', monacoLang: 'java' },
  { value: 'cpp', label: 'C++', monacoLang: 'cpp' },
  { value: 'go', label: 'Go', monacoLang: 'go' },
]

// Difficulty colors
const DIFFICULTY_COLORS = {
  Easy: 'text-green-600 bg-green-100',
  Medium: 'text-yellow-600 bg-yellow-100',
  Hard: 'text-red-600 bg-red-100',
}

function LeetCode() {
  // View state
  const [view, setView] = useState('browse') // 'browse' | 'solve'

  // Browse state
  const [problems, setProblems] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [difficultyFilter, setDifficultyFilter] = useState('')
  const [janeStreetMode, setJaneStreetMode] = useState(false)
  const [progress, setProgress] = useState(null)

  // Solve state
  const [currentProblem, setCurrentProblem] = useState(null)
  const [problemLoading, setProblemLoading] = useState(false)
  const [language, setLanguage] = useState('python')
  const [code, setCode] = useState('')
  const [timer, setTimer] = useState(0)
  const [timerRunning, setTimerRunning] = useState(false)
  const [hints, setHints] = useState([])
  const [currentHintLevel, setCurrentHintLevel] = useState(-1)
  const [hintLoading, setHintLoading] = useState(false)
  const [testResults, setTestResults] = useState(null)

  const timerRef = useRef(null)
  const editorRef = useRef(null)

  // Load problems and progress
  useEffect(() => {
    loadProblems()
    loadProgress()
  }, [difficultyFilter, janeStreetMode])

  // Timer effect
  useEffect(() => {
    if (timerRunning) {
      timerRef.current = setInterval(() => {
        setTimer(t => t + 1)
      }, 1000)
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [timerRunning])

  const loadProblems = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await leetcodeGetProblems({
        difficulty: difficultyFilter || undefined,
        jane_street_mode: janeStreetMode,
        limit: 50
      })
      setProblems(res.data.problems || [])
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load problems')
    } finally {
      setLoading(false)
    }
  }

  const loadProgress = async () => {
    try {
      const res = await leetcodeGetProgress()
      setProgress(res.data)
    } catch (err) {
      console.error('Failed to load progress:', err)
    }
  }

  const selectProblem = async (slug) => {
    setProblemLoading(true)
    setError(null)
    try {
      const res = await leetcodeGetProblem(slug)
      if (res.data.error) {
        setError(res.data.error)
        return
      }

      setCurrentProblem(res.data)

      // Set initial code from template
      const templates = res.data.code_templates || {}
      const langKey = language === 'cpp' ? 'cpp' : language
      setCode(templates[langKey] || templates['python3'] || templates['python'] || '# Start coding here\n')

      // Reset solve state
      setTimer(0)
      setTimerRunning(true) // Auto-start timer
      setHints([])
      setCurrentHintLevel(-1)
      setTestResults(null)
      setView('solve')
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load problem')
    } finally {
      setProblemLoading(false)
    }
  }

  const handleLanguageChange = (newLang) => {
    setLanguage(newLang)
    if (currentProblem?.code_templates) {
      const langKey = newLang === 'cpp' ? 'cpp' : newLang
      const template = currentProblem.code_templates[langKey] ||
                       currentProblem.code_templates['python3'] ||
                       currentProblem.code_templates['python'] || ''
      setCode(template)
    }
  }

  const requestHint = async () => {
    if (!currentProblem || hintLoading) return
    if (currentHintLevel >= (currentProblem.hints_count || 0) - 1) return

    setHintLoading(true)
    try {
      const nextLevel = currentHintLevel + 1
      const res = await leetcodeGetHint(currentProblem.id, nextLevel)

      if (res.data.error) {
        // No more hints
        return
      }

      setHints(prev => [...prev, res.data.hint])
      setCurrentHintLevel(nextLevel)
    } catch (err) {
      console.error('Failed to get hint:', err)
    } finally {
      setHintLoading(false)
    }
  }

  const runTests = useCallback(() => {
    // Manual test verification - show test cases
    if (!currentProblem?.test_cases?.length) {
      setTestResults({
        status: 'info',
        message: 'No test cases available. Verify your solution manually.',
        cases: []
      })
      return
    }

    setTestResults({
      status: 'pending',
      message: 'Test cases shown below. Run your code locally and compare results.',
      cases: currentProblem.test_cases
    })
  }, [currentProblem])

  const submitSolution = async (passed) => {
    if (!currentProblem) return

    setTimerRunning(false)

    try {
      await leetcodeSaveSolution({
        problem_id: currentProblem.id,
        code,
        language,
        passed,
        time_taken_ms: timer * 1000,
        hints_used: hints.length
      })

      setTestResults({
        status: passed ? 'success' : 'failed',
        message: passed
          ? `Solution submitted! Time: ${formatTime(timer)}, Hints used: ${hints.length}`
          : 'Marked as attempted. Keep practicing!',
        cases: []
      })

      // Refresh progress
      loadProgress()
    } catch (err) {
      console.error('Failed to save solution:', err)
    }
  }

  const backToBrowse = () => {
    setTimerRunning(false)
    if (timerRef.current) clearInterval(timerRef.current)
    setView('browse')
    setCurrentProblem(null)
  }

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  // Filter problems by search query
  const filteredProblems = problems.filter(p =>
    p.title?.toLowerCase().includes(searchQuery.toLowerCase()) ||
    p.slug?.toLowerCase().includes(searchQuery.toLowerCase())
  )

  if (view === 'solve' && currentProblem) {
    return (
      <SolveView
        problem={currentProblem}
        code={code}
        setCode={setCode}
        language={language}
        onLanguageChange={handleLanguageChange}
        timer={timer}
        timerRunning={timerRunning}
        setTimerRunning={setTimerRunning}
        hints={hints}
        currentHintLevel={currentHintLevel}
        hintLoading={hintLoading}
        onRequestHint={requestHint}
        testResults={testResults}
        onRunTests={runTests}
        onSubmit={submitSolution}
        onBack={backToBrowse}
        editorRef={editorRef}
        formatTime={formatTime}
      />
    )
  }

  return (
    <div className="p-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Code className="w-8 h-8 text-orange-600" />
          <div>
            <h1 className="text-2xl font-bold text-gray-800">LeetCode Practice</h1>
            <p className="text-gray-500">Coding interview preparation</p>
          </div>
        </div>
        <button
          onClick={loadProblems}
          className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
          title="Refresh"
        >
          <RefreshCw className="w-5 h-5 text-gray-600" />
        </button>
      </div>

      {/* Progress Stats */}
      {progress && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <StatCard icon={Trophy} label="Total Solved" value={progress.total_solved} color="text-yellow-600" />
          <StatCard icon={Target} label="Easy" value={progress.easy_solved} color="text-green-600" />
          <StatCard icon={Flame} label="Medium" value={progress.medium_solved} color="text-orange-500" />
          <StatCard icon={Code} label="Hard" value={progress.hard_solved} color="text-red-600" />
        </div>
      )}

      {/* Filters */}
      <div className="bg-white rounded-lg border p-4 mb-6">
        <div className="flex flex-wrap gap-4 items-center">
          {/* Search */}
          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search problems..."
              className="w-full pl-10 pr-4 py-2 border rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
            />
          </div>

          {/* Difficulty Filter */}
          <select
            value={difficultyFilter}
            onChange={(e) => setDifficultyFilter(e.target.value)}
            className="px-4 py-2 border rounded-lg focus:ring-2 focus:ring-orange-500"
          >
            <option value="">All Difficulties</option>
            <option value="Easy">Easy</option>
            <option value="Medium">Medium</option>
            <option value="Hard">Hard</option>
          </select>

          {/* Jane Street Mode */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={janeStreetMode}
              onChange={(e) => setJaneStreetMode(e.target.checked)}
              className="w-4 h-4 text-orange-600 rounded focus:ring-orange-500"
            />
            <span className="text-sm font-medium text-gray-700">Jane Street Mode</span>
            <span className="text-xs text-gray-400">(Quant-focused)</span>
          </label>
        </div>

        {janeStreetMode && (
          <p className="mt-3 text-sm text-orange-600 bg-orange-50 p-2 rounded">
            Showing problems relevant for quant interviews: DP, Math, Recursion, Binary Search, Game Theory, Probability
          </p>
        )}
      </div>

      {/* Error State */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-500 mt-0.5" />
          <div>
            <p className="font-medium text-red-800">Error</p>
            <p className="text-sm text-red-600">{error}</p>
          </div>
        </div>
      )}

      {/* Loading State */}
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin text-orange-600" />
        </div>
      ) : (
        /* Problem List */
        <div className="bg-white rounded-lg border overflow-hidden">
          <table className="w-full">
            <thead className="bg-gray-50 border-b">
              <tr>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">Title</th>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">Difficulty</th>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-600 hidden md:table-cell">Topics</th>
                <th className="px-4 py-3 text-right text-sm font-medium text-gray-600">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {filteredProblems.length === 0 ? (
                <tr>
                  <td colSpan={4} className="px-4 py-8 text-center text-gray-500">
                    <BookOpen className="w-12 h-12 text-gray-300 mx-auto mb-2" />
                    <p>No problems found. Try adjusting filters or fetching a specific problem.</p>
                  </td>
                </tr>
              ) : (
                filteredProblems.map((problem) => (
                  <tr key={problem.id} className="hover:bg-gray-50">
                    <td className="px-4 py-3">
                      <div>
                        <p className="font-medium text-gray-800">{problem.title}</p>
                        <p className="text-sm text-gray-400">{problem.slug}</p>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${DIFFICULTY_COLORS[problem.difficulty] || 'text-gray-600 bg-gray-100'}`}>
                        {problem.difficulty}
                      </span>
                    </td>
                    <td className="px-4 py-3 hidden md:table-cell">
                      <div className="flex flex-wrap gap-1">
                        {(problem.topics || []).slice(0, 3).map((topic, i) => (
                          <span key={i} className="px-2 py-0.5 bg-gray-100 rounded text-xs text-gray-600">
                            {topic}
                          </span>
                        ))}
                        {(problem.topics || []).length > 3 && (
                          <span className="text-xs text-gray-400">+{problem.topics.length - 3}</span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-right">
                      <button
                        onClick={() => selectProblem(problem.slug)}
                        disabled={problemLoading}
                        className="inline-flex items-center gap-1 px-3 py-1.5 bg-orange-600 text-white text-sm rounded-lg hover:bg-orange-700 disabled:opacity-50"
                      >
                        {problemLoading ? (
                          <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                          <>
                            Solve <ChevronRight className="w-4 h-4" />
                          </>
                        )}
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Manual Problem Fetch */}
      <div className="mt-6 bg-white rounded-lg border p-4">
        <h3 className="font-medium text-gray-800 mb-2">Fetch Problem by Slug</h3>
        <p className="text-sm text-gray-500 mb-3">
          Enter a LeetCode problem slug (e.g., "two-sum", "merge-k-sorted-lists")
        </p>
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="Enter problem slug..."
            id="problem-slug-input"
            className="flex-1 px-4 py-2 border rounded-lg focus:ring-2 focus:ring-orange-500"
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                selectProblem(e.target.value.toLowerCase().trim())
              }
            }}
          />
          <button
            onClick={() => {
              const input = document.getElementById('problem-slug-input')
              if (input.value) selectProblem(input.value.toLowerCase().trim())
            }}
            className="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700"
          >
            Fetch
          </button>
        </div>
      </div>
    </div>
  )
}

// Stat Card Component
function StatCard({ icon: Icon, label, value, color }) {
  return (
    <div className="bg-white rounded-lg border p-4">
      <div className="flex items-center gap-2 mb-1">
        <Icon className={`w-4 h-4 ${color}`} />
        <span className="text-sm text-gray-500">{label}</span>
      </div>
      <p className={`text-2xl font-bold ${color}`}>{value}</p>
    </div>
  )
}

// Solve View Component
function SolveView({
  problem,
  code,
  setCode,
  language,
  onLanguageChange,
  timer,
  timerRunning,
  setTimerRunning,
  hints,
  currentHintLevel,
  hintLoading,
  onRequestHint,
  testResults,
  onRunTests,
  onSubmit,
  onBack,
  editorRef,
  formatTime
}) {
  const hintsAvailable = problem.hints_count || 0
  const hintsRemaining = hintsAvailable - hints.length

  return (
    <div className="h-screen flex flex-col">
      {/* Top Bar */}
      <div className="bg-white border-b px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={onBack}
            className="text-gray-600 hover:text-gray-800"
          >
            &larr; Back
          </button>
          <div>
            <h2 className="font-bold text-gray-800">{problem.title}</h2>
            <span className={`text-xs px-2 py-0.5 rounded-full ${DIFFICULTY_COLORS[problem.difficulty]}`}>
              {problem.difficulty}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* Timer */}
          <div className="flex items-center gap-2">
            <Clock className={`w-5 h-5 ${timerRunning ? 'text-orange-600' : 'text-gray-400'}`} />
            <span className="font-mono text-lg">{formatTime(timer)}</span>
            <button
              onClick={() => setTimerRunning(!timerRunning)}
              className="text-sm text-gray-500 hover:text-gray-700"
            >
              {timerRunning ? 'Pause' : 'Resume'}
            </button>
          </div>

          {/* Language Selector */}
          <select
            value={language}
            onChange={(e) => onLanguageChange(e.target.value)}
            className="px-3 py-1.5 border rounded-lg text-sm"
          >
            {LANGUAGES.map(lang => (
              <option key={lang.value} value={lang.value}>{lang.label}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel - Problem Description */}
        <div className="w-1/2 border-r overflow-y-auto">
          <div className="p-4">
            {/* Problem Content */}
            <div
              className="prose prose-sm max-w-none"
              dangerouslySetInnerHTML={{ __html: problem.content || '<p>No description available</p>' }}
            />

            {/* Topics */}
            {problem.topics?.length > 0 && (
              <div className="mt-4 pt-4 border-t">
                <h4 className="text-sm font-medium text-gray-600 mb-2">Related Topics</h4>
                <div className="flex flex-wrap gap-1">
                  {problem.topics.map((topic, i) => (
                    <span key={i} className="px-2 py-1 bg-gray-100 rounded text-xs text-gray-600">
                      {topic}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Hints Section */}
            <div className="mt-4 pt-4 border-t">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-medium text-gray-600 flex items-center gap-2">
                  <Lightbulb className="w-4 h-4" />
                  Hints ({hints.length}/{hintsAvailable})
                </h4>
                {hintsRemaining > 0 && (
                  <button
                    onClick={onRequestHint}
                    disabled={hintLoading}
                    className="text-sm text-orange-600 hover:text-orange-700 disabled:opacity-50"
                  >
                    {hintLoading ? 'Loading...' : `Get Hint (${hintsRemaining} left)`}
                  </button>
                )}
              </div>

              {hints.length === 0 ? (
                <p className="text-sm text-gray-400 italic">
                  No hints revealed yet. Try solving first!
                </p>
              ) : (
                <div className="space-y-2">
                  {hints.map((hint, i) => (
                    <div key={i} className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                      <p className="text-sm text-yellow-800">
                        <strong>Hint {i + 1}:</strong> {hint}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Test Cases */}
            {problem.test_cases?.length > 0 && (
              <div className="mt-4 pt-4 border-t">
                <h4 className="text-sm font-medium text-gray-600 mb-2">Test Cases</h4>
                <div className="space-y-2">
                  {problem.test_cases.slice(0, 3).map((tc, i) => (
                    <div key={i} className="p-2 bg-gray-50 rounded text-sm font-mono">
                      <p className="text-gray-600">Input: {tc.input}</p>
                      {tc.expected && <p className="text-gray-600">Expected: {tc.expected}</p>}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Panel - Code Editor */}
        <div className="w-1/2 flex flex-col">
          <div className="flex-1">
            <Editor
              height="100%"
              language={LANGUAGES.find(l => l.value === language)?.monacoLang || 'python'}
              value={code}
              onChange={(value) => setCode(value || '')}
              onMount={(editor) => { editorRef.current = editor }}
              theme="vs-dark"
              options={{
                fontSize: 14,
                minimap: { enabled: false },
                scrollBeyondLastLine: false,
                automaticLayout: true,
                tabSize: 4,
                wordWrap: 'on'
              }}
            />
          </div>

          {/* Actions & Results */}
          <div className="border-t bg-white p-4">
            {/* Action Buttons */}
            <div className="flex items-center gap-3 mb-4">
              <button
                onClick={onRunTests}
                className="flex items-center gap-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200"
              >
                <Play className="w-4 h-4" />
                Show Test Cases
              </button>
              <button
                onClick={() => onSubmit(true)}
                className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
              >
                <CheckCircle className="w-4 h-4" />
                Mark Solved
              </button>
              <button
                onClick={() => onSubmit(false)}
                className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
              >
                <XCircle className="w-4 h-4" />
                Mark Attempted
              </button>
            </div>

            {/* Test Results */}
            {testResults && (
              <div className={`p-3 rounded-lg ${
                testResults.status === 'success' ? 'bg-green-50 border border-green-200' :
                testResults.status === 'failed' ? 'bg-red-50 border border-red-200' :
                'bg-gray-50 border border-gray-200'
              }`}>
                <p className={`font-medium ${
                  testResults.status === 'success' ? 'text-green-800' :
                  testResults.status === 'failed' ? 'text-red-800' :
                  'text-gray-800'
                }`}>
                  {testResults.message}
                </p>

                {testResults.cases?.length > 0 && (
                  <div className="mt-2 space-y-1">
                    {testResults.cases.map((tc, i) => (
                      <div key={i} className="text-sm font-mono text-gray-600">
                        <span className="text-gray-400">Case {i + 1}:</span> {tc.input}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default LeetCode
