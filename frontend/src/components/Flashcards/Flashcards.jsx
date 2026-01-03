/**
 * Flashcards - SM-2 Spaced Repetition Review Interface
 *
 * Features:
 * - Card flip animation
 * - Keyboard shortcuts (Space=flip, 1-4=rating)
 * - GitHub-style activity heatmap
 * - Offline support with sync
 */

import { useState, useEffect, useCallback, useMemo } from 'react'
import {
  BookOpen, RefreshCw, Check, X, ChevronRight, Calendar,
  Loader2, AlertCircle, Flame, Target, Brain
} from 'lucide-react'
import LatexRenderer from '../shared/LatexRenderer'
import {
  flashcardGetDue,
  flashcardReview,
  flashcardGetStats,
  flashcardGetHeatmap,
  syncOfflineReviews
} from '../../services/api'

// Quality ratings for SM-2
const QUALITY_RATINGS = [
  { value: 0, label: 'Again', shortcut: '1', color: 'bg-red-500 hover:bg-red-600', desc: '<1 min' },
  { value: 2, label: 'Hard', shortcut: '2', color: 'bg-orange-500 hover:bg-orange-600', desc: '10 min' },
  { value: 4, label: 'Good', shortcut: '3', color: 'bg-green-500 hover:bg-green-600', desc: '1 day' },
  { value: 5, label: 'Easy', shortcut: '4', color: 'bg-emerald-600 hover:bg-emerald-700', desc: '4 days' },
]

function Flashcards() {
  const [dueCards, setDueCards] = useState([])
  const [currentIndex, setCurrentIndex] = useState(0)
  const [showAnswer, setShowAnswer] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [stats, setStats] = useState(null)
  const [heatmapData, setHeatmapData] = useState([])
  const [reviewStartTime, setReviewStartTime] = useState(null)
  const [sessionStats, setSessionStats] = useState({ reviewed: 0, correct: 0 })

  // Load due cards and stats
  useEffect(() => {
    loadData()
    syncOfflineReviews() // Sync any pending offline reviews
  }, [])

  const loadData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [cardsRes, statsRes, heatmapRes] = await Promise.all([
        flashcardGetDue(50),
        flashcardGetStats(),
        flashcardGetHeatmap()
      ])
      setDueCards(cardsRes.data.cards || [])
      setStats(statsRes.data)
      setHeatmapData(heatmapRes.data.data || [])
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to load flashcards')
    } finally {
      setLoading(false)
    }
  }

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Prevent if typing in input
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return

      if (e.code === 'Space') {
        e.preventDefault()
        if (!showAnswer) {
          setShowAnswer(true)
          setReviewStartTime(Date.now())
        }
      }

      if (showAnswer && ['1', '2', '3', '4'].includes(e.key)) {
        const rating = QUALITY_RATINGS[parseInt(e.key) - 1]
        if (rating) handleRate(rating.value)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [showAnswer, currentIndex, dueCards])

  const handleFlip = useCallback(() => {
    if (!showAnswer) {
      setShowAnswer(true)
      setReviewStartTime(Date.now())
    }
  }, [showAnswer])

  const handleRate = useCallback(async (quality) => {
    const card = dueCards[currentIndex]
    if (!card) return

    const timeMs = reviewStartTime ? Date.now() - reviewStartTime : null

    try {
      await flashcardReview(card.id, quality, timeMs)
      setSessionStats(prev => ({
        reviewed: prev.reviewed + 1,
        correct: quality >= 3 ? prev.correct + 1 : prev.correct
      }))
    } catch (err) {
      // Error is handled in API (queued for offline)
      console.error('Review error:', err)
    }

    // Move to next card
    setShowAnswer(false)
    setReviewStartTime(null)
    setCurrentIndex(i => i + 1)
  }, [currentIndex, dueCards, reviewStartTime])

  const currentCard = dueCards[currentIndex]
  const isComplete = currentIndex >= dueCards.length

  if (loading) {
    return (
      <div className="p-6 flex items-center justify-center min-h-[400px]">
        <Loader2 className="w-8 h-8 animate-spin text-violet-600" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-6">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-500 mt-0.5" />
          <div>
            <p className="font-medium text-red-800">Error loading flashcards</p>
            <p className="text-sm text-red-600">{error}</p>
            <button onClick={loadData} className="mt-2 text-sm text-red-700 hover:underline">
              Try again
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <BookOpen className="w-8 h-8 text-violet-600" />
          <div>
            <h1 className="text-2xl font-bold text-gray-800">Flashcard Review</h1>
            <p className="text-gray-500">
              {isComplete ? 'All done!' : `${dueCards.length - currentIndex} cards remaining`}
            </p>
          </div>
        </div>
        <button
          onClick={loadData}
          className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
          title="Refresh"
        >
          <RefreshCw className="w-5 h-5 text-gray-600" />
        </button>
      </div>

      {/* Stats Row */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <StatCard icon={Target} label="Due Today" value={stats.due_today} color="text-violet-600" />
          <StatCard icon={Brain} label="Mastered" value={stats.mastered} color="text-green-600" />
          <StatCard icon={Flame} label="Retention" value={`${stats.retention_rate}%`} color="text-orange-500" />
          <StatCard icon={Check} label="This Session" value={`${sessionStats.correct}/${sessionStats.reviewed}`} color="text-blue-600" />
        </div>
      )}

      {/* Main Card Area */}
      {isComplete ? (
        <CompletionScreen stats={sessionStats} onReload={loadData} />
      ) : currentCard ? (
        <div className="space-y-6">
          {/* Flashcard */}
          <div
            className="flashcard-container perspective-1000 cursor-pointer"
            onClick={handleFlip}
          >
            <div className={`flashcard ${showAnswer ? 'flipped' : ''}`}>
              {/* Front */}
              <div className="flashcard-face flashcard-front bg-white border-2 border-gray-200">
                <div className="text-center p-6">
                  <p className="text-sm text-gray-400 mb-4">QUESTION</p>
                  <div className="text-lg text-gray-800">
                    <LatexRenderer content={currentCard.front} />
                  </div>
                  {!showAnswer && (
                    <p className="mt-6 text-sm text-gray-400">
                      Click or press <kbd className="px-1.5 py-0.5 bg-gray-100 rounded text-xs">Space</kbd> to reveal
                    </p>
                  )}
                </div>
              </div>

              {/* Back */}
              <div className="flashcard-face flashcard-back bg-gradient-to-br from-violet-50 to-indigo-100 border-2 border-violet-200">
                <div className="text-center p-6">
                  <p className="text-sm text-violet-500 mb-4">ANSWER</p>
                  <div className="text-lg text-gray-800">
                    <LatexRenderer content={currentCard.back} />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Rating Buttons (show after flip) */}
          {showAnswer && (
            <div className="space-y-4">
              <p className="text-center text-gray-600">How well did you know this?</p>
              <div className="flex justify-center gap-3">
                {QUALITY_RATINGS.map((rating) => (
                  <button
                    key={rating.value}
                    onClick={() => handleRate(rating.value)}
                    className={`rating-btn ${rating.color} text-white px-6 py-3 rounded-lg font-medium transition-all`}
                  >
                    <span className="block">{rating.label}</span>
                    <span className="text-xs opacity-75">({rating.shortcut}) {rating.desc}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Progress */}
          <div className="flex items-center gap-4">
            <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-violet-500 transition-all duration-300"
                style={{ width: `${((currentIndex + 1) / dueCards.length) * 100}%` }}
              />
            </div>
            <span className="text-sm text-gray-500">
              {currentIndex + 1} / {dueCards.length}
            </span>
          </div>

          {/* Card metadata */}
          <div className="flex items-center justify-between text-sm text-gray-400">
            <span>Source: {currentCard.source_type}</span>
            {currentCard.tags && currentCard.tags.length > 0 && (
              <div className="flex gap-1">
                {(Array.isArray(currentCard.tags) ? currentCard.tags : []).map((tag, i) => (
                  <span key={i} className="px-2 py-0.5 bg-gray-100 rounded-full text-xs">
                    {tag}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      ) : (
        <EmptyState onReload={loadData} />
      )}

      {/* Heatmap */}
      <div className="mt-8">
        <div className="flex items-center gap-2 mb-4">
          <Calendar className="w-5 h-5 text-gray-600" />
          <h2 className="text-lg font-semibold text-gray-800">Review Activity</h2>
        </div>
        <Heatmap data={heatmapData} />
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

// Completion Screen
function CompletionScreen({ stats, onReload }) {
  const percentage = stats.reviewed > 0
    ? Math.round((stats.correct / stats.reviewed) * 100)
    : 0

  return (
    <div className="text-center py-12">
      <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
        <Check className="w-10 h-10 text-green-600" />
      </div>
      <h2 className="text-2xl font-bold text-gray-800 mb-2">Session Complete!</h2>
      <p className="text-gray-600 mb-6">
        You reviewed {stats.reviewed} cards with {percentage}% accuracy
      </p>
      <button
        onClick={onReload}
        className="btn-primary inline-flex items-center gap-2"
      >
        <RefreshCw className="w-4 h-4" />
        Load More Cards
      </button>
    </div>
  )
}

// Empty State
function EmptyState({ onReload }) {
  return (
    <div className="text-center py-12">
      <BookOpen className="w-16 h-16 text-gray-300 mx-auto mb-4" />
      <h2 className="text-xl font-semibold text-gray-600 mb-2">No cards due for review</h2>
      <p className="text-gray-500 mb-6">
        Great job! Come back later or add new cards from your practice sessions.
      </p>
      <button
        onClick={onReload}
        className="btn-secondary inline-flex items-center gap-2"
      >
        <RefreshCw className="w-4 h-4" />
        Refresh
      </button>
    </div>
  )
}

// Heatmap Component
function Heatmap({ data }) {
  const grid = useMemo(() => {
    const year = new Date().getFullYear()
    const startDate = new Date(year, 0, 1)
    const endDate = new Date(year, 11, 31)

    // Create lookup map
    const dataMap = new Map(
      data.map(d => [d.date, d.flashcard_reviews + (d.leetcode_solved || 0)])
    )

    const weeks = []
    let currentWeek = []
    let currentDate = new Date(startDate)

    // Fill empty days at start
    const startDay = startDate.getDay()
    for (let i = 0; i < startDay; i++) {
      currentWeek.push(null)
    }

    while (currentDate <= endDate) {
      const dateStr = currentDate.toISOString().split('T')[0]
      const count = dataMap.get(dateStr) || 0

      currentWeek.push({
        date: dateStr,
        count,
        level: getLevel(count)
      })

      if (currentDate.getDay() === 6) {
        weeks.push(currentWeek)
        currentWeek = []
      }

      currentDate.setDate(currentDate.getDate() + 1)
    }

    if (currentWeek.length > 0) {
      weeks.push(currentWeek)
    }

    return weeks
  }, [data])

  const colors = ['#ebedf0', '#9be9a8', '#40c463', '#30a14e', '#216e39']

  return (
    <div className="overflow-x-auto">
      <div className="flex gap-0.5" style={{ minWidth: 'max-content' }}>
        {grid.map((week, weekIndex) => (
          <div key={weekIndex} className="flex flex-col gap-0.5">
            {week.map((day, dayIndex) => (
              <div
                key={dayIndex}
                className="w-3 h-3 rounded-sm"
                style={{ backgroundColor: day ? colors[day.level] : 'transparent' }}
                title={day ? `${day.date}: ${day.count} reviews` : ''}
              />
            ))}
          </div>
        ))}
      </div>
      <div className="flex items-center justify-end gap-1 mt-2 text-xs text-gray-500">
        <span>Less</span>
        {colors.map((color, i) => (
          <div
            key={i}
            className="w-3 h-3 rounded-sm"
            style={{ backgroundColor: color }}
          />
        ))}
        <span>More</span>
      </div>
    </div>
  )
}

function getLevel(count) {
  if (count === 0) return 0
  if (count < 3) return 1
  if (count < 6) return 2
  if (count < 10) return 3
  return 4
}

export default Flashcards
