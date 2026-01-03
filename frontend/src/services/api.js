import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000, // 60s for complex AI queries
  headers: {
    'Content-Type': 'application/json'
  }
})

// Response interceptor for timing
api.interceptors.response.use(
  (response) => {
    response.durationMs = response.headers['x-duration-ms'] || null
    return response
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Health check
export const healthCheck = () => api.get('/health')

// Math endpoints
export const mathSolve = (expression) =>
  api.post('/math/solve', { expression })

export const mathCalculus = (expression, operation, variable = 'x') =>
  api.post('/math/calculus', { expression, operation, variable })

export const mathAlgebra = (equation, variable = 'x') =>
  api.post('/math/algebra', { equation, variable })

// Vision endpoints
export const visionAnalyze = (imageBase64, prompt) =>
  api.post('/vision/analyze', { image_base64: imageBase64, prompt })

// CS endpoints
export const csExplain = (topic) =>
  api.post('/cs/explain', { topic })

export const csDebug = (code, language, error = null) =>
  api.post('/cs/debug', { code, language, error })

export const csComplexity = (code, language) =>
  api.post('/cs/complexity', { code, language })

export const csReview = (code, language) =>
  api.post('/cs/review', { code, language })

// Poker endpoints
export const pokerOdds = (holeCards, communityCards = [], numOpponents = 1) =>
  api.post('/poker/odds', {
    hole_cards: holeCards,
    community_cards: communityCards,
    num_opponents: numOpponents
  })

export const pokerStrategy = (situation) =>
  api.post('/poker/strategy', { situation })

export const pokerHandAnalysis = (hand) =>
  api.post('/poker/hand-analysis', { hand })

// Chemistry endpoints
export const chemistryBalance = (equation) =>
  api.post('/chemistry/balance', { equation })

export const chemistryMolecularWeight = (formula) =>
  api.post('/chemistry/molecular-weight', { formula })

export const chemistryMolarity = (soluteMoles, volumeLiters) =>
  api.post('/chemistry/molarity', {
    solute_moles: soluteMoles,
    volume_liters: volumeLiters
  })

export const chemistryStoichiometry = (problem) =>
  api.post('/chemistry/stoichiometry', { problem })

export const chemistrySolve = (problem) =>
  api.post('/chemistry/solve', { problem })

// Biology endpoints
export const biologyPunnett = (parent1, parent2) =>
  api.post('/biology/punnett', { parent1, parent2 })

export const biologyProbability = (targetGenotype, parent1, parent2) =>
  api.post('/biology/probability', {
    target_genotype: targetGenotype,
    parent1,
    parent2
  })

export const biologyConcept = (concept) =>
  api.post('/biology/concept', { concept })

export const biologyGenetics = (problem) =>
  api.post('/biology/genetics', { problem })

// Statistics endpoints
export const statsDescriptive = (data) =>
  api.post('/statistics/descriptive', { data })

export const statsHypothesis = (testType, testData) =>
  api.post('/statistics/hypothesis', {
    test_type: testType,
    data: testData
  })

export const statsCorrelation = (x, y) =>
  api.post('/statistics/correlation', { x, y })

export const statsRegression = (x, y) =>
  api.post('/statistics/regression', { x, y })

export const statsProbability = (problem) =>
  api.post('/statistics/probability', { problem })

export const statsConcept = (concept) =>
  api.post('/statistics/concept', { concept })

// ==================== Quant Finance Endpoints ====================

// Mental Math
export const quantMentalMathGenerate = (problemType = null, difficulty = 2) =>
  api.post('/quant/mental-math/generate', {
    problem_type: problemType,
    difficulty
  })

export const quantMentalMathCheck = (problemId, userAnswer, timeMs) =>
  api.post('/quant/mental-math/check', {
    problem_id: problemId,
    user_answer: userAnswer,
    time_ms: timeMs
  })

export const quantMentalMathTypes = () =>
  api.get('/quant/mental-math/types')

// Probability
export const quantProbabilityGenerate = (problemType = null, difficulty = 2) =>
  api.post('/quant/probability/generate', {
    problem_type: problemType,
    difficulty
  })

export const quantProbabilityCheck = (problemId, userAnswer) =>
  api.post('/quant/probability/check', {
    problem_id: problemId,
    user_answer: userAnswer
  })

export const quantProbabilitySimulations = () =>
  api.get('/quant/probability/simulations')

export const quantProbabilityMontyHall = (iterations = 10000) =>
  api.post('/quant/probability/monty-hall', { iterations })

export const quantProbabilityBirthday = (nPeople = 23) =>
  api.post('/quant/probability/birthday', { n_people: nPeople })

// Options Pricing
export const quantOptionsBlackScholes = (S, K, r, sigma, T, optionType = 'call') =>
  api.post('/quant/options/black-scholes', {
    S, K, r, sigma, T,
    option_type: optionType
  })

export const quantOptionsGreeks = (S, K, r, sigma, T, optionType = 'call') =>
  api.post('/quant/options/greeks', {
    S, K, r, sigma, T,
    option_type: optionType
  })

export const quantOptionsImpliedVol = (marketPrice, S, K, r, T, optionType = 'call') =>
  api.post('/quant/options/implied-volatility', {
    market_price: marketPrice,
    S, K, r, T,
    option_type: optionType
  })

export const quantOptionsParity = (callPrice, putPrice, S, K, r, T) =>
  api.post('/quant/options/parity-check', {
    call_price: callPrice,
    put_price: putPrice,
    S, K, r, T
  })

export const quantOptionsChain = (S, strikes, r, sigma, T) =>
  api.post('/quant/options/chain', {
    S, strikes, r, sigma, T
  })

export const quantOptionsFormulas = () =>
  api.get('/quant/options/formulas')

// Market Making
export const quantMarketScenario = (scenarioType = null, difficulty = 2) =>
  api.post('/quant/market-making/scenario', {
    scenario_type: scenarioType,
    difficulty
  })

export const quantMarketCheck = (problemId, userAnswer) =>
  api.post('/quant/market-making/check', {
    problem_id: problemId,
    user_answer: userAnswer
  })

export const quantMarketEdge = (probWin, payoutWin, payoutLose) =>
  api.post('/quant/market-making/edge', {
    prob_win: probWin,
    payout_win: payoutWin,
    payout_lose: payoutLose
  })

export const quantMarketKelly = (probWin, odds, bankroll = 10000) =>
  api.post('/quant/market-making/kelly', {
    prob_win: probWin,
    odds,
    bankroll
  })

export const quantMarketSharpe = (returns, riskFreeRate = 0.02, periodsPerYear = 252) =>
  api.post('/quant/market-making/sharpe', {
    returns,
    risk_free_rate: riskFreeRate,
    periods_per_year: periodsPerYear
  })

export const quantMarketSortino = (returns, riskFreeRate = 0.02, periodsPerYear = 252) =>
  api.post('/quant/market-making/sortino', {
    returns,
    risk_free_rate: riskFreeRate,
    periods_per_year: periodsPerYear
  })

export const quantMarketFormulas = () =>
  api.get('/quant/market-making/formulas')

// Fermi Estimation
export const quantFermiGenerate = (category = null) =>
  api.post('/quant/fermi/generate', { category })

export const quantFermiHint = (problemId, hintLevel = 1) =>
  api.get(`/quant/fermi/hint/${problemId}`, { params: { hint_level: hintLevel } })

export const quantFermiEvaluate = (problemId, estimate) =>
  api.post('/quant/fermi/evaluate', {
    problem_id: problemId,
    estimate
  })

export const quantFermiCategories = () =>
  api.get('/quant/fermi/categories')

// Interview Mode
export const quantInterviewStart = (durationMin = 30, firmStyle = 'general', difficulty = 2) =>
  api.post('/quant/interview/start', {
    duration_min: durationMin,
    firm_style: firmStyle,
    difficulty
  })

export const quantInterviewNext = (sessionId, prevAnswer = null, prevTimeMs = null) =>
  api.post('/quant/interview/next', {
    session_id: sessionId,
    prev_answer: prevAnswer,
    prev_time_ms: prevTimeMs
  })

export const quantInterviewEnd = (sessionId) =>
  api.post('/quant/interview/end', { session_id: sessionId })

export const quantInterviewFirms = () =>
  api.get('/quant/interview/firms')

// Quant Progress (aggregate stats)
export const quantProgress = () =>
  api.get('/quant/progress')

// History endpoints
export const getHistory = (limit = 50, engine = null) => {
  const params = { limit }
  if (engine) params.engine = engine
  return api.get('/history', { params })
}

export const getStats = () => api.get('/stats')

export const clearHistory = () => api.delete('/history')

// Cache endpoints
export const getCacheStats = () => api.get('/cache/stats')

export const clearCache = () => api.delete('/cache')

// ==================== LeetCode Endpoints ====================

export const leetcodeGetProblems = (params = {}) =>
  api.get('/leetcode/problems', { params })

export const leetcodeGetProblem = (slug) =>
  api.get(`/leetcode/problem/${slug}`)

export const leetcodeGetHint = (problemId, level) =>
  api.get(`/leetcode/hints/${problemId}/${level}`)

export const leetcodeSaveSolution = (problemId, code, language, passed = false, timeTakenMs = null, hintsUsed = 0) =>
  api.post('/leetcode/solution', {
    problem_id: problemId,
    code,
    language,
    passed,
    time_taken_ms: timeTakenMs,
    hints_used: hintsUsed
  })

export const leetcodeGetSolutions = (problemId) =>
  api.get(`/leetcode/solutions/${problemId}`)

export const leetcodeGetProgress = () =>
  api.get('/leetcode/progress')

// ==================== Flashcard Endpoints ====================

const REVIEW_QUEUE_KEY = 'flashcard_review_queue'

export const flashcardGetDue = (limit = 20, sourceType = null) => {
  const params = { limit }
  if (sourceType) params.source_type = sourceType
  return api.get('/flashcards/due', { params })
}

export const flashcardGetAll = (limit = 100, offset = 0, sourceType = null) => {
  const params = { limit, offset }
  if (sourceType) params.source_type = sourceType
  return api.get('/flashcards/all', { params })
}

export const flashcardCreate = (front, back, sourceType = 'manual', sourceId = null, tags = null) =>
  api.post('/flashcards', {
    front,
    back,
    source_type: sourceType,
    source_id: sourceId,
    tags
  })

export const flashcardReview = async (cardId, quality, timeMs = null) => {
  try {
    const response = await api.post('/flashcards/review', {
      card_id: cardId,
      quality,
      time_ms: timeMs
    })
    return response
  } catch (error) {
    // Queue for later sync if offline
    if (!navigator.onLine) {
      const queue = JSON.parse(localStorage.getItem(REVIEW_QUEUE_KEY) || '[]')
      queue.push({ card_id: cardId, quality, time_ms: timeMs, reviewed_at: Date.now() })
      localStorage.setItem(REVIEW_QUEUE_KEY, JSON.stringify(queue))
      return { data: { queued: true, card_id: cardId } }
    }
    throw error
  }
}

export const flashcardGenerate = (sourceType, problemData) =>
  api.post('/flashcards/generate', {
    source_type: sourceType,
    problem_data: problemData
  })

export const flashcardSync = (reviews) =>
  api.post('/flashcards/sync', { reviews })

export const flashcardDelete = (cardId) =>
  api.delete(`/flashcards/${cardId}`)

export const flashcardGetStats = () =>
  api.get('/flashcards/stats')

export const flashcardGetHeatmap = (year = null) => {
  const params = year ? { year } : {}
  return api.get('/flashcards/heatmap', { params })
}

export const flashcardGetForecast = (days = 7) =>
  api.get('/flashcards/forecast', { params: { days } })

// Sync offline reviews when back online
export const syncOfflineReviews = async () => {
  const queue = JSON.parse(localStorage.getItem(REVIEW_QUEUE_KEY) || '[]')
  if (queue.length === 0) return { synced: 0 }

  try {
    const result = await flashcardSync(queue)
    localStorage.removeItem(REVIEW_QUEUE_KEY)
    return result.data
  } catch (error) {
    console.error('Failed to sync offline reviews:', error)
    return { error: error.message }
  }
}

// Auto-sync when coming back online
if (typeof window !== 'undefined') {
  window.addEventListener('online', syncOfflineReviews)
}

export default api
