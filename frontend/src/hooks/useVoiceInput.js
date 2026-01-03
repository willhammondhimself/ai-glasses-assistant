import { useState, useEffect, useRef, useCallback } from 'react'

/**
 * useVoiceInput - Web Speech API hook for voice input
 *
 * Features:
 * - Real-time transcription with interim results
 * - Command parsing for engine routing
 * - Speech-to-math symbol conversion
 * - Browser compatibility detection
 *
 * @param {Object} options Configuration options
 * @param {Function} options.onResult Callback when final result is ready
 * @param {Function} options.onError Callback on error
 * @param {boolean} options.continuous Keep listening after result
 * @param {boolean} options.interimResults Show results while speaking
 * @param {string} options.language Speech recognition language
 */
export function useVoiceInput(options = {}) {
  const {
    onResult = () => {},
    onError = () => {},
    continuous = false,
    interimResults = true,
    language = 'en-US'
  } = options

  const [isListening, setIsListening] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [interimTranscript, setInterimTranscript] = useState('')
  const [confidence, setConfidence] = useState(0)
  const [isSupported, setIsSupported] = useState(true)
  const [error, setError] = useState(null)

  const recognitionRef = useRef(null)

  // Initialize SpeechRecognition
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition

    if (!SpeechRecognition) {
      setIsSupported(false)
      setError('Speech recognition not supported in this browser')
      return
    }

    const recognition = new SpeechRecognition()
    recognition.continuous = continuous
    recognition.interimResults = interimResults
    recognition.lang = language
    recognition.maxAlternatives = 1

    recognition.onstart = () => {
      setIsListening(true)
      setError(null)
    }

    recognition.onresult = (event) => {
      let interim = ''
      let final = ''

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i]
        if (result.isFinal) {
          final += result[0].transcript
          setConfidence(result[0].confidence || 0)
        } else {
          interim += result[0].transcript
        }
      }

      if (final) {
        setTranscript(prev => (prev ? prev + ' ' : '') + final.trim())
        const parsed = parseCommand(final)
        onResult(parsed)
      }
      setInterimTranscript(interim)
    }

    recognition.onerror = (event) => {
      const errorMessages = {
        'no-speech': 'No speech detected. Please try again.',
        'audio-capture': 'Microphone not available. Check permissions.',
        'not-allowed': 'Microphone permission denied.',
        'network': 'Network error. Check your connection.',
        'aborted': 'Speech recognition was aborted.',
        'service-not-allowed': 'Speech service not allowed.'
      }
      const message = errorMessages[event.error] || `Speech error: ${event.error}`
      setError(message)
      onError(event.error, message)
      setIsListening(false)
    }

    recognition.onend = () => {
      setIsListening(false)
      setInterimTranscript('')
    }

    recognitionRef.current = recognition

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.abort()
      }
    }
  }, [continuous, interimResults, language, onResult, onError])

  const startListening = useCallback(() => {
    if (!recognitionRef.current || !isSupported) return false

    try {
      setError(null)
      setTranscript('')
      setInterimTranscript('')
      setConfidence(0)
      recognitionRef.current.start()
      return true
    } catch (err) {
      // Handle "already started" error
      if (err.name === 'InvalidStateError') {
        recognitionRef.current.stop()
        setTimeout(() => {
          try {
            recognitionRef.current.start()
          } catch (e) {
            setError('Failed to start speech recognition')
          }
        }, 100)
      } else {
        setError('Failed to start speech recognition')
      }
      return false
    }
  }, [isSupported])

  const stopListening = useCallback(() => {
    if (!recognitionRef.current) return

    try {
      recognitionRef.current.stop()
    } catch (err) {
      // Ignore errors when stopping
    }
    setIsListening(false)
  }, [])

  const resetTranscript = useCallback(() => {
    setTranscript('')
    setInterimTranscript('')
    setConfidence(0)
    setError(null)
  }, [])

  return {
    isListening,
    transcript,
    interimTranscript,
    confidence,
    isSupported,
    error,
    startListening,
    stopListening,
    resetTranscript
  }
}

/**
 * Parse voice command to extract engine and query
 *
 * Examples:
 * - "math solve 2x plus 5 equals 13" → { engine: 'math', query: '2x + 5 = 13' }
 * - "chemistry balance H2 plus O2" → { engine: 'chemistry', query: 'H2 + O2' }
 * - "quant kelly 60% win 2 to 1 odds" → { engine: 'quant', query: 'kelly 60% win 2 to 1 odds' }
 */
function parseCommand(text) {
  const lower = text.toLowerCase().trim()

  // Engine prefix mappings
  const prefixes = {
    'math': 'math',
    'solve': 'math',
    'calculate': 'math',
    'compute': 'math',
    'chemistry': 'chemistry',
    'balance': 'chemistry',
    'molecular': 'chemistry',
    'biology': 'biology',
    'genetics': 'biology',
    'punnett': 'biology',
    'stats': 'statistics',
    'statistics': 'statistics',
    'probability': 'statistics',
    'hypothesis': 'statistics',
    'code': 'cs',
    'debug': 'cs',
    'explain code': 'cs',
    'complexity': 'cs',
    'poker': 'poker',
    'hand': 'poker',
    'quant': 'quant',
    'mental math': 'quant',
    'options': 'quant',
    'black scholes': 'quant',
    'kelly': 'quant',
    'sharpe': 'quant',
    'fermi': 'quant',
    'interview': 'quant'
  }

  let engine = null
  let query = lower

  // Check for engine prefix (longest match first)
  const sortedPrefixes = Object.entries(prefixes).sort((a, b) => b[0].length - a[0].length)

  for (const [prefix, eng] of sortedPrefixes) {
    if (lower.startsWith(prefix)) {
      engine = eng
      query = lower.slice(prefix.length).replace(/^[:\s]+/, '').trim()
      break
    }
  }

  // Convert speech to math symbols
  query = speechToMath(query)

  return {
    engine,
    query,
    raw: text,
    hasEngine: engine !== null
  }
}

/**
 * Convert spoken math to mathematical symbols
 */
function speechToMath(text) {
  return text
    // Basic operators
    .replace(/\bplus\b/gi, '+')
    .replace(/\bminus\b/gi, '-')
    .replace(/\btimes\b/gi, '*')
    .replace(/\bmultiplied by\b/gi, '*')
    .replace(/\bdivided by\b/gi, '/')
    .replace(/\bover\b/gi, '/')
    // Powers
    .replace(/\bsquared\b/gi, '**2')
    .replace(/\bcubed\b/gi, '**3')
    .replace(/\bto the power of\b/gi, '**')
    .replace(/\bto the\s+(\d+)\b/gi, '**$1')
    // Roots
    .replace(/\bsquare root of\b/gi, 'sqrt(')
    .replace(/\bsqrt of\b/gi, 'sqrt(')
    .replace(/\bcube root of\b/gi, 'cbrt(')
    // Equality
    .replace(/\bequals\b/gi, '=')
    .replace(/\bis equal to\b/gi, '=')
    // Variables and constants
    .replace(/\bpi\b/gi, 'pi')
    .replace(/\beuler\b/gi, 'e')
    // Trigonometric
    .replace(/\bsine of\b/gi, 'sin(')
    .replace(/\bcosine of\b/gi, 'cos(')
    .replace(/\btangent of\b/gi, 'tan(')
    .replace(/\bsin of\b/gi, 'sin(')
    .replace(/\bcos of\b/gi, 'cos(')
    .replace(/\btan of\b/gi, 'tan(')
    // Logarithms
    .replace(/\bnatural log of\b/gi, 'ln(')
    .replace(/\blog of\b/gi, 'log(')
    .replace(/\blog base (\d+) of\b/gi, 'log$1(')
    // Absolute value
    .replace(/\babsolute value of\b/gi, 'abs(')
    // Percentages
    .replace(/\bpercent\b/gi, '%')
    // Fractions (partial support)
    .replace(/(\d+)\s+and\s+(\d+)\/(\d+)/gi, '$1+$2/$3') // "3 and 1/4" → "3+1/4"
    // Numbers written as words (basic)
    .replace(/\bzero\b/gi, '0')
    .replace(/\bone\b/gi, '1')
    .replace(/\btwo\b/gi, '2')
    .replace(/\bthree\b/gi, '3')
    .replace(/\bfour\b/gi, '4')
    .replace(/\bfive\b/gi, '5')
    .replace(/\bsix\b/gi, '6')
    .replace(/\bseven\b/gi, '7')
    .replace(/\beight\b/gi, '8')
    .replace(/\bnine\b/gi, '9')
    .replace(/\bten\b/gi, '10')
    // Clean up extra spaces
    .replace(/\s+/g, ' ')
    .trim()
}

/**
 * Text-to-Speech utility for reading responses
 *
 * @param {string} text Text to speak
 * @param {Object} options Speech options
 * @returns {boolean} Whether speech synthesis is supported
 */
export function speak(text, options = {}) {
  if (!window.speechSynthesis) return false

  // Cancel any ongoing speech
  window.speechSynthesis.cancel()

  const utterance = new SpeechSynthesisUtterance(text)
  utterance.rate = options.rate || 1.0
  utterance.pitch = options.pitch || 1.0
  utterance.volume = options.volume || 1.0
  utterance.lang = options.lang || 'en-US'

  if (options.onEnd) {
    utterance.onend = options.onEnd
  }
  if (options.onError) {
    utterance.onerror = options.onError
  }

  window.speechSynthesis.speak(utterance)
  return true
}

/**
 * Stop any ongoing speech synthesis
 */
export function stopSpeaking() {
  if (window.speechSynthesis) {
    window.speechSynthesis.cancel()
  }
}

/**
 * Check if speech synthesis is currently speaking
 */
export function isSpeaking() {
  return window.speechSynthesis?.speaking || false
}

export default useVoiceInput
