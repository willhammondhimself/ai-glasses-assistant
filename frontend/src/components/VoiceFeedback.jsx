import React from 'react'
import { AlertCircle, Mic } from 'lucide-react'

/**
 * VoiceFeedback - Display voice transcription and status
 *
 * Shows:
 * - Listening indicator with animation
 * - Real-time transcript with interim results
 * - Confidence score
 * - Error messages
 */
export function VoiceFeedback({
  transcript,
  interimTranscript,
  confidence,
  isListening,
  error,
  showConfidence = true,
  className = ''
}) {
  // Don't render if nothing to show
  if (!isListening && !transcript && !error) {
    return null
  }

  return (
    <div className={`mt-2 ${className}`}>
      {/* Error display */}
      {error && (
        <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span className="text-sm">{error}</span>
        </div>
      )}

      {/* Transcript display */}
      {(isListening || transcript) && !error && (
        <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
          {/* Status header */}
          <div className="flex items-center justify-between mb-1">
            <div className="flex items-center gap-2 text-sm text-gray-600">
              <span
                className={`
                  w-2 h-2 rounded-full
                  ${isListening ? 'bg-red-500 animate-pulse' : 'bg-gray-400'}
                `}
              />
              <span className="font-medium">
                {isListening ? 'Listening...' : 'Voice input'}
              </span>
            </div>

            {/* Confidence indicator */}
            {showConfidence && confidence > 0 && (
              <span className="text-xs text-gray-400">
                {Math.round(confidence * 100)}% confidence
              </span>
            )}
          </div>

          {/* Transcript text */}
          <p className="text-gray-800 min-h-[1.5rem]">
            {transcript}
            {interimTranscript && (
              <span className="text-gray-400 italic"> {interimTranscript}</span>
            )}
            {isListening && !transcript && !interimTranscript && (
              <span className="text-gray-400 italic">Speak now...</span>
            )}
          </p>
        </div>
      )}
    </div>
  )
}

/**
 * VoiceFeedbackCompact - Minimal inline version
 */
export function VoiceFeedbackCompact({
  transcript,
  interimTranscript,
  isListening,
  error
}) {
  if (!isListening && !transcript && !error) {
    return null
  }

  if (error) {
    return (
      <span className="text-xs text-red-500 ml-2">{error}</span>
    )
  }

  return (
    <span className="text-xs text-gray-500 ml-2">
      {isListening && <Mic className="w-3 h-3 inline animate-pulse text-red-500 mr-1" />}
      {transcript || interimTranscript || (isListening ? 'Listening...' : '')}
    </span>
  )
}

/**
 * VoiceFeedbackOverlay - Full-screen overlay for continuous listening mode
 */
export function VoiceFeedbackOverlay({
  transcript,
  interimTranscript,
  confidence,
  isListening,
  error,
  onClose
}) {
  if (!isListening) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-xl p-8 max-w-lg w-full mx-4">
        {/* Listening indicator */}
        <div className="flex flex-col items-center mb-6">
          <div className="w-20 h-20 rounded-full bg-red-100 flex items-center justify-center mb-4 animate-pulse">
            <Mic className="w-10 h-10 text-red-500" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900">Listening...</h3>
          <p className="text-sm text-gray-500">Speak clearly into your microphone</p>
        </div>

        {/* Transcript */}
        <div className="min-h-[100px] p-4 bg-gray-50 rounded-lg mb-4">
          {error ? (
            <p className="text-red-600">{error}</p>
          ) : (
            <p className="text-gray-800 text-lg">
              {transcript}
              <span className="text-gray-400 italic">{interimTranscript}</span>
              {!transcript && !interimTranscript && (
                <span className="text-gray-400 italic">Your words will appear here...</span>
              )}
            </p>
          )}
        </div>

        {/* Confidence bar */}
        {confidence > 0 && (
          <div className="mb-4">
            <div className="flex justify-between text-xs text-gray-500 mb-1">
              <span>Confidence</span>
              <span>{Math.round(confidence * 100)}%</span>
            </div>
            <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-green-500 transition-all duration-300"
                style={{ width: `${confidence * 100}%` }}
              />
            </div>
          </div>
        )}

        {/* Close button */}
        <button
          onClick={onClose}
          className="w-full py-2 px-4 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
        >
          Stop listening
        </button>
      </div>
    </div>
  )
}

export default VoiceFeedback
