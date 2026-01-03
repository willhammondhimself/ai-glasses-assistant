import React from 'react'
import { Mic, MicOff } from 'lucide-react'

/**
 * MicButton - Voice input toggle button with visual feedback
 *
 * Features:
 * - Pulsing animation when listening
 * - Disabled state for unsupported browsers
 * - Accessible with proper ARIA attributes
 */
export function MicButton({
  isListening,
  isSupported = true,
  onStart,
  onStop,
  className = '',
  size = 'md',
  showLabel = false
}) {
  // Size variants
  const sizes = {
    sm: 'p-1.5',
    md: 'p-2',
    lg: 'p-3'
  }

  const iconSizes = {
    sm: 'w-4 h-4',
    md: 'w-5 h-5',
    lg: 'w-6 h-6'
  }

  // Unsupported browser state
  if (!isSupported) {
    return (
      <button
        disabled
        className={`
          inline-flex items-center justify-center gap-2
          ${sizes[size]}
          rounded-lg
          bg-gray-100 text-gray-400
          cursor-not-allowed
          ${className}
        `}
        title="Voice input not supported in this browser. Try Chrome or Edge."
        aria-label="Voice input not supported"
      >
        <MicOff className={iconSizes[size]} />
        {showLabel && <span className="text-sm">Voice unavailable</span>}
      </button>
    )
  }

  const handleClick = () => {
    if (isListening) {
      onStop?.()
    } else {
      onStart?.()
    }
  }

  return (
    <button
      type="button"
      onClick={handleClick}
      className={`
        inline-flex items-center justify-center gap-2
        ${sizes[size]}
        rounded-lg
        transition-all duration-200
        ${isListening
          ? 'bg-red-100 text-red-600 ring-2 ring-red-300 animate-pulse'
          : 'bg-gray-100 text-gray-600 hover:bg-gray-200 hover:text-gray-800'
        }
        focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2
        ${className}
      `}
      title={isListening ? 'Stop listening (click or wait)' : 'Start voice input'}
      aria-label={isListening ? 'Stop voice input' : 'Start voice input'}
      aria-pressed={isListening}
    >
      <Mic className={`${iconSizes[size]} ${isListening ? 'text-red-600' : ''}`} />
      {showLabel && (
        <span className="text-sm font-medium">
          {isListening ? 'Listening...' : 'Voice'}
        </span>
      )}
    </button>
  )
}

/**
 * MicButtonInline - Compact version for inline use in input fields
 */
export function MicButtonInline({
  isListening,
  isSupported = true,
  onStart,
  onStop,
  className = ''
}) {
  if (!isSupported) {
    return (
      <button
        disabled
        className={`p-1 text-gray-300 cursor-not-allowed ${className}`}
        title="Voice not supported"
        aria-hidden="true"
      >
        <MicOff className="w-4 h-4" />
      </button>
    )
  }

  return (
    <button
      type="button"
      onClick={isListening ? onStop : onStart}
      className={`
        p-1 rounded
        transition-colors duration-150
        ${isListening
          ? 'text-red-500 animate-pulse'
          : 'text-gray-400 hover:text-gray-600'
        }
        focus:outline-none focus:text-primary-500
        ${className}
      `}
      title={isListening ? 'Stop' : 'Voice input'}
      aria-label={isListening ? 'Stop voice input' : 'Start voice input'}
    >
      <Mic className={`w-4 h-4 ${isListening ? 'animate-pulse' : ''}`} />
    </button>
  )
}

export default MicButton
