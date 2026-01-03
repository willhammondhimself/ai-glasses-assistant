/**
 * LatexRenderer - KaTeX-based LaTeX rendering component
 *
 * Supports both inline ($...$) and display ($$...$$) math.
 * Handles mixed content with text and LaTeX seamlessly.
 */

import { useMemo, memo } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'

/**
 * Render mixed content containing text and LaTeX expressions
 * @param {string} content - Text with LaTeX delimiters
 * @returns {string} HTML string with rendered LaTeX
 */
function renderMixedContent(content) {
  if (!content) return ''

  let result = content

  // First handle display math ($$...$$)
  result = result.replace(/\$\$([\s\S]+?)\$\$/g, (match, tex) => {
    try {
      return katex.renderToString(tex.trim(), {
        displayMode: true,
        throwOnError: false,
        strict: false,
        trust: true
      })
    } catch (e) {
      console.warn('LaTeX display render error:', e)
      return `<span class="text-red-500">${match}</span>`
    }
  })

  // Then handle inline math ($...$)
  result = result.replace(/\$([^\$\n]+?)\$/g, (match, tex) => {
    try {
      return katex.renderToString(tex.trim(), {
        displayMode: false,
        throwOnError: false,
        strict: false,
        trust: true
      })
    } catch (e) {
      console.warn('LaTeX inline render error:', e)
      return `<span class="text-red-500">${match}</span>`
    }
  })

  return result
}

/**
 * Convert common plain-text math notation to LaTeX
 * @param {string} text - Plain text math
 * @returns {string} Text with LaTeX notation
 */
export function convertToLatex(text) {
  if (!text) return ''

  let result = text

  // Convert Python/SymPy notation to LaTeX
  result = result.replace(/\*\*/g, '^')  // ** to ^
  result = result.replace(/\*/g, ' \\cdot ')  // * to \cdot
  result = result.replace(/sqrt\(([^)]+)\)/g, '\\sqrt{$1}')  // sqrt(x) to \sqrt{x}
  result = result.replace(/pi/gi, '\\pi')  // pi to \pi
  result = result.replace(/infinity|inf/gi, '\\infty')  // infinity to \infty
  result = result.replace(/exp\(([^)]+)\)/g, 'e^{$1}')  // exp(x) to e^{x}
  result = result.replace(/log\(([^)]+)\)/g, '\\log($1)')  // log(x) to \log(x)
  result = result.replace(/ln\(([^)]+)\)/g, '\\ln($1)')  // ln(x) to \ln(x)
  result = result.replace(/sin\(([^)]+)\)/g, '\\sin($1)')
  result = result.replace(/cos\(([^)]+)\)/g, '\\cos($1)')
  result = result.replace(/tan\(([^)]+)\)/g, '\\tan($1)')

  return result
}

/**
 * Memoized LaTeX renderer component
 */
const LatexRenderer = memo(function LatexRenderer({
  content,
  className = '',
  convertPlainMath = false
}) {
  const rendered = useMemo(() => {
    if (!content) return ''

    let processedContent = content

    // Optionally convert plain math notation
    if (convertPlainMath) {
      processedContent = convertToLatex(processedContent)
    }

    return renderMixedContent(processedContent)
  }, [content, convertPlainMath])

  return (
    <span
      className={`latex-content ${className}`}
      dangerouslySetInnerHTML={{ __html: rendered }}
    />
  )
})

/**
 * Display-mode only LaTeX block
 */
export function LatexBlock({ content, className = '' }) {
  const rendered = useMemo(() => {
    if (!content) return ''

    try {
      return katex.renderToString(content, {
        displayMode: true,
        throwOnError: false,
        strict: false,
        trust: true
      })
    } catch (e) {
      console.warn('LaTeX block render error:', e)
      return `<span class="text-red-500">${content}</span>`
    }
  }, [content])

  return (
    <div
      className={`latex-block bg-slate-50 rounded-lg p-4 my-4 border-l-4 border-blue-500 overflow-x-auto ${className}`}
      dangerouslySetInnerHTML={{ __html: rendered }}
    />
  )
}

/**
 * Inline-mode only LaTeX span
 */
export function LatexInline({ content, className = '' }) {
  const rendered = useMemo(() => {
    if (!content) return ''

    try {
      return katex.renderToString(content, {
        displayMode: false,
        throwOnError: false,
        strict: false,
        trust: true
      })
    } catch (e) {
      console.warn('LaTeX inline render error:', e)
      return `<span class="text-red-500">${content}</span>`
    }
  }, [content])

  return (
    <span
      className={`latex-inline ${className}`}
      dangerouslySetInnerHTML={{ __html: rendered }}
    />
  )
}

export default LatexRenderer
