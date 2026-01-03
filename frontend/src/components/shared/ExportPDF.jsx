/**
 * ExportPDF - Export content to PDF with LaTeX support
 *
 * Uses html2canvas to capture rendered content and jspdf to create PDF.
 * Lazy loads dependencies to reduce initial bundle size.
 */

import { useState, useCallback } from 'react'
import { Download, Loader2 } from 'lucide-react'

/**
 * Export PDF button component
 *
 * @param {Object} props
 * @param {React.RefObject} props.contentRef - Ref to the element to export
 * @param {string} props.title - PDF filename (without extension)
 * @param {Function} props.onExport - Callback when export completes
 * @param {string} props.className - Additional CSS classes
 */
export function ExportPDF({
  contentRef,
  title = 'export',
  onExport,
  className = ''
}) {
  const [exporting, setExporting] = useState(false)

  const handleExport = useCallback(async () => {
    if (!contentRef?.current) {
      console.error('No content ref provided for PDF export')
      return
    }

    setExporting(true)

    try {
      // Lazy load heavy dependencies
      const [{ default: jsPDF }, { default: html2canvas }] = await Promise.all([
        import('jspdf'),
        import('html2canvas')
      ])

      // Capture the content as canvas
      const canvas = await html2canvas(contentRef.current, {
        scale: 2,  // Higher quality
        useCORS: true,
        logging: false,
        backgroundColor: '#ffffff'
      })

      // Create PDF
      const pdf = new jsPDF('p', 'mm', 'a4')
      const imgData = canvas.toDataURL('image/png')
      const pdfWidth = pdf.internal.pageSize.getWidth()
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width

      // Handle multi-page if content is too long
      const pageHeight = pdf.internal.pageSize.getHeight()
      let heightLeft = pdfHeight
      let position = 0

      pdf.addImage(imgData, 'PNG', 0, position, pdfWidth, pdfHeight)
      heightLeft -= pageHeight

      while (heightLeft >= 0) {
        position = heightLeft - pdfHeight
        pdf.addPage()
        pdf.addImage(imgData, 'PNG', 0, position, pdfWidth, pdfHeight)
        heightLeft -= pageHeight
      }

      // Generate filename with date
      const date = new Date().toISOString().split('T')[0]
      pdf.save(`${title}-${date}.pdf`)

      onExport?.()
    } catch (error) {
      console.error('PDF export error:', error)
    } finally {
      setExporting(false)
    }
  }, [contentRef, title, onExport])

  return (
    <button
      onClick={handleExport}
      disabled={exporting}
      className={`inline-flex items-center gap-2 px-3 py-2 text-sm font-medium rounded-lg
        bg-gray-100 hover:bg-gray-200 text-gray-700 disabled:opacity-50
        transition-colors ${className}`}
    >
      {exporting ? (
        <>
          <Loader2 className="w-4 h-4 animate-spin" />
          Exporting...
        </>
      ) : (
        <>
          <Download className="w-4 h-4" />
          Export PDF
        </>
      )}
    </button>
  )
}

/**
 * Copy LaTeX source button
 */
export function CopyLatex({ content, className = '' }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (error) {
      console.error('Copy failed:', error)
    }
  }, [content])

  return (
    <button
      onClick={handleCopy}
      className={`inline-flex items-center gap-1 px-2 py-1 text-xs font-medium rounded
        ${copied ? 'bg-green-100 text-green-700' : 'bg-gray-100 hover:bg-gray-200 text-gray-600'}
        transition-colors ${className}`}
    >
      {copied ? 'Copied!' : 'Copy LaTeX'}
    </button>
  )
}

/**
 * Export solution as PDF - creates a formatted document
 */
export async function exportSolutionToPDF(solution, title = 'Solution') {
  const { default: jsPDF } = await import('jspdf')

  const pdf = new jsPDF('p', 'mm', 'a4')
  const margin = 20
  let y = margin

  // Title
  pdf.setFontSize(18)
  pdf.setFont('helvetica', 'bold')
  pdf.text(title, margin, y)
  y += 15

  // Date
  pdf.setFontSize(10)
  pdf.setFont('helvetica', 'normal')
  pdf.setTextColor(128, 128, 128)
  pdf.text(`Generated: ${new Date().toLocaleString()}`, margin, y)
  y += 10

  // Problem
  pdf.setTextColor(0, 0, 0)
  pdf.setFontSize(12)
  pdf.setFont('helvetica', 'bold')
  pdf.text('Problem:', margin, y)
  y += 7

  pdf.setFont('helvetica', 'normal')
  const problemLines = pdf.splitTextToSize(solution.problem || '', 170)
  pdf.text(problemLines, margin, y)
  y += problemLines.length * 6 + 10

  // Solution
  pdf.setFont('helvetica', 'bold')
  pdf.text('Solution:', margin, y)
  y += 7

  pdf.setFont('helvetica', 'normal')
  const solutionLines = pdf.splitTextToSize(solution.answer || solution.solution || '', 170)
  pdf.text(solutionLines, margin, y)
  y += solutionLines.length * 6 + 10

  // Explanation (if available)
  if (solution.explanation || solution.steps) {
    pdf.setFont('helvetica', 'bold')
    pdf.text('Explanation:', margin, y)
    y += 7

    pdf.setFont('helvetica', 'normal')
    const explainLines = pdf.splitTextToSize(solution.explanation || solution.steps || '', 170)
    pdf.text(explainLines, margin, y)
  }

  // Save
  const date = new Date().toISOString().split('T')[0]
  pdf.save(`${title.replace(/\s+/g, '-').toLowerCase()}-${date}.pdf`)
}

export default ExportPDF
