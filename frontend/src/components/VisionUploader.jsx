import React, { useState, useRef } from 'react'
import { Eye, Upload, Play, Loader2, AlertCircle, Image, X } from 'lucide-react'
import { visionAnalyze } from '../services/api'

function VisionUploader() {
  const [imageBase64, setImageBase64] = useState('')
  const [imagePreview, setImagePreview] = useState(null)
  const [prompt, setPrompt] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [duration, setDuration] = useState(null)
  const fileInputRef = useRef(null)

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (!file) return

    if (!file.type.startsWith('image/')) {
      setError('Please select an image file')
      return
    }

    const reader = new FileReader()
    reader.onload = (e) => {
      const base64 = e.target.result.split(',')[1]
      setImageBase64(base64)
      setImagePreview(e.target.result)
      setError(null)
    }
    reader.readAsDataURL(file)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader()
      reader.onload = (e) => {
        const base64 = e.target.result.split(',')[1]
        setImageBase64(base64)
        setImagePreview(e.target.result)
        setError(null)
      }
      reader.readAsDataURL(file)
    }
  }

  const clearImage = () => {
    setImageBase64('')
    setImagePreview(null)
    setResult(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!imageBase64 || !prompt.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await visionAnalyze(imageBase64, prompt)
      setResult(response.data)
      setDuration(response.durationMs)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const promptExamples = [
    'What is in this image?',
    'Solve this math problem',
    'Explain this diagram',
    'Transcribe the text in this image',
    'What type of chart is this and what does it show?'
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="p-3 bg-purple-100 rounded-xl">
          <Eye className="w-6 h-6 text-purple-600" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Vision Analyzer</h2>
          <p className="text-gray-600">Analyze images with GPT-4o Vision</p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Image upload area */}
        <div className="card">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Image
          </label>

          {!imagePreview ? (
            <div
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center cursor-pointer hover:border-primary-400 hover:bg-primary-50 transition-colors"
            >
              <Image className="w-12 h-12 text-gray-400 mx-auto mb-3" />
              <p className="text-gray-600">
                Drop an image here or click to upload
              </p>
              <p className="text-sm text-gray-500 mt-1">
                Supports JPG, PNG, GIF, WebP
              </p>
            </div>
          ) : (
            <div className="relative">
              <img
                src={imagePreview}
                alt="Preview"
                className="max-h-64 rounded-lg mx-auto"
              />
              <button
                type="button"
                onClick={clearImage}
                className="absolute top-2 right-2 p-1 bg-red-500 text-white rounded-full hover:bg-red-600"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          )}

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
          />
        </div>

        {/* Prompt input */}
        <div className="card">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Prompt
          </label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="What would you like to know about this image?"
            rows={3}
            className="input-field resize-none"
          />

          <div className="mt-3">
            <p className="text-xs text-gray-500 mb-2">Quick prompts:</p>
            <div className="flex flex-wrap gap-2">
              {promptExamples.map((ex) => (
                <button
                  key={ex}
                  type="button"
                  onClick={() => setPrompt(ex)}
                  className="px-2 py-1 bg-gray-100 rounded text-xs hover:bg-gray-200 transition-colors"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        </div>

        <button
          type="submit"
          disabled={loading || !imageBase64 || !prompt.trim()}
          className="btn-primary flex items-center gap-2"
        >
          {loading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Play className="w-4 h-4" />
          )}
          {loading ? 'Analyzing...' : 'Analyze Image'}
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
            <h3 className="font-medium text-gray-900">Analysis Result</h3>
            {duration && (
              <span className="text-xs text-gray-500">{duration}ms</span>
            )}
          </div>

          <div className="result-box result-success whitespace-pre-wrap">
            {result.analysis}
          </div>

          {result.model && (
            <p className="text-xs text-gray-500">Model: {result.model}</p>
          )}
        </div>
      )}
    </div>
  )
}

export default VisionUploader
