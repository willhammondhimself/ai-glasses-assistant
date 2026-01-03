import React, { useState, useEffect } from 'react'
import { Routes, Route, NavLink, useLocation } from 'react-router-dom'
import {
  Calculator,
  Eye,
  Code,
  Spade,
  FlaskConical,
  Dna,
  BarChart3,
  History,
  DollarSign,
  Menu,
  X,
  Activity,
  TrendingUp,
  BookOpen,
  Terminal
} from 'lucide-react'

import MathSolver from './components/MathSolver'
import VisionUploader from './components/VisionUploader'
import CSHelper from './components/CSHelper'
import PokerAnalyzer from './components/PokerAnalyzer'
import ChemistrySolver from './components/ChemistrySolver'
import BiologyHelper from './components/BiologyHelper'
import StatsCalculator from './components/StatsCalculator'
import QuantFinance from './components/QuantFinance/QuantFinance'
import Flashcards from './components/Flashcards/Flashcards'
import LeetCode from './components/LeetCode/LeetCode'
import HistoryView from './components/HistoryView'
import CostTracker from './components/CostTracker'
import { healthCheck } from './services/api'

const navItems = [
  { path: '/', icon: Calculator, label: 'Math', color: 'text-blue-600' },
  { path: '/vision', icon: Eye, label: 'Vision', color: 'text-purple-600' },
  { path: '/cs', icon: Code, label: 'CS', color: 'text-green-600' },
  { path: '/poker', icon: Spade, label: 'Poker', color: 'text-red-600' },
  { path: '/chemistry', icon: FlaskConical, label: 'Chemistry', color: 'text-orange-600' },
  { path: '/biology', icon: Dna, label: 'Biology', color: 'text-pink-600' },
  { path: '/stats', icon: BarChart3, label: 'Stats', color: 'text-cyan-600' },
  { path: '/quant', icon: TrendingUp, label: 'Quant', color: 'text-emerald-600' },
  { path: '/leetcode', icon: Terminal, label: 'LeetCode', color: 'text-orange-500' },
  { path: '/flashcards', icon: BookOpen, label: 'Flashcards', color: 'text-violet-600' },
  { path: '/history', icon: History, label: 'History', color: 'text-gray-600' },
  { path: '/costs', icon: DollarSign, label: 'Costs', color: 'text-yellow-600' }
]

function App() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [serverStatus, setServerStatus] = useState('checking')
  const location = useLocation()

  useEffect(() => {
    setMobileMenuOpen(false)
  }, [location])

  useEffect(() => {
    const checkHealth = async () => {
      try {
        await healthCheck()
        setServerStatus('online')
      } catch {
        setServerStatus('offline')
      }
    }
    checkHealth()
    const interval = setInterval(checkHealth, 30000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-700 rounded-xl flex items-center justify-center">
                <Activity className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">AI Glasses Coach</h1>
                <p className="text-xs text-gray-500">Test Dashboard v3.0</p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${
                  serverStatus === 'online' ? 'bg-green-500' :
                  serverStatus === 'offline' ? 'bg-red-500' : 'bg-yellow-500'
                }`} />
                <span className="text-sm text-gray-600 hidden sm:inline">
                  {serverStatus === 'online' ? 'Server Online' :
                   serverStatus === 'offline' ? 'Server Offline' : 'Checking...'}
                </span>
              </div>

              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="lg:hidden p-2 rounded-lg hover:bg-gray-100"
              >
                {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar - Desktop */}
        <aside className="hidden lg:flex lg:flex-col lg:w-64 lg:fixed lg:inset-y-0 lg:pt-16 lg:bg-white lg:border-r lg:border-gray-200">
          <nav className="flex-1 px-4 py-6 space-y-1 overflow-y-auto">
            {navItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors ${
                    isActive
                      ? 'bg-primary-50 text-primary-700 font-medium'
                      : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                  }`
                }
              >
                <item.icon className={`w-5 h-5 ${item.color}`} />
                <span>{item.label}</span>
              </NavLink>
            ))}
          </nav>
        </aside>

        {/* Mobile menu */}
        {mobileMenuOpen && (
          <div className="fixed inset-0 z-40 lg:hidden">
            <div
              className="fixed inset-0 bg-black/50"
              onClick={() => setMobileMenuOpen(false)}
            />
            <div className="fixed inset-y-0 left-0 w-64 bg-white shadow-xl">
              <nav className="px-4 py-6 space-y-1 mt-16">
                {navItems.map((item) => (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    className={({ isActive }) =>
                      `flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors ${
                        isActive
                          ? 'bg-primary-50 text-primary-700 font-medium'
                          : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                      }`
                    }
                  >
                    <item.icon className={`w-5 h-5 ${item.color}`} />
                    <span>{item.label}</span>
                  </NavLink>
                ))}
              </nav>
            </div>
          </div>
        )}

        {/* Main content */}
        <main className="flex-1 lg:ml-64 min-h-screen">
          <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <Routes>
              <Route path="/" element={<MathSolver />} />
              <Route path="/vision" element={<VisionUploader />} />
              <Route path="/cs" element={<CSHelper />} />
              <Route path="/poker" element={<PokerAnalyzer />} />
              <Route path="/chemistry" element={<ChemistrySolver />} />
              <Route path="/biology" element={<BiologyHelper />} />
              <Route path="/stats" element={<StatsCalculator />} />
              <Route path="/quant" element={<QuantFinance />} />
              <Route path="/leetcode" element={<LeetCode />} />
              <Route path="/flashcards" element={<Flashcards />} />
              <Route path="/history" element={<HistoryView />} />
              <Route path="/costs" element={<CostTracker />} />
            </Routes>
          </div>
        </main>
      </div>
    </div>
  )
}

export default App
