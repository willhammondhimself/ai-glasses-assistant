/**
 * WHAM Dashboard - JavaScript
 * Handles all API interactions and UI updates
 */

// API Base URL
const API_BASE = '/dashboard/api';

// State
let currentCapturePage = 1;
const capturesPerPage = 20;

// WebSocket State
let ws = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 10;
let heartbeatInterval = null;

// ============================================================
// Initialization
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize tabs
    initTabs();

    // Load initial data
    loadHealth();
    loadTodayData();
    loadChallenges();

    // Connect WebSocket for live updates
    connectWebSocket();

    // Set up auto-refresh as fallback (reduced frequency since we have WebSocket)
    setInterval(loadHealth, 60000); // Every minute (was 30s)
    setInterval(loadTodayData, 120000); // Every 2 minutes (was 1min)

    // Set up event listeners
    document.getElementById('captureSearch').addEventListener('input', debounce(loadCaptures, 300));
    document.getElementById('captureFilter').addEventListener('change', loadCaptures);
    document.getElementById('memoryQuery').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') searchMemory();
    });
});

// ============================================================
// WebSocket Connection
// ============================================================

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/dashboard/ws/dashboard`;

    console.log('Connecting to WebSocket:', wsUrl);
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('Dashboard WebSocket connected');
        reconnectAttempts = 0;
        updateConnectionStatus(true);

        // Start heartbeat
        if (heartbeatInterval) clearInterval(heartbeatInterval);
        heartbeatInterval = setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    };

    ws.onmessage = (event) => {
        try {
            const message = JSON.parse(event.data);
            handleWebSocketMessage(message);
        } catch (e) {
            console.error('Failed to parse WebSocket message:', e);
        }
    };

    ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        updateConnectionStatus(false);

        if (heartbeatInterval) {
            clearInterval(heartbeatInterval);
            heartbeatInterval = null;
        }

        // Attempt reconnection with exponential backoff
        if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
            reconnectAttempts++;
            const delay = Math.min(3000 * reconnectAttempts, 30000);
            console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts})`);
            setTimeout(connectWebSocket, delay);
        } else {
            console.log('Max reconnection attempts reached');
            document.getElementById('connectionStatus').textContent = 'Offline';
        }
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

function updateConnectionStatus(connected) {
    const dot = document.querySelector('.live-dot');
    const status = document.getElementById('connectionStatus');

    if (dot && status) {
        if (connected) {
            dot.classList.add('connected');
            status.textContent = 'Live';
        } else {
            dot.classList.remove('connected');
            status.textContent = reconnectAttempts > 0 ? 'Reconnecting...' : 'Connecting...';
        }
    }
}

function handleWebSocketMessage(message) {
    console.log('WS message:', message.type, message.data);

    switch (message.type) {
        case 'connected':
            console.log('Server confirmed connection');
            break;

        case 'session_update':
            // Add new activity to timeline
            addTimelineItem(message.data);
            break;

        case 'challenge_complete':
            // Reload challenges and show notification
            loadChallenges();
            showNotification(`🏆 Challenge complete! +${message.data.xp} XP`);
            break;

        case 'cost_update':
            // Update cost display
            const costEl = document.getElementById('todayCost');
            if (costEl && message.data.today !== undefined) {
                costEl.textContent = `$${message.data.today.toFixed(2)}`;
            }
            break;

        case 'capture_created':
            // Refresh recent captures
            loadTodayData();
            showNotification(`📝 New capture saved`);
            break;

        case 'pong':
            // Heartbeat acknowledged
            break;

        case 'vision_status':
            // Update WHAM Vision status
            const visionStatus = document.getElementById('visionStatus');
            const visionMode = document.getElementById('visionMode');
            const visionPill = document.querySelector('.vision-pill');

            if (visionStatus && visionMode && visionPill) {
                if (message.data.active) {
                    visionMode.textContent = message.data.current_detection || 'Scanning';
                    visionPill.classList.add('active');
                } else {
                    visionMode.textContent = 'Idle';
                    visionPill.classList.remove('active');
                }
            }
            break;

        case 'homework_solution':
            // Real-time homework solution push from phone client
            handleHomeworkSolutionPush(message.data);
            break;

        case 'agent_response':
            // Real-time agent response push for multi-client sync
            if (message.data && message.data.response) {
                let responseHtml = message.data.response;
                if (message.data.tool_used) {
                    responseHtml = `<span class="tool-badge">${message.data.tool_used}</span> ${responseHtml}`;
                }
                addAgentMessage('assistant', responseHtml);
            }
            break;

        default:
            console.log('Unknown message type:', message.type);
    }
}

function addTimelineItem(data) {
    const timeline = document.getElementById('activityTimeline');
    if (!timeline) return;

    // Remove "No activities" message if present
    const emptyMsg = timeline.querySelector('.empty');
    if (emptyMsg) emptyMsg.remove();

    const item = document.createElement('div');
    item.className = 'timeline-item new';
    item.innerHTML = `
        <div class="timeline-icon">${getActivityIcon(data.session_type || data.type)}</div>
        <div class="timeline-content">
            <div class="timeline-time">${formatTime(new Date().toISOString())}</div>
            <div class="timeline-desc">${data.description || data.session_type || 'Activity'}</div>
        </div>
    `;

    // Insert at top of timeline
    if (timeline.firstChild) {
        timeline.insertBefore(item, timeline.firstChild);
    } else {
        timeline.appendChild(item);
    }

    // Remove highlight after animation
    setTimeout(() => item.classList.remove('new'), 500);

    // Limit timeline items (remove old ones)
    const items = timeline.querySelectorAll('.timeline-item');
    if (items.length > 20) {
        items[items.length - 1].remove();
    }
}

function showNotification(message) {
    const toast = document.createElement('div');
    toast.className = 'toast-notification';
    toast.textContent = message;
    document.body.appendChild(toast);

    // Remove after animation
    setTimeout(() => {
        if (toast.parentNode) {
            toast.remove();
        }
    }, 3000);
}

/**
 * Handle real-time homework solution push from phone client.
 * Auto-switches to Homework tab and displays the solution.
 */
function handleHomeworkSolutionPush(data) {
    // Show toast notification
    const problemPreview = data.problem_latex?.substring(0, 30) || 'New problem';
    showNotification(`📐 Solved: ${problemPreview}...`);

    // Auto-switch to Homework tab (if not already there)
    const currentTab = document.querySelector('.tab.active')?.dataset.tab;
    const needsTabSwitch = currentTab !== 'homework';

    if (needsTabSwitch) {
        const homeworkTabBtn = document.querySelector('[data-tab="homework"]');
        if (homeworkTabBtn) {
            homeworkTabBtn.click();
        }
    }

    // Prepare the result object
    const result = {
        problem: {
            equation: data.problem_latex,
            problem_type: data.problem_type,
            confidence: data.confidence
        },
        solution_steps: data.solution_steps,
        concept_explanation: data.concept_explanation,
        tool_used: data.tool_used,
        execution_time_ms: data.execution_time_ms,
        success: data.success
    };

    // Display solution - delay if tab switch needed to let DOM update
    const displayAndScroll = () => {
        displayHomeworkSolution(result);
        const solutionEl = document.getElementById('homeworkSolution');
        if (solutionEl) {
            solutionEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    };

    if (needsTabSwitch) {
        setTimeout(displayAndScroll, 100);  // Wait for tab DOM to update
    } else {
        displayAndScroll();
    }

    // Refresh history to include new entry
    setTimeout(() => {
        if (typeof loadHomeworkHistory === 'function') {
            loadHomeworkHistory();
        }
    }, 500);
}

// ============================================================
// Tab Navigation
// ============================================================

function initTabs() {
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active from all
            tabs.forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            // Add active to clicked
            tab.classList.add('active');
            const tabId = tab.dataset.tab;
            document.getElementById(`tab-${tabId}`).classList.add('active');

            // Load tab-specific data
            switch (tabId) {
                case 'analytics':
                    loadAnalytics();
                    break;
                case 'captures':
                    loadCaptures();
                    break;
                case 'memory':
                    loadMemoryStats();
                    break;
                case 'opponents':
                    loadOpponents();
                    break;
                case 'suggestions':
                    loadSuggestions();
                    break;
                case 'research':
                    loadResearch();
                    break;
                case 'skills':
                    loadSkills();
                    break;
                case 'academic':
                    // Academic tab loaded on demand via tool buttons
                    break;
                case 'homework':
                    loadHomeworkHistory();
                    loadHomeworkStats();
                    break;
                case 'poker-lab':
                    loadPokerSessions();
                    break;
            }
        });
    });
}

// ============================================================
// API Helpers
// ============================================================

async function api(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function formatTime(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function formatDate(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

function formatDuration(seconds) {
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

// ============================================================
// Health & System Status
// ============================================================

async function loadHealth() {
    try {
        const health = await api('/health');

        // Update footer
        document.getElementById('systemStatus').textContent =
            health.status === 'healthy' ? 'System Online' : 'System Degraded';
        document.querySelector('.status-indicator').className =
            `status-indicator ${health.status === 'healthy' ? 'online' : 'offline'}`;
        document.getElementById('dbSize').textContent = `DB: ${health.database_size_mb} MB`;
        document.getElementById('uptime').textContent = `Uptime: ${formatDuration(health.uptime_seconds)}`;
    } catch (error) {
        document.getElementById('systemStatus').textContent = 'Connection Error';
        document.querySelector('.status-indicator').className = 'status-indicator offline';
    }
}

// ============================================================
// Today Tab
// ============================================================

async function loadTodayData() {
    try {
        const data = await api('/sessions/today');

        // Update timeline
        const timeline = document.getElementById('activityTimeline');
        if (data.activities.length === 0) {
            timeline.innerHTML = '<p class="empty">No activities yet today. Start capturing!</p>';
        } else {
            timeline.innerHTML = data.activities.map(activity => `
                <div class="timeline-item">
                    <div class="timeline-icon">${getActivityIcon(activity.type)}</div>
                    <div class="timeline-content">
                        <div class="timeline-time">${formatTime(activity.timestamp)}</div>
                        <div class="timeline-desc">${activity.description}</div>
                    </div>
                </div>
            `).join('');
        }

        // Update stats
        document.getElementById('todayCost').textContent = `$${data.stats.total_cost.toFixed(2)}`;

        // Update recent captures
        const capturesDiv = document.getElementById('recentCaptures');
        if (data.recent_captures.length === 0) {
            capturesDiv.innerHTML = '<p class="empty">No captures yet</p>';
        } else {
            capturesDiv.innerHTML = data.recent_captures.map(capture => `
                <div class="timeline-item">
                    <div class="timeline-icon">📝</div>
                    <div class="timeline-content">
                        <div class="timeline-time">${formatTime(capture.created_at)}</div>
                        <div class="timeline-desc">${capture.text.substring(0, 100)}${capture.text.length > 100 ? '...' : ''}</div>
                    </div>
                </div>
            `).join('');
        }
    } catch (error) {
        console.error('Failed to load today data:', error);
    }
}

function getActivityIcon(type) {
    const icons = {
        'capture': '📝',
        'focus': '🎯',
        'poker': '🎰',
        'homework': '📚',
        'debug': '🔧',
        'briefing': '☀️'
    };
    return icons[type] || '•';
}

// ============================================================
// Challenges
// ============================================================

async function loadChallenges() {
    try {
        const data = await api('/challenges/today');

        // Update challenges list
        const list = document.getElementById('challengesList');
        list.innerHTML = data.challenges.map(challenge => {
            const progress = Math.min(100, (challenge.progress / challenge.target) * 100);
            const isComplete = challenge.completed;
            return `
                <div class="challenge-item">
                    <div class="challenge-header">
                        <span class="challenge-title">${isComplete ? '✅' : '⬜'} ${challenge.title}</span>
                        <span class="challenge-xp">+${challenge.xp_reward} XP</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill ${isComplete ? 'complete' : ''}" style="width: ${progress}%"></div>
                    </div>
                    <div style="font-size: 12px; color: var(--text-dim); margin-top: 4px;">
                        ${challenge.progress}/${challenge.target} - ${challenge.description}
                    </div>
                </div>
            `;
        }).join('');

        // Update XP bar
        const xp = data.xp;
        const xpProgress = 100 - (xp.xp_to_next_level / (xp.total_xp + xp.xp_to_next_level) * 100);
        document.getElementById('xpProgress').style.width = `${xpProgress}%`;
        document.getElementById('xpLevel').textContent = `Level ${xp.level}`;
        document.getElementById('xpAmount').textContent = `${xp.total_xp} XP`;

        // Update analytics tab if loaded
        document.getElementById('totalXP').textContent = xp.total_xp;
        document.getElementById('streakDays').textContent = xp.current_streak;
        document.getElementById('achievements').textContent = xp.achievements_unlocked;
    } catch (error) {
        console.error('Failed to load challenges:', error);
    }
}

// ============================================================
// Analytics Tab
// ============================================================

async function loadAnalytics() {
    try {
        // Load session stats
        const stats = await api('/sessions/stats');
        document.getElementById('focusMinutes').textContent = stats.focus_minutes;

        // Load history for chart
        const history = await api('/sessions/history?days=7');
        renderCostChart(history.days);

        // Load poker stats
        const poker = await api('/poker/stats');
        document.getElementById('pokerHands').textContent = poker.total_hands;
        document.getElementById('pokerProfit').textContent = `${poker.total_profit_bb >= 0 ? '+' : ''}${poker.total_profit_bb.toFixed(1)} BB`;
        document.getElementById('pokerVPIP').textContent = `${poker.vpip.toFixed(1)}%`;
    } catch (error) {
        console.error('Failed to load analytics:', error);
    }
}

// Store chart instance for cleanup
let costChartInstance = null;

function renderCostChart(days) {
    const canvas = document.getElementById('costChart');
    if (!canvas) return;

    // Destroy existing chart if any
    if (costChartInstance) {
        costChartInstance.destroy();
        costChartInstance = null;
    }

    // Check if Chart.js is available
    if (typeof Chart === 'undefined') {
        // Fallback to simple canvas rendering
        renderCostChartFallback(canvas, days);
        return;
    }

    // Reverse to show oldest first
    const sortedDays = [...days].reverse();

    costChartInstance = new Chart(canvas, {
        type: 'line',
        data: {
            labels: sortedDays.map(d => formatDate(d.date)),
            datasets: [{
                label: 'Daily Cost ($)',
                data: sortedDays.map(d => d.stats.total_cost),
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.15)',
                tension: 0.4,
                fill: true,
                pointBackgroundColor: '#6366f1',
                pointBorderColor: '#fff',
                pointRadius: 4,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(26, 26, 36, 0.95)',
                    titleColor: '#fff',
                    bodyColor: '#a0a0b0',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false,
                    callbacks: {
                        label: (ctx) => `Cost: $${ctx.parsed.y.toFixed(2)}`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.06)' },
                    ticks: {
                        color: '#606070',
                        callback: (value) => '$' + value.toFixed(2)
                    }
                },
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.06)' },
                    ticks: { color: '#606070' }
                }
            }
        }
    });
}

function renderCostChartFallback(canvas, days) {
    // Simple canvas fallback if Chart.js isn't loaded
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const padding = 40;
    const width = canvas.width - padding * 2;
    const height = canvas.height - padding * 2;
    const sortedDays = [...days].reverse();
    const maxCost = Math.max(...sortedDays.map(d => d.stats.total_cost), 5);
    const barWidth = width / sortedDays.length - 10;

    sortedDays.forEach((day, i) => {
        const x = padding + i * (barWidth + 10);
        const barHeight = (day.stats.total_cost / maxCost) * height;
        const y = padding + height - barHeight;

        ctx.fillStyle = '#6366f1';
        ctx.fillRect(x, y, barWidth, barHeight);

        ctx.fillStyle = '#a0a0b0';
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(formatDate(day.date), x + barWidth / 2, canvas.height - 10);
    });

    ctx.fillStyle = '#606070';
    ctx.textAlign = 'right';
    ctx.fillText('$0', padding - 5, padding + height);
    ctx.fillText(`$${maxCost.toFixed(2)}`, padding - 5, padding + 10);
}

// ============================================================
// Captures Tab
// ============================================================

async function loadCaptures() {
    try {
        const search = document.getElementById('captureSearch').value;
        const filter = document.getElementById('captureFilter').value;

        let url = `/captures?page=${currentCapturePage}&per_page=${capturesPerPage}`;
        if (filter) url += `&category=${filter}`;

        const data = await api(url);

        const tbody = document.getElementById('capturesBody');
        if (data.captures.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="empty">No captures found</td></tr>';
        } else {
            tbody.innerHTML = data.captures
                .filter(c => !search || c.text.toLowerCase().includes(search.toLowerCase()))
                .map(capture => `
                    <tr>
                        <td>${formatTime(capture.created_at)}</td>
                        <td>${capture.text.substring(0, 60)}${capture.text.length > 60 ? '...' : ''}</td>
                        <td>${capture.tags.map(t => `<span class="tag">${t}</span>`).join('')}</td>
                        <td class="${capture.priority === 'high' || capture.priority === 'urgent' ? 'priority-' + capture.priority : ''}">${capture.priority}</td>
                        <td class="table-actions">
                            <button class="table-btn" onclick="editCapture('${capture.id}')">Edit</button>
                            <button class="table-btn danger" onclick="deleteCapture('${capture.id}')">Delete</button>
                        </td>
                    </tr>
                `).join('');
        }

        // Update pagination
        document.getElementById('capturesPage').textContent = `Page ${data.page}`;
        document.getElementById('prevCaptures').disabled = data.page <= 1;
        document.getElementById('nextCaptures').disabled = data.page * capturesPerPage >= data.total;
    } catch (error) {
        console.error('Failed to load captures:', error);
    }
}

async function deleteCapture(id) {
    if (!confirm('Delete this capture?')) return;

    try {
        await api(`/captures/${id}`, { method: 'DELETE' });
        loadCaptures();
        loadTodayData();
    } catch (error) {
        alert('Failed to delete capture');
    }
}

function editCapture(id) {
    // Would open edit modal - simplified for now
    alert('Edit functionality coming soon');
}

// ============================================================
// Memory Tab
// ============================================================

async function loadMemoryStats() {
    try {
        const stats = await api('/memory/stats');

        document.getElementById('totalMemories').textContent = stats.total_memories;
        document.getElementById('totalEntities').textContent = stats.total_entities;

        // Render categories
        const maxCount = Math.max(...Object.values(stats.by_category), 1);
        document.getElementById('memoryCategories').innerHTML =
            Object.entries(stats.by_category).map(([cat, count]) => `
                <div class="category-bar">
                    <span class="category-label">${cat}</span>
                    <div class="category-progress">
                        <div class="category-fill" style="width: ${(count / maxCount) * 100}%"></div>
                    </div>
                    <span class="category-count">${count}</span>
                </div>
            `).join('');
    } catch (error) {
        console.error('Failed to load memory stats:', error);
    }
}

async function searchMemory() {
    const query = document.getElementById('memoryQuery').value.trim();
    if (!query) return;

    try {
        const data = await api(`/memory/search?q=${encodeURIComponent(query)}`);

        const results = document.getElementById('memoryResults');
        if (data.memories.length === 0) {
            results.innerHTML = '<p class="empty">No memories found for this query</p>';
        } else {
            results.innerHTML = data.memories.map(memory => `
                <div class="memory-card">
                    <div class="memory-key">${memory.key}</div>
                    <div class="memory-value">${memory.value}</div>
                    <div class="memory-meta">
                        ${memory.category} • Accessed ${memory.access_count} times
                    </div>
                </div>
            `).join('');
        }
    } catch (error) {
        console.error('Failed to search memory:', error);
    }
}

async function addMemory(event) {
    event.preventDefault();

    const key = document.getElementById('memoryKey').value;
    const value = document.getElementById('memoryValue').value;
    const category = document.getElementById('memoryCategory').value;

    try {
        await api('/memory', {
            method: 'POST',
            body: JSON.stringify({ key, value, category })
        });

        // Clear form
        document.getElementById('memoryKey').value = '';
        document.getElementById('memoryValue').value = '';

        // Reload stats
        loadMemoryStats();
        alert('Memory stored successfully!');
    } catch (error) {
        alert('Failed to store memory');
    }
}

// ============================================================
// Opponents Tab
// ============================================================

async function loadOpponents() {
    try {
        const data = await api('/poker/opponents');

        const tbody = document.getElementById('opponentsBody');
        if (data.opponents.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="empty">No opponents tracked yet. Start a poker session to track opponents.</td></tr>';
        } else {
            tbody.innerHTML = data.opponents.map(opp => `
                <tr>
                    <td><strong>${opp.name}</strong></td>
                    <td>${opp.vpip.toFixed(1)}%</td>
                    <td>${opp.pfr.toFixed(1)}%</td>
                    <td>${opp.aggression.toFixed(2)}</td>
                    <td>${opp.hands_observed}</td>
                    <td><span class="tag">${opp.archetype}</span></td>
                    <td>${opp.last_seen ? formatDate(opp.last_seen) : '-'}</td>
                </tr>
            `).join('');
        }
    } catch (error) {
        console.error('Failed to load opponents:', error);
    }
}

// ============================================================
// Suggestions Tab
// ============================================================

async function loadSuggestions() {
    try {
        const suggestions = await api('/suggestions');
        const container = document.getElementById('suggestionsContent');

        if (suggestions.length === 0) {
            container.innerHTML = `
                <div class="empty">
                    <p>No suggestions right now.</p>
                    <p class="hint">Keep using WHAM to build usage patterns!</p>
                </div>
            `;
            return;
        }

        container.innerHTML = suggestions.map(sugg => `
            <div class="suggestion-card">
                <div class="suggestion-header">
                    <span class="suggestion-icon">💡</span>
                    <span class="suggestion-title">${sugg.title}</span>
                    <span class="suggestion-confidence">${Math.round(sugg.confidence * 100)}% confident</span>
                </div>
                <div class="suggestion-message">${sugg.message}</div>
                ${sugg.action ? `
                    <button class="suggestion-action" onclick="executeSuggestionAction('${sugg.action}')">
                        Take Action
                    </button>
                ` : ''}
            </div>
        `).join('');
    } catch (error) {
        console.error('Failed to load suggestions:', error);
        document.getElementById('suggestionsContent').innerHTML = `
            <div class="empty">
                <p>Failed to load suggestions.</p>
                <p class="hint">Check console for details.</p>
            </div>
        `;
    }
}

function executeSuggestionAction(action) {
    console.log('Executing suggestion action:', action);
    showNotification(`Action: ${action.replace(/_/g, ' ')}`);

    // Could integrate with desktop app or trigger API calls
    // For now, just show acknowledgment
}

// ============================================================
// Research Tab (Phase 4)
// ============================================================

async function loadResearch() {
    try {
        const response = await api('/research/history?days=7');
        const container = document.getElementById('researchContent');

        if (response.queries.length === 0) {
            container.innerHTML = `
                <div class="empty">
                    <p>No research queries yet.</p>
                    <p class="hint">Use Perplexity research mode to get started!</p>
                </div>
            `;
            return;
        }

        container.innerHTML = `
            <div class="stats-row">
                <div class="stat">
                    <span class="stat-label">Total Queries</span>
                    <span class="stat-value">${response.total_queries}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Total Cost</span>
                    <span class="stat-value">$${response.total_cost.toFixed(3)}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Period</span>
                    <span class="stat-value">${response.period_days} days</span>
                </div>
            </div>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Query</th>
                        <th>Time</th>
                        <th>Cost</th>
                        <th>Citations</th>
                    </tr>
                </thead>
                <tbody>
                    ${response.queries.map(q => `
                        <tr>
                            <td>${q.query}</td>
                            <td>${formatTime(q.timestamp)}</td>
                            <td>$${q.cost.toFixed(3)}</td>
                            <td>${q.citations.length}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
    } catch (error) {
        console.error('Failed to load research:', error);
        document.getElementById('researchContent').innerHTML = `
            <div class="empty">
                <p>Failed to load research history.</p>
                <p class="hint">Check console for details.</p>
            </div>
        `;
    }
}

// ============================================================
// Skills Tab (Phase 4)
// ============================================================

async function loadSkills() {
    try {
        const response = await api('/skills?days=30');
        const container = document.getElementById('skillsContent');

        if (!response.success) {
            container.innerHTML = `
                <div class="empty">
                    <p>Insufficient data for skill analysis.</p>
                    <p class="hint">Keep using WHAM to build skill history!</p>
                </div>
            `;
            return;
        }

        const metrics = response.metrics;
        let html = '';

        // Render skill cards
        if (metrics.poker) {
            html += renderSkillCard('🎰 Poker', metrics.poker);
        }
        if (metrics.homework) {
            html += renderSkillCard('📚 Homework', metrics.homework);
        }
        if (metrics.focus) {
            html += renderSkillCard('⏱️ Focus', metrics.focus);
        }

        // Overall consistency card
        html += `
            <div class="skill-card">
                <h3>📊 Consistency</h3>
                <div class="skill-level">${Math.round(metrics.overall_consistency * 100)}%</div>
                <p class="hint">Activity regularity over ${response.days_analyzed} days</p>
            </div>
        `;

        container.innerHTML = html;
    } catch (error) {
        console.error('Failed to load skills:', error);
        document.getElementById('skillsContent').innerHTML = `
            <div class="empty">
                <p>Failed to load skill metrics.</p>
                <p class="hint">Check console for details.</p>
            </div>
        `;
    }
}

function renderSkillCard(title, skill) {
    const trendEmoji = {
        'improving': '📈',
        'stable': '➡️',
        'declining': '📉'
    }[skill.trend];

    const levelColor = {
        'beginner': '#fbbf24',
        'intermediate': '#60a5fa',
        'advanced': '#34d399'
    }[skill.skill_level];

    return `
        <div class="skill-card">
            <h3>${title}</h3>
            <div class="skill-level" style="color: ${levelColor}">
                ${skill.skill_level.toUpperCase()}
            </div>
            <div class="skill-trend">${trendEmoji} ${skill.trend}</div>
            <div class="skill-bar">
                <div class="skill-progress" style="width: ${skill.confidence * 100}%"></div>
            </div>
            <p class="hint">${Math.round(skill.confidence * 100)}% confidence</p>
        </div>
    `;
}

// ============================================================
// Academic Tab (Phase 4B)
// ============================================================

async function buildConceptBridge() {
    const concept = document.getElementById('conceptInput').value.trim();
    const connectToInput = document.getElementById('connectToInput').value.trim();
    const resultDiv = document.getElementById('conceptBridgeResult');

    if (!concept || !connectToInput) {
        resultDiv.innerHTML = '<p class="empty">Please enter both a concept and topics to connect to.</p>';
        return;
    }

    const connectTo = connectToInput.split(',').map(s => s.trim()).filter(s => s);
    if (connectTo.length === 0) {
        resultDiv.innerHTML = '<p class="empty">Please enter at least one topic to connect to.</p>';
        return;
    }

    resultDiv.innerHTML = '<p class="hint">Building concept bridge... 🌉</p>';

    try {
        const result = await api('/academic/concept-bridge', {
            method: 'POST',
            body: JSON.stringify({ concept, connect_to: connectTo, level: 'undergraduate' })
        });

        let html = `
            <div class="academic-result">
                <h4>${result.concept}</h4>
                <p><strong>Importance:</strong> ${result.importance}</p>
        `;

        result.relationships.forEach(rel => {
            html += `
                <div class="connection-card">
                    <h5>→ ${rel.target}</h5>
                    <p><strong>Relationship:</strong> ${rel.relationship}</p>
                    <p><strong>Example:</strong> ${rel.example}</p>
                </div>
            `;
        });

        html += `
                <p class="hint" style="margin-top: 12px;">
                    Cost: $${result.cost.toFixed(3)} |
                    Sources: ${result.sources.length}
                </p>
            </div>
        `;

        resultDiv.innerHTML = html;
    } catch (error) {
        console.error('Failed to build concept bridge:', error);
        resultDiv.innerHTML = '<p class="empty">Failed to build concept bridge. Check console for details.</p>';
    }
}

async function explainDerivation() {
    const equation = document.getElementById('equationInput').value.trim();
    const context = document.getElementById('contextInput').value.trim();
    const resultDiv = document.getElementById('derivationResult');

    if (!equation) {
        resultDiv.innerHTML = '<p class="empty">Please enter an equation to derive.</p>';
        return;
    }

    resultDiv.innerHTML = '<p class="hint">Deriving steps... 📐</p>';

    try {
        const result = await api('/academic/derivation', {
            method: 'POST',
            body: JSON.stringify({ equation, context, show_all_steps: true })
        });

        let html = `
            <div class="academic-result">
                <h4>${result.equation}</h4>
                ${result.context ? `<p class="hint">${result.context}</p>` : ''}
                <div class="derivation-steps">
        `;

        result.steps.forEach((step, idx) => {
            html += `
                <div class="step-card">
                    <div class="step-number">Step ${idx + 1}</div>
                    <div class="step-content">
                        <p><strong>Expression:</strong> ${step.expression}</p>
                        <p><strong>Explanation:</strong> ${step.explanation}</p>
                        ${step.justification ? `<p class="hint">${step.justification}</p>` : ''}
                    </div>
                </div>
            `;
        });

        html += `
                </div>
                ${result.final_result ? `<p style="margin-top: 12px;"><strong>Final Result:</strong> ${result.final_result}</p>` : ''}
                <p class="hint" style="margin-top: 12px;">
                    Cost: $${result.cost.toFixed(3)} |
                    Sources: ${result.sources.length}
                </p>
            </div>
        `;

        resultDiv.innerHTML = html;
    } catch (error) {
        console.error('Failed to explain derivation:', error);
        resultDiv.innerHTML = '<p class="empty">Failed to explain derivation. Check console for details.</p>';
    }
}

async function generateStrategy() {
    const problemType = document.getElementById('problemTypeInput').value.trim();
    const resultDiv = document.getElementById('strategyResult');

    if (!problemType) {
        resultDiv.innerHTML = '<p class="empty">Please enter a problem type.</p>';
        return;
    }

    resultDiv.innerHTML = '<p class="hint">Generating strategy... 🎯</p>';

    try {
        const result = await api('/academic/problem-strategy', {
            method: 'POST',
            body: JSON.stringify({ problem_type: problemType, include_examples: true })
        });

        let html = `
            <div class="academic-result">
                <h4>${result.problem_type}</h4>
                <div class="strategy-section">
                    <h5>🎯 Approach Steps</h5>
                    <ol>
        `;

        result.approach_steps.forEach(step => {
            html += `<li>${step}</li>`;
        });

        html += `
                    </ol>
                </div>
                <div class="strategy-section">
                    <h5>💡 Key Insights</h5>
                    <ul>
        `;

        result.key_insights.forEach(insight => {
            html += `<li>${insight}</li>`;
        });

        html += `
                    </ul>
                </div>
                <div class="strategy-section">
                    <h5>⚠️ Common Pitfalls</h5>
                    <ul>
        `;

        result.common_pitfalls.forEach(pitfall => {
            html += `<li>${pitfall}</li>`;
        });

        html += `
                    </ul>
                </div>
        `;

        if (result.example_problem) {
            html += `
                <div class="strategy-section">
                    <h5>📝 Example Problem</h5>
                    <p>${result.example_problem}</p>
                </div>
            `;
        }

        html += `
                <p class="hint" style="margin-top: 12px;">
                    Cost: $${result.cost.toFixed(3)} |
                    Sources: ${result.sources.length}
                </p>
            </div>
        `;

        resultDiv.innerHTML = html;
    } catch (error) {
        console.error('Failed to generate strategy:', error);
        resultDiv.innerHTML = '<p class="empty">Failed to generate strategy. Check console for details.</p>';
    }
}

async function analyzeExamPattern() {
    const course = document.getElementById('courseInput').value.trim();
    const examType = document.getElementById('examTypeSelect').value;
    const resultDiv = document.getElementById('examPatternResult');

    if (!course) {
        resultDiv.innerHTML = '<p class="empty">Please enter a course name.</p>';
        return;
    }

    resultDiv.innerHTML = '<p class="hint">Analyzing exam patterns... 📊</p>';

    try {
        const result = await api('/academic/exam-pattern', {
            method: 'POST',
            body: JSON.stringify({ course, exam_type: examType, past_exams_context: '' })
        });

        let html = `
            <div class="academic-result">
                <h4>${result.course} - ${result.exam_type}</h4>
                <div class="pattern-section">
                    <h5>🔥 High Frequency Topics</h5>
                    <ul>
        `;

        result.high_frequency_topics.forEach(topic => {
            html += `<li>${topic}</li>`;
        });

        html += `
                    </ul>
                </div>
                <div class="pattern-section">
                    <h5>📋 Common Question Types</h5>
                    <ul>
        `;

        result.common_question_types.forEach(qtype => {
            html += `<li>${qtype}</li>`;
        });

        html += `
                    </ul>
                </div>
                <div class="pattern-section">
                    <h5>💡 Study Recommendations</h5>
                    <ul>
        `;

        result.study_recommendations.forEach(rec => {
            html += `<li>${rec}</li>`;
        });

        html += `
                    </ul>
                </div>
                <p class="hint" style="margin-top: 12px;">
                    Confidence: ${Math.round(result.confidence * 100)}% |
                    Cost: $${result.cost.toFixed(3)} |
                    Sources: ${result.sources.length}
                </p>
            </div>
        `;

        resultDiv.innerHTML = html;
    } catch (error) {
        console.error('Failed to analyze exam pattern:', error);
        resultDiv.innerHTML = '<p class="empty">Failed to analyze exam pattern. Check console for details.</p>';
    }
}

async function decodeNotation() {
    const notation = document.getElementById('notationInput').value.trim();
    const subjectContext = document.getElementById('notationContextInput').value.trim();
    const resultDiv = document.getElementById('notationResult');

    if (!notation) {
        resultDiv.innerHTML = '<p class="empty">Please enter a symbol or notation.</p>';
        return;
    }

    resultDiv.innerHTML = '<p class="hint">Decoding notation... 🔤</p>';

    try {
        const result = await api('/academic/notation', {
            method: 'POST',
            body: JSON.stringify({ notation, subject_context: subjectContext })
        });

        let html = `
            <div class="academic-result">
                <h4>${result.notation}</h4>
                <p><strong>Name:</strong> ${result.name}</p>
                <p><strong>Meaning:</strong> ${result.meaning}</p>
        `;

        if (result.common_contexts && result.common_contexts.length > 0) {
            html += `
                <div class="notation-section">
                    <h5>📚 Common Contexts</h5>
                    <ul>
            `;
            result.common_contexts.forEach(ctx => {
                html += `<li>${ctx}</li>`;
            });
            html += `
                    </ul>
                </div>
            `;
        }

        if (result.example_usage) {
            html += `
                <div class="notation-section">
                    <h5>💡 Example Usage</h5>
                    <p>${result.example_usage}</p>
                </div>
            `;
        }

        html += `
                <p class="hint" style="margin-top: 12px;">
                    Cost: $${result.cost.toFixed(3)} |
                    Sources: ${result.sources.length}
                </p>
            </div>
        `;

        resultDiv.innerHTML = html;
    } catch (error) {
        console.error('Failed to decode notation:', error);
        resultDiv.innerHTML = '<p class="empty">Failed to decode notation. Check console for details.</p>';
    }
}

// ============================================================
// Phase 4C: Advanced Academic Features
// ============================================================

// Helper function for escaping HTML in LaTeX formulas
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function findVisualIntuition() {
    const concept = document.getElementById('visualConcept').value.trim();
    const subjectArea = document.getElementById('visualSubject').value.trim();
    const resultDiv = document.getElementById('visualIntuitionResult');

    if (!concept) {
        resultDiv.innerHTML = '<p class="empty">Please enter a concept.</p>';
        return;
    }

    resultDiv.innerHTML = '<p class="empty">🎨 Finding visual intuition...</p>';

    try {
        const result = await api('/api/academic/visual-intuition', {
            method: 'POST',
            body: JSON.stringify({
                concept: concept,
                subject_area: subjectArea
            })
        });

        let html = `
            <div class="academic-result">
                <h4>🎨 Visual Intuition: ${result.concept}</h4>

                <div class="notation-section">
                    <h5>🔄 Analogies</h5>
                    <ul>
        `;

        result.analogies.forEach(analogy => {
            html += `<li>${analogy}</li>`;
        });

        html += `
                    </ul>
                </div>

                <div class="notation-section">
                    <h5>🧠 Mental Models</h5>
                    <ul>
        `;

        result.mental_models.forEach(model => {
            html += `<li>${model}</li>`;
        });

        html += `
                    </ul>
                </div>

                <div class="notation-section">
                    <h5>👁️ Visual Resources</h5>
                    <ul>
        `;

        result.visual_resources.forEach(resource => {
            const typeEmoji = resource.type === 'video' ? '🎥' : '📊';
            html += `<li>${typeEmoji} <strong>${resource.type}:</strong> ${resource.description}</li>`;
        });

        html += `
                    </ul>
                </div>

                <div class="notation-section">
                    <h5>📚 Learning Approach</h5>
                    <p>${result.learning_approach}</p>
                </div>

                <p class="hint" style="margin-top: 12px;">
                    Cost: $${result.cost.toFixed(3)} |
                    Sources: ${result.sources.length}
                </p>
            </div>
        `;

        resultDiv.innerHTML = html;
    } catch (error) {
        console.error('Failed to find visual intuition:', error);
        resultDiv.innerHTML = '<p class="empty">Failed to find visual intuition. Check console for details.</p>';
    }
}

async function generateFormulaSheet() {
    const topic = document.getElementById('formulaTopic').value.trim();
    const resultDiv = document.getElementById('formulaSheetResult');

    if (!topic) {
        resultDiv.innerHTML = '<p class="empty">Please enter a topic.</p>';
        return;
    }

    resultDiv.innerHTML = '<p class="empty">📐 Generating formula sheet...</p>';

    try {
        const result = await api('/api/academic/formula-sheet', {
            method: 'POST',
            body: JSON.stringify({
                topic: topic
            })
        });

        let html = `
            <div class="academic-result">
                <h4>📐 Formula Sheet: ${result.topic}</h4>
        `;

        result.categories.forEach(category => {
            html += `
                <div class="notation-section">
                    <h5>📋 ${category.name}</h5>
                    <ul>
            `;

            category.formulas.forEach(formula => {
                html += `<li><code>${escapeHtml(formula)}</code></li>`;
            });

            html += `
                    </ul>
                </div>
            `;
        });

        html += `
                <div class="notation-section">
                    <h5>🔗 Key Relationships</h5>
                    <ul>
        `;

        result.key_relationships.forEach(relationship => {
            html += `<li>${relationship}</li>`;
        });

        html += `
                    </ul>
                </div>

                <div class="notation-section">
                    <h5>⚠️ Common Mistakes</h5>
                    <ul>
        `;

        result.common_mistakes.forEach(mistake => {
            html += `<li>${mistake}</li>`;
        });

        html += `
                    </ul>
                </div>

                <p class="hint" style="margin-top: 12px;">
                    Cost: $${result.cost.toFixed(3)} |
                    Sources: ${result.sources.length}
                </p>
            </div>
        `;

        resultDiv.innerHTML = html;
    } catch (error) {
        console.error('Failed to generate formula sheet:', error);
        resultDiv.innerHTML = '<p class="empty">Failed to generate formula sheet. Check console for details.</p>';
    }
}

async function prereadPaper() {
    const title = document.getElementById('paperTitle').value.trim();
    const authors = document.getElementById('paperAuthors').value.trim();
    const year = document.getElementById('paperYear').value.trim();
    const resultDiv = document.getElementById('paperSummaryResult');

    if (!title) {
        resultDiv.innerHTML = '<p class="empty">Please enter a paper title.</p>';
        return;
    }

    resultDiv.innerHTML = '<p class="empty">📄 Analyzing paper...</p>';

    try {
        const result = await api('/api/academic/paper-summary', {
            method: 'POST',
            body: JSON.stringify({
                title: title,
                authors: authors,
                year: year
            })
        });

        const relevancePercent = (result.relevance_score * 100).toFixed(0);
        const relevanceColor = result.relevance_score >= 0.7 ? '#4ade80' : result.relevance_score >= 0.4 ? '#fbbf24' : '#f87171';

        let html = `
            <div class="academic-result">
                <h4>📄 ${result.title}</h4>
                <p class="hint">${result.authors} (${result.year})</p>

                <div class="notation-section">
                    <h5>🎯 Main Contribution</h5>
                    <p>${result.main_contribution}</p>
                </div>

                <div class="notation-section">
                    <h5>🔍 Key Findings</h5>
                    <ul>
        `;

        result.key_findings.forEach(finding => {
            html += `<li>${finding}</li>`;
        });

        html += `
                    </ul>
                </div>

                <div class="notation-section">
                    <h5>🔬 Methodology</h5>
                    <p>${result.methodology}</p>
                </div>

                <div class="notation-section">
                    <h5>⚠️ Limitations</h5>
                    <ul>
        `;

        result.limitations.forEach(limitation => {
            html += `<li>${limitation}</li>`;
        });

        html += `
                    </ul>
                </div>

                <div class="notation-section">
                    <h5>📊 Relevance Score</h5>
                    <div style="background: #2a2a2a; border-radius: 8px; overflow: hidden; margin-top: 8px;">
                        <div style="background: ${relevanceColor}; width: ${relevancePercent}%; padding: 8px; text-align: center; color: #000; font-weight: bold; min-width: 60px;">
                            ${relevancePercent}%
                        </div>
                    </div>
                </div>

                <p class="hint" style="margin-top: 12px;">
                    Cost: $${result.cost.toFixed(3)} |
                    Sources: ${result.sources.length}
                </p>
            </div>
        `;

        resultDiv.innerHTML = html;
    } catch (error) {
        console.error('Failed to analyze paper:', error);
        resultDiv.innerHTML = '<p class="empty">Failed to analyze paper. Check console for details.</p>';
    }
}

// ============================================================
// Poker Lab (Phase 5)
// ============================================================

let currentSessionId = null;

async function loadPokerSessions() {
    try {
        const data = await api('/poker/files');
        const sessionList = document.getElementById('sessionList');

        if (data.length === 0) {
            sessionList.innerHTML = '<p class="empty">No poker sessions yet</p>';
            return;
        }

        sessionList.innerHTML = data.map(session => {
            const date = new Date(session.date * 1000);
            const dateStr = date.toLocaleDateString([], { month: 'short', day: 'numeric' });
            const profitClass = session.profit_bb >= 0 ? 'positive' : 'negative';
            const profitSign = session.profit_bb >= 0 ? '+' : '';
            const reviewedClass = session.reviewed ? 'reviewed' : '';

            return `
                <div class="session-item ${reviewedClass}" onclick="selectSession('${session.session_id}')">
                    <div class="session-date">${dateStr}</div>
                    <div class="session-stats">
                        <span>${session.hands} hands</span>
                        <span class="session-profit ${profitClass}">${profitSign}${session.profit_bb.toFixed(1)} BB</span>
                    </div>
                </div>
            `;
        }).join('');
    } catch (error) {
        console.error('Failed to load poker sessions:', error);
        document.getElementById('sessionList').innerHTML = '<p class="empty">Failed to load sessions</p>';
    }
}

async function selectSession(sessionId) {
    currentSessionId = sessionId;

    // Highlight selected session
    document.querySelectorAll('.session-item').forEach(item => {
        item.classList.remove('active');
    });
    event.target.closest('.session-item').classList.add('active');

    // Clear previous results
    document.getElementById('handReview').innerHTML = '<p class="hint">Analyzing session... 🎰</p>';

    try {
        const result = await api(`/poker/analyze/${sessionId}`, { method: 'POST' });
        displayHandReview(result);
    } catch (error) {
        console.error('Failed to analyze session:', error);
        document.getElementById('handReview').innerHTML = '<p class="empty">Failed to analyze session</p>';
    }
}

function displayHandReview(result) {
    const handReview = document.getElementById('handReview');

    if (result.reviews.length === 0) {
        handReview.innerHTML = '<p class="empty">No significant hands to analyze</p>';
        return;
    }

    let html = `
        <div style="margin-bottom: 16px; padding: 12px; background: var(--bg-card); border-radius: var(--radius-sm);">
            <strong>${result.hands_analyzed} hands analyzed</strong> •
            Cost: $${result.cost.toFixed(3)}
        </div>
    `;

    result.reviews.forEach((review, idx) => {
        const severityClass = review.mistake_severity.toLowerCase().replace(' ', '-');
        const cards = review.hero_cards.join(' ');
        const board = review.board.join(' ');

        html += `
            <div class="hand-card ${severityClass}">
                <div class="hand-header">
                    <span class="hand-number">Hand #${review.hand_number}</span>
                    <span class="hand-score ${severityClass}">${review.gto_score}/100</span>
                </div>
                <div class="hand-situation">
                    ${cards} • Board: ${board} • Pot: ${review.pot_bb.toFixed(1)} BB
                </div>
                <div class="hand-analysis">${review.coach_analysis}</div>
                ${review.better_line ? `
                    <div class="hand-better-line">
                        <strong>Better Line:</strong>
                        ${review.better_line}
                    </div>
                ` : ''}
            </div>
        `;
    });

    // Add summary if available
    if (result.summary) {
        html += `
            <div style="margin-top: 16px; padding: 12px; background: var(--bg-card); border-radius: var(--radius-sm); border-left: 3px solid var(--accent);">
                <strong>Session Summary:</strong><br>
                ${result.summary.overall_grade || 'Analysis complete'}
            </div>
        `;
    }

    handReview.innerHTML = html;
}

async function profileOpponent() {
    const playerName = document.getElementById('opponentName').value.trim();
    const resultDiv = document.getElementById('opponentProfile');

    if (!playerName) {
        resultDiv.innerHTML = '<p class="empty">Enter a player name</p>';
        return;
    }

    resultDiv.innerHTML = '<p class="hint">Profiling opponent... 👤</p>';

    try {
        const profile = await api(`/poker/profile/${encodeURIComponent(playerName)}`);

        let html = `
            <div class="opponent-profile">
                <h5>${profile.player_name}</h5>
                <div class="opponent-stats">
                    <div class="opponent-stat">VPIP: <strong>${profile.vpip.toFixed(1)}%</strong></div>
                    <div class="opponent-stat">PFR: <strong>${profile.pfr.toFixed(1)}%</strong></div>
                    <div class="opponent-stat">Agg: <strong>${profile.aggression_freq.toFixed(1)}%</strong></div>
                    <div class="opponent-stat">Hands: <strong>${profile.hands_seen}</strong></div>
                </div>
                <p style="margin-top: 8px;">
                    <strong>Type:</strong> ${profile.play_style} (${profile.skill_level})<br>
                    <strong>Weakness:</strong> ${profile.key_weakness}
                </p>
                <div class="opponent-exploits">
                    <strong>Exploits:</strong>
                    <ul>
        `;

        profile.exploits.forEach(exploit => {
            html += `<li>${exploit}</li>`;
        });

        html += `
                    </ul>
                </div>
                <p class="hint" style="margin-top: 8px;">
                    Cost: $${profile.cost.toFixed(3)} •
                    Confidence: ${(profile.confidence * 100).toFixed(0)}%
                </p>
            </div>
        `;

        resultDiv.innerHTML = html;
    } catch (error) {
        console.error('Failed to profile opponent:', error);
        if (error.message.includes('404')) {
            resultDiv.innerHTML = '<p class="empty">Player not found or insufficient hands (need 10+)</p>';
        } else {
            resultDiv.innerHTML = '<p class="empty">Failed to profile opponent</p>';
        }
    }
}

async function analyzeLeaks() {
    const sessions = parseInt(document.getElementById('leakSessions').value);
    const resultDiv = document.getElementById('leakReport');

    resultDiv.innerHTML = '<p class="hint">Analyzing leaks... 🔍</p>';

    try {
        const report = await api(`/poker/leaks?last_n_sessions=${sessions}`);

        let html = `
            <div class="leak-report">
                <h5>🔍 Leak Analysis</h5>
                <p style="margin-bottom: 12px;">
                    ${report.sessions_analyzed} sessions • ${report.total_hands} hands<br>
                    <strong>Grade:</strong> ${report.overall_grade}
                </p>
        `;

        if (report.leaks.length === 0) {
            html += '<p class="empty">No significant leaks detected! 🎉</p>';
        } else {
            report.leaks.forEach(leak => {
                const severityColor = leak.severity === 'High' ? 'var(--danger)' :
                                     leak.severity === 'Medium' ? 'var(--warning)' : 'var(--text-secondary)';
                html += `
                    <div class="leak-item" style="border-left-color: ${severityColor};">
                        <div class="leak-name">${leak.leak_name}</div>
                        <div class="leak-frequency">${leak.frequency}x • ${leak.severity} severity</div>
                        <div class="leak-fix">${leak.fix_drill}</div>
                    </div>
                `;
            });

            html += `
                <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border);">
                    <strong>Focus Areas:</strong>
                    <ul style="margin: 8px 0; padding-left: 20px;">
            `;

            report.study_priorities.forEach(priority => {
                html += `<li>${priority}</li>`;
            });

            html += `
                    </ul>
                </div>
            `;
        }

        html += `
                <p class="hint" style="margin-top: 12px;">
                    Cost: $${report.cost.toFixed(3)} •
                    EV Loss: ${report.total_ev_loss.toFixed(1)} BB
                </p>
            </div>
        `;

        resultDiv.innerHTML = html;
    } catch (error) {
        console.error('Failed to analyze leaks:', error);
        resultDiv.innerHTML = '<p class="empty">Failed to analyze leaks</p>';
    }
}

// ============================================================
// Quick Actions
// ============================================================

function openCaptureModal() {
    document.getElementById('captureModal').classList.add('active');
    document.getElementById('captureText').focus();
}

function closeCaptureModal() {
    document.getElementById('captureModal').classList.remove('active');
    document.getElementById('quickCaptureForm').reset();
}

async function submitCapture(event) {
    event.preventDefault();

    const text = document.getElementById('captureText').value;
    const type = document.getElementById('captureType').value;
    const priority = document.getElementById('capturePriority').value;

    try {
        await api('/captures', {
            method: 'POST',
            body: JSON.stringify({ text, type, priority, tags: [] })
        });

        closeCaptureModal();
        loadTodayData();
        loadChallenges();
        alert('Capture saved!');
    } catch (error) {
        alert('Failed to save capture');
    }
}

async function startFocus() {
    const task = prompt('What are you focusing on?', 'work');
    if (!task) return;

    // This would integrate with focus mode API
    alert(`Focus session would start: "${task}"\n\nFocus mode integration coming soon.`);
}

async function startPoker() {
    alert('Poker mode would start here.\n\nPoker integration coming soon.');
}

// ============================================================
// Homework Tab (Phase 7)
// ============================================================

let selectedHomeworkFile = null;

// Initialize homework file input
document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('homeworkFile');
    if (fileInput) {
        fileInput.addEventListener('change', handleHomeworkFileSelect);
    }

    // Initialize KaTeX auto-render when available
    if (typeof renderMathInElement !== 'undefined') {
        initKaTeX();
    }
});

function initKaTeX() {
    // Auto-render math in elements with class 'math-display'
    document.querySelectorAll('.math-display').forEach(el => {
        renderMathInElement(el, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false},
                {left: '\\[', right: '\\]', display: true},
                {left: '\\(', right: '\\)', display: false}
            ],
            throwOnError: false
        });
    });
}

function handleHomeworkFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Validate file type and size
    if (!['image/jpeg', 'image/png'].includes(file.type)) {
        alert('Please select a JPEG or PNG image');
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        alert('File too large. Maximum size is 10MB');
        return;
    }

    selectedHomeworkFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('homeworkImage').src = e.target.result;
        document.getElementById('homeworkPreview').style.display = 'block';
        document.getElementById('homeworkSolution').style.display = 'none';
    };
    reader.readAsDataURL(file);
}

async function solveHomework() {
    if (!selectedHomeworkFile) {
        alert('Please select an image first');
        return;
    }

    const solutionDiv = document.getElementById('homeworkSolution');
    solutionDiv.style.display = 'block';

    // Show loading without destroying the structure
    let loadingEl = document.getElementById('homeworkLoading');
    if (!loadingEl) {
        loadingEl = document.createElement('div');
        loadingEl.id = 'homeworkLoading';
        loadingEl.className = 'loading-overlay';
        loadingEl.innerHTML = '<p class="hint">Analyzing and solving... 🔍</p>';
        solutionDiv.prepend(loadingEl);
    }
    loadingEl.style.display = 'block';

    try {
        const formData = new FormData();
        formData.append('file', selectedHomeworkFile);

        const response = await fetch(`${API_BASE}/homework/solve`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const result = await response.json();
        displayHomeworkSolution(result);

        // Refresh history
        loadHomeworkHistory();
        loadHomeworkStats();
    } catch (error) {
        console.error('Failed to solve homework:', error);
        // Hide loading and show error without destroying structure
        const loadingEl = document.getElementById('homeworkLoading');
        if (loadingEl) loadingEl.style.display = 'none';
        displayHomeworkSolution({ success: false, error_message: 'Failed to solve problem. Check console for details.' });
    }
}

function displayHomeworkSolution(result) {
    const solutionDiv = document.getElementById('homeworkSolution');

    // Null guard - elements may not exist if tab hasn't rendered
    if (!solutionDiv) {
        console.error('homeworkSolution element not found in DOM');
        return;
    }

    // Hide loading indicator if present
    const loadingEl = document.getElementById('homeworkLoading');
    if (loadingEl) {
        loadingEl.style.display = 'none';
    }

    if (!result.success) {
        solutionDiv.innerHTML = `
            <div class="solution-error">
                <h3>Unable to Solve</h3>
                <p>${result.error_message || 'No math equation detected in image'}</p>
            </div>
        `;
        return;
    }

    // Display problem equation
    const equationEl = document.getElementById('solutionEquation');
    if (!equationEl) {
        console.error('Solution child elements not found - are you on the Homework tab?');
        return;
    }
    equationEl.textContent = result.problem.equation;
    if (typeof katex !== 'undefined') {
        try {
            katex.render(result.problem.equation, equationEl, {
                throwOnError: false,
                displayMode: true
            });
        } catch (e) {
            equationEl.textContent = result.problem.equation;
        }
    }

    // Display problem type badge
    document.getElementById('solutionType').textContent = result.problem.problem_type.replace('_', ' ');

    // Display steps
    const stepsEl = document.getElementById('solutionSteps');
    stepsEl.innerHTML = result.solution_steps.map(step => `<li>${step}</li>`).join('');

    // Display explanation
    document.getElementById('solutionExplanation').textContent = result.concept_explanation || 'No additional context available.';

    // Display meta
    document.getElementById('solutionTool').textContent = result.tool_used;
    document.getElementById('solutionTime').textContent = `${result.execution_time_ms.toFixed(0)}ms`;

    solutionDiv.style.display = 'block';

    // Re-render KaTeX in solution
    if (typeof renderMathInElement !== 'undefined') {
        renderMathInElement(solutionDiv, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false},
                {left: '\\[', right: '\\]', display: true},
                {left: '\\(', right: '\\)', display: false}
            ],
            throwOnError: false
        });
    }
}

async function loadHomeworkHistory() {
    const historyDiv = document.getElementById('homeworkHistory');
    const typeFilter = document.getElementById('homeworkTypeFilter')?.value || '';
    const starredOnly = document.getElementById('starredOnly')?.checked || false;

    try {
        let url = `/homework/history?limit=20`;
        if (typeFilter) url += `&problem_type=${typeFilter}`;
        if (starredOnly) url += `&starred_only=true`;

        const data = await api(url);
        renderHomeworkHistory(data.entries, data.total);
    } catch (error) {
        console.error('Failed to load homework history:', error);
        historyDiv.innerHTML = '<p class="empty">Failed to load history</p>';
    }
}

function renderHomeworkHistory(entries, total) {
    const historyDiv = document.getElementById('homeworkHistory');

    if (entries.length === 0) {
        historyDiv.innerHTML = '<p class="empty">No problems solved yet. Upload an image to get started!</p>';
        return;
    }

    historyDiv.innerHTML = entries.map(entry => {
        const date = new Date(entry.timestamp * 1000);
        const dateStr = date.toLocaleDateString([], { month: 'short', day: 'numeric' });
        const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const starClass = entry.starred ? 'starred' : '';

        return `
            <div class="history-item">
                <div class="history-header">
                    <span class="history-equation">${entry.problem_latex.substring(0, 40)}${entry.problem_latex.length > 40 ? '...' : ''}</span>
                    <button class="star-btn ${starClass}" onclick="toggleHomeworkStar('${entry.id}', ${entry.starred})">
                        ${entry.starred ? '⭐' : '☆'}
                    </button>
                </div>
                <div class="history-meta">
                    <span class="history-type">${entry.problem_type.replace('_', ' ')}</span>
                    <span class="history-date">${dateStr} ${timeStr}</span>
                </div>
                <div class="history-summary">${entry.solution_summary}</div>
            </div>
        `;
    }).join('');
}

async function toggleHomeworkStar(entryId, currentStarred) {
    try {
        await fetch(`${API_BASE}/homework/history/${entryId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ starred: !currentStarred })
        });
        loadHomeworkHistory();
        loadHomeworkStats();
    } catch (error) {
        console.error('Failed to toggle star:', error);
    }
}

async function loadHomeworkStats() {
    try {
        const stats = await api('/homework/stats');

        document.getElementById('homeworkTotal').textContent = stats.total_count;
        document.getElementById('homeworkRecent').textContent = stats.recent_count;
        document.getElementById('homeworkStarred').textContent = stats.starred_count;
    } catch (error) {
        console.error('Failed to load homework stats:', error);
    }
}

// ============================================================
// Keyboard Shortcuts
// ============================================================

// Keyboard shortcut for quick capture (Ctrl/Cmd + K)
document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        openCaptureModal();
    }

    // Escape to close modal
    if (e.key === 'Escape') {
        closeCaptureModal();
    }
});

// ============================================================
// RAG Document Q&A Functions
// ============================================================

async function uploadRAGDoc() {
    const fileInput = document.getElementById('ragFile');
    const file = fileInput.files[0];
    if (!file) return;

    // Validate file type
    const allowedTypes = ['text/plain', 'application/pdf'];
    if (!allowedTypes.includes(file.type)) {
        document.getElementById('ragStatus').innerHTML =
            '<span style="color: var(--error);">Only TXT and PDF files supported</span>';
        return;
    }

    // Validate file size (10MB)
    if (file.size > 10 * 1024 * 1024) {
        document.getElementById('ragStatus').innerHTML =
            '<span style="color: var(--error);">File too large (max 10MB)</span>';
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('category', 'general');

    document.getElementById('ragStatus').innerHTML =
        '<span style="color: var(--primary);">Uploading and indexing...</span>';

    try {
        const response = await fetch(`${API_BASE}/rag/upload`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok && result.success) {
            document.getElementById('ragStatus').innerHTML =
                `<span style="color: var(--success);">✓ Indexed ${result.chars.toLocaleString()} characters</span>`;
            loadRAGStats();
        } else {
            document.getElementById('ragStatus').innerHTML =
                `<span style="color: var(--error);">Error: ${result.detail || 'Upload failed'}</span>`;
        }
    } catch (error) {
        console.error('RAG upload failed:', error);
        document.getElementById('ragStatus').innerHTML =
            `<span style="color: var(--error);">Upload failed: ${error.message}</span>`;
    }

    // Clear file input for next upload
    fileInput.value = '';
}

async function queryRAG() {
    const queryInput = document.getElementById('ragQuery');
    const query = queryInput.value.trim();
    if (!query) return;

    const answerDiv = document.getElementById('ragAnswer');
    answerDiv.innerHTML = '<p style="color: var(--primary);">Searching documents...</p>';

    try {
        const response = await fetch(`${API_BASE}/rag/query?q=${encodeURIComponent(query)}`);
        const result = await response.json();

        if (result.sources && result.sources.length > 0) {
            answerDiv.innerHTML = `
                <div style="background: var(--bg-secondary); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                    <strong style="color: var(--primary);">Answer:</strong>
                    <p style="margin-top: 8px; white-space: pre-wrap;">${escapeHtml(result.answer)}</p>
                </div>
                <div style="font-size: 0.9em;">
                    <strong>Sources:</strong>
                    <div style="margin-top: 8px; display: flex; flex-wrap: wrap; gap: 8px;">
                        ${result.sources.map(s => `
                            <span class="tag" title="${escapeHtml(s.snippet)}">
                                ${escapeHtml(s.filename)} (${Math.round(s.score * 100)}%)
                            </span>
                        `).join('')}
                    </div>
                </div>
            `;
        } else {
            answerDiv.innerHTML = `
                <div style="background: var(--bg-secondary); padding: 15px; border-radius: 8px;">
                    <p>${escapeHtml(result.answer)}</p>
                </div>
            `;
        }
    } catch (error) {
        console.error('RAG query failed:', error);
        answerDiv.innerHTML =
            `<p style="color: var(--error);">Query failed: ${error.message}</p>`;
    }
}

async function loadRAGStats() {
    try {
        const response = await fetch(`${API_BASE}/rag/stats`);
        const stats = await response.json();

        const docCountEl = document.getElementById('ragDocCount');
        if (docCountEl) {
            docCountEl.innerHTML = `${stats.total_documents} document${stats.total_documents !== 1 ? 's' : ''} indexed`;
        }
    } catch (error) {
        console.error('Failed to load RAG stats:', error);
    }
}

// Helper function to escape HTML
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Load RAG stats when Memory tab is activated
const originalLoadMemoryStats = typeof loadMemoryStats === 'function' ? loadMemoryStats : null;
if (originalLoadMemoryStats) {
    loadMemoryStats = async function() {
        await originalLoadMemoryStats();
        await loadRAGStats();
    };
}

// ============================================================
// AGENT FUNCTIONS (Phase 9)
// ============================================================

async function sendAgentMessage() {
    const input = document.getElementById('agentInput');
    const message = input.value.trim();
    if (!message) return;

    // Add user message to history
    addAgentMessage('user', message);
    input.value = '';

    // Show thinking indicator
    const thinkingId = addAgentMessage('assistant', '<span style="color: var(--primary);">🤔 Thinking...</span>');

    try {
        const response = await fetch(`${API_BASE}/agent/chat`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message})
        });
        const result = await response.json();

        // Remove thinking indicator
        const thinkingEl = document.getElementById(thinkingId);
        if (thinkingEl) thinkingEl.remove();

        // Build response HTML
        let responseHtml = escapeHtml(result.response);

        // Add tool badge if tool was used
        if (result.tool_used) {
            responseHtml = `<span class="tool-badge" style="background: var(--primary); color: var(--bg); padding: 2px 8px; border-radius: 4px; font-size: 0.8em; margin-right: 8px;">🔧 ${escapeHtml(result.tool_used)}</span>${responseHtml}`;
        }

        // Add success/error indicator
        if (result.success === false) {
            responseHtml = `<span style="color: var(--error);">❌</span> ${responseHtml}`;
        }

        addAgentMessage('assistant', responseHtml);

    } catch (error) {
        // Remove thinking indicator
        const thinkingEl = document.getElementById(thinkingId);
        if (thinkingEl) thinkingEl.remove();

        addAgentMessage('assistant', `<span style="color: var(--error);">❌ Error: ${escapeHtml(error.message)}</span>`);
    }
}

function addAgentMessage(role, content) {
    const history = document.getElementById('agentHistory');
    const id = 'agent-msg-' + Date.now();

    const msgDiv = document.createElement('div');
    msgDiv.id = id;
    msgDiv.className = `agent-msg agent-msg-${role}`;
    msgDiv.innerHTML = content;

    // Style based on role
    if (role === 'user') {
        msgDiv.style.cssText = 'background: var(--primary); color: var(--bg); align-self: flex-end; padding: 10px 15px; border-radius: 15px 15px 5px 15px; max-width: 80%; margin: 5px 0;';
    } else {
        msgDiv.style.cssText = 'background: var(--bg-secondary); padding: 10px 15px; border-radius: 15px 15px 15px 5px; max-width: 80%; margin: 5px 0;';
    }

    history.appendChild(msgDiv);
    history.scrollTop = history.scrollHeight;

    return id;
}

async function loadAgentTools() {
    try {
        const response = await fetch(`${API_BASE}/agent/tools`);
        const data = await response.json();

        const toolsDiv = document.getElementById('agentTools');
        if (!toolsDiv) return;

        if (data.tools && data.tools.length > 0) {
            toolsDiv.innerHTML = data.tools.map(t => `
                <div style="padding: 10px; background: var(--bg-secondary); border-radius: 8px; margin-bottom: 8px;">
                    <strong style="color: var(--primary);">🔧 ${escapeHtml(t.name)}</strong>
                    <p style="margin: 5px 0 0 0; font-size: 0.9em; color: var(--text-dim);">${escapeHtml(t.description)}</p>
                </div>
            `).join('');
        } else {
            toolsDiv.innerHTML = '<p class="hint">No tools available</p>';
        }
    } catch (error) {
        console.error('Failed to load agent tools:', error);
        const toolsDiv = document.getElementById('agentTools');
        if (toolsDiv) {
            toolsDiv.innerHTML = `<p style="color: var(--error);">Failed to load tools</p>`;
        }
    }
}

async function loadCalendarStatus() {
    try {
        const response = await fetch(`${API_BASE}/agent/calendar/status`);
        const data = await response.json();

        const statusDiv = document.getElementById('calendarStatus');
        if (!statusDiv) return;

        if (data.authenticated) {
            statusDiv.innerHTML = `
                <span class="status-indicator" style="background: var(--success);"></span>
                <span style="color: var(--success);">Calendar connected</span>
            `;
        } else {
            statusDiv.innerHTML = `
                <span class="status-indicator" style="background: var(--warning);"></span>
                <span style="color: var(--warning);">${escapeHtml(data.message)}</span>
            `;
        }
    } catch (error) {
        console.error('Failed to load calendar status:', error);
        const statusDiv = document.getElementById('calendarStatus');
        if (statusDiv) {
            statusDiv.innerHTML = `
                <span class="status-indicator" style="background: var(--error);"></span>
                <span style="color: var(--error);">Status check failed</span>
            `;
        }
    }
}

async function resetAgentChat() {
    try {
        await fetch(`${API_BASE}/agent/reset`, { method: 'POST' });

        // Clear chat history and add welcome message
        const history = document.getElementById('agentHistory');
        if (history) {
            history.innerHTML = `
                <div class="agent-msg agent-msg-assistant" style="background: var(--bg-secondary); padding: 10px 15px; border-radius: 15px 15px 15px 5px; max-width: 80%; margin: 5px 0;">
                    Hello! I can help you manage your calendar, search documents, and more. Try saying "What's on my calendar today?" or "Schedule a meeting at 2pm tomorrow".
                </div>
            `;
        }
    } catch (error) {
        console.error('Failed to reset agent chat:', error);
    }
}

// Load agent data when Agent tab is activated
function loadAgentTab() {
    loadAgentTools();
    loadCalendarStatus();
}

// Hook into tab switching to load agent data
const originalTabClick = document.querySelector('.tabs')?.addEventListener;
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', function() {
        if (this.dataset.tab === 'agent') {
            setTimeout(loadAgentTab, 100);
        }
    });
});
