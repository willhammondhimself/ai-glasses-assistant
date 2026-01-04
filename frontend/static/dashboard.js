/**
 * WHAM Dashboard - JavaScript
 * Handles all API interactions and UI updates
 */

// API Base URL
const API_BASE = '/dashboard/api';

// State
let currentCapturePage = 1;
const capturesPerPage = 20;

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

    // Set up auto-refresh
    setInterval(loadHealth, 30000); // Every 30 seconds
    setInterval(loadTodayData, 60000); // Every minute

    // Set up event listeners
    document.getElementById('captureSearch').addEventListener('input', debounce(loadCaptures, 300));
    document.getElementById('captureFilter').addEventListener('change', loadCaptures);
    document.getElementById('memoryQuery').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') searchMemory();
    });
});

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

function renderCostChart(days) {
    const canvas = document.getElementById('costChart');
    const ctx = canvas.getContext('2d');

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Simple bar chart
    const padding = 40;
    const width = canvas.width - padding * 2;
    const height = canvas.height - padding * 2;

    // Reverse to show oldest first
    const sortedDays = [...days].reverse();

    const maxCost = Math.max(...sortedDays.map(d => d.stats.total_cost), 5);
    const barWidth = width / sortedDays.length - 10;

    // Draw bars
    sortedDays.forEach((day, i) => {
        const x = padding + i * (barWidth + 10);
        const barHeight = (day.stats.total_cost / maxCost) * height;
        const y = padding + height - barHeight;

        // Bar
        ctx.fillStyle = '#6366f1';
        ctx.fillRect(x, y, barWidth, barHeight);

        // Label
        ctx.fillStyle = '#a0a0b0';
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(formatDate(day.date), x + barWidth / 2, canvas.height - 10);
    });

    // Y axis labels
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
