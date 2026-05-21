const sportSelect = document.getElementById("sport-select");
const loadMatchesBtn = document.getElementById("load-matches-btn");
const leagueFilter = document.getElementById("league-filter");
const teamFilter = document.getElementById("team-filter");
const refreshIntervalSelect = document.getElementById("refresh-interval");
const upcomingViewModeSelect = document.getElementById("upcoming-view-mode");
const upcomingBody = document.getElementById("upcoming-body");
const upcomingTableWrap = document.getElementById("upcoming-table-wrap");
const upcomingCards = document.getElementById("upcoming-cards");
const upcomingMeta = document.getElementById("upcoming-meta");
const demoBanner = document.getElementById("demo-banner");

const historySport = document.getElementById("history-sport");
const historyLeague = document.getElementById("history-league");
const historyTeam = document.getElementById("history-team");
const historyLimit = document.getElementById("history-limit");
const loadHistoryBtn = document.getElementById("load-history-btn");
const clearHistoryBtn = document.getElementById("clear-history-btn");
const historyBody = document.getElementById("history-body");
const historyMeta = document.getElementById("history-meta");
const historyChart = document.getElementById("history-chart");

const whatIfForm = document.getElementById("what-if-form");
const whatIfResult = document.getElementById("what-if-result");
const whatIfSportSelect = document.getElementById("whatif-sport");
const whatIfSection = document.getElementById("what-if-section");

const injuryForm = document.getElementById("injury-form");
const injurySportSelect = document.getElementById("injury-sport");
const injuryTeamInput = document.getElementById("injury-team");
const injuryRatingInput = document.getElementById("injury-rating");
const injuryFormInput = document.getElementById("injury-form-delta");
const injuryOffenseInput = document.getElementById("injury-offense");
const injuryDefenseInput = document.getElementById("injury-defense");
const injuryNotesInput = document.getElementById("injury-notes");
const injuryBody = document.getElementById("injury-body");
const injuryMeta = document.getElementById("injury-meta");
const deleteInjuryBtn = document.getElementById("delete-injury-btn");
const clearSportInjuriesBtn = document.getElementById("clear-sport-injuries-btn");

const modelMeta = document.getElementById("model-meta");
const modelMetricsBody = document.getElementById("model-metrics-body");
const featureImportanceWrap = document.getElementById("feature-importance-wrap");

const kpiTotalMatches = document.getElementById("kpi-total-matches");
const kpiAvgConfidence = document.getElementById("kpi-avg-confidence");
const kpiTopLeague = document.getElementById("kpi-top-league");
const kpiModel = document.getElementById("kpi-model");
const compareTeamASelect = document.getElementById("compare-team-a");
const compareTeamBSelect = document.getElementById("compare-team-b");
const compareTeamsBtn = document.getElementById("compare-teams-btn");
const teamCompareResult = document.getElementById("team-compare-result");

let allUpcomingMatches = [];
let autoRefreshHandle = null;

function pct(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function asLocalTimestamp(utcValue) {
  const parsed = new Date(utcValue);
  if (Number.isNaN(parsed.getTime())) {
    return utcValue;
  }
  return parsed.toLocaleString();
}

function asUtcTimestamp(utcValue) {
  const parsed = new Date(utcValue);
  if (Number.isNaN(parsed.getTime())) {
    return utcValue || "";
  }
  return parsed.toISOString().replace("T", " ").replace("Z", "");
}

function readNumberFromElement(element, fallback = 0) {
  const value = Number(element.value);
  return Number.isFinite(value) ? value : fallback;
}

function readNumericInput(id, fallback) {
  const value = Number(document.getElementById(id).value);
  return Number.isFinite(value) ? value : fallback;
}

function setUpcomingMeta(text) {
  upcomingMeta.textContent = text;
}

function setHistoryMeta(text) {
  historyMeta.textContent = text;
}

function setInjuryMeta(text) {
  injuryMeta.textContent = text;
}

function setModelMeta(text) {
  modelMeta.textContent = text;
}

function parsePredictedScore(score) {
  if (!score || typeof score !== "string" || !score.includes("-")) {
    return null;
  }
  const [left, right] = score.split("-");
  const home = Number(left);
  const away = Number(right);
  if (!Number.isFinite(home) || !Number.isFinite(away)) {
    return null;
  }
  return { home, away };
}

function confidenceBar(value) {
  const p = Math.max(0, Math.min(100, Number(value) * 100));
  return `
    <div class="confidence-bar">
      <div class="confidence-bar-fill" style="width:${p.toFixed(1)}%"></div>
    </div>
  `;
}

function renderHistoryChart(rows) {
  if (!rows.length) {
    historyChart.innerHTML = `<div class="history-empty">No history points yet.</div>`;
    return;
  }
  const recent = rows.slice(0, 20).reverse();
  const bars = recent.map((row) => {
    const c = Number(row.confidence || 0) * 100;
    return `<div class="history-bar" title="${row.home_team} vs ${row.away_team} (${c.toFixed(1)}%)">
      <span style="height:${Math.max(8, c).toFixed(1)}%"></span>
    </div>`;
  });
  historyChart.innerHTML = `<div class="history-bars">${bars.join("")}</div>`;
}

function updateKpis(matches, training) {
  kpiTotalMatches.textContent = String(matches.length);
  if (!matches.length) {
    kpiAvgConfidence.textContent = "-";
    kpiTopLeague.textContent = "-";
  } else {
    const avg = matches.reduce((acc, m) => acc + Number(m.confidence || 0), 0) / matches.length;
    kpiAvgConfidence.textContent = pct(avg);
    const leagueCounts = {};
    matches.forEach((m) => {
      leagueCounts[m.league] = (leagueCounts[m.league] || 0) + 1;
    });
    const topLeague = Object.entries(leagueCounts).sort((a, b) => b[1] - a[1])[0];
    kpiTopLeague.textContent = topLeague ? `${topLeague[0]} (${topLeague[1]})` : "-";
  }
  kpiModel.textContent = training?.best_model || "heuristic";
}

function fillWhatIfFromMatch(match) {
  document.getElementById("home-team").value = match.home_team;
  document.getElementById("away-team").value = match.away_team;
  whatIfSportSelect.value = match.sport;

  if (match.factors) {
    document.getElementById("home-rating").value = Math.round(Number(match.factors.home_rating || 1500));
    document.getElementById("away-rating").value = Math.round(Number(match.factors.away_rating || 1500));
    document.getElementById("home-form").value = Number(match.factors.home_form ?? 0.5).toFixed(2);
    document.getElementById("away-form").value = Number(match.factors.away_form ?? 0.5).toFixed(2);
  }

  const parsedScore = parsePredictedScore(match.predicted_score);
  if (parsedScore) {
    document.getElementById("home-scored").value = parsedScore.home;
    document.getElementById("away-scored").value = parsedScore.away;
    document.getElementById("home-allowed").value = parsedScore.away;
    document.getElementById("away-allowed").value = parsedScore.home;
  }

  document.getElementById("neutral-site").checked = false;
  document.getElementById("ignore-injuries").checked = false;
}

function fillInjuryFormFromRow(row) {
  injurySportSelect.value = row.sport;
  injuryTeamInput.value = row.team || "";
  injuryRatingInput.value = Number(row.rating_delta || 0);
  injuryFormInput.value = Number(row.form_delta || 0);
  injuryOffenseInput.value = Number(row.offense_delta || 0);
  injuryDefenseInput.value = Number(row.defense_delta || 0);
  injuryNotesInput.value = row.notes || "";
}

function updateLeagueFilterOptions(matches, sport) {
  const currentValue = leagueFilter.value;
  const leagues = [...new Set(matches.map((item) => item.league).filter(Boolean))].sort((a, b) => a.localeCompare(b));
  leagueFilter.innerHTML = `<option value="">All leagues</option>`;
  leagues.forEach((league) => {
    const option = document.createElement("option");
    option.value = league;
    option.textContent = league;
    leagueFilter.appendChild(option);
  });

  if (leagues.includes(currentValue)) {
    leagueFilter.value = currentValue;
    return;
  }
  if (sport === "basketball") {
    const nbaLeague = leagues.find((league) => {
      const text = league.toLowerCase();
      return text === "nba" || text.includes("national basketball association");
    });
    if (nbaLeague) {
      leagueFilter.value = nbaLeague;
    }
  }
}

function filterUpcomingMatches() {
  const league = leagueFilter.value.trim().toLowerCase();
  const teamTerm = teamFilter.value.trim().toLowerCase();
  return allUpcomingMatches.filter((match) => {
    const leaguePass = !league || (match.league || "").toLowerCase() === league;
    const teamsJoined = `${match.home_team} ${match.away_team}`.toLowerCase();
    const teamPass = !teamTerm || teamsJoined.includes(teamTerm);
    return leaguePass && teamPass;
  });
}

function updateTeamComparisonOptions(matches) {
  const teams = [...new Set(matches.flatMap((m) => [m.home_team, m.away_team]).filter(Boolean))].sort((a, b) =>
    a.localeCompare(b),
  );
  const currentA = compareTeamASelect.value;
  const currentB = compareTeamBSelect.value;

  compareTeamASelect.innerHTML = `<option value="">Select team</option>${teams.map((t) => `<option value="${t}">${t}</option>`).join("")}`;
  compareTeamBSelect.innerHTML = `<option value="">Select team</option>${teams.map((t) => `<option value="${t}">${t}</option>`).join("")}`;

  if (teams.includes(currentA)) compareTeamASelect.value = currentA;
  if (teams.includes(currentB)) compareTeamBSelect.value = currentB;
}

function renderUpcomingCards(matches) {
  if (!matches.length) {
    upcomingCards.innerHTML = `<div class="history-empty">No matches match the current filters.</div>`;
    return;
  }
  upcomingCards.innerHTML = matches
    .map(
      (match) => `
      <article class="match-card">
        <h3>${match.home_team} vs ${match.away_team}</h3>
        <p><strong>${asLocalTimestamp(match.kickoff_utc)}</strong> | ${match.league}</p>
        <p>Venue: ${match.venue || "Unknown Venue"}</p>
        <p class="winner">Winner: ${match.predicted_winner}</p>
        <p class="confidence">Confidence: ${pct(match.confidence)}</p>
        ${confidenceBar(match.confidence)}
        <p>Rough Score: ${match.predicted_score}</p>
        <p><button type="button" class="use-match-btn" data-event-id="${match.event_id}">Use In What-If</button></p>
      </article>
    `,
    )
    .join("");
}

function applyUpcomingViewMode() {
  const mode = upcomingViewModeSelect.value;
  if (mode === "cards") {
    upcomingTableWrap.classList.add("hidden");
    upcomingCards.classList.remove("hidden");
  } else {
    upcomingTableWrap.classList.remove("hidden");
    upcomingCards.classList.add("hidden");
  }
}

function renderUpcomingMatches() {
  const matches = filterUpcomingMatches();
  if (!matches.length) {
    upcomingBody.innerHTML = `<tr><td colspan="9">No matches match the current filters.</td></tr>`;
    renderUpcomingCards([]);
    updateTeamComparisonOptions([]);
    teamCompareResult.innerHTML = `<div class="history-empty">No teams available for comparison with current filters.</div>`;
    applyUpcomingViewMode();
    return;
  }

  const rows = matches.map((match) => `
      <tr>
        <td>${asLocalTimestamp(match.kickoff_utc)}</td>
        <td>${match.league}</td>
        <td>${match.home_team} vs ${match.away_team}</td>
        <td>${match.venue || "Unknown Venue"}</td>
        <td class="winner">${match.predicted_winner}</td>
        <td class="confidence">${pct(match.confidence)}</td>
        <td>${confidenceBar(match.confidence)}</td>
        <td>${match.predicted_score}</td>
        <td><button type="button" class="use-match-btn" data-event-id="${match.event_id}">Use</button></td>
      </tr>
    `);
  upcomingBody.innerHTML = rows.join("");
  renderUpcomingCards(matches);
  updateTeamComparisonOptions(matches);
  applyUpcomingViewMode();

  const buttons = document.querySelectorAll(".use-match-btn");
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const eventId = button.getAttribute("data-event-id");
      const selected = allUpcomingMatches.find((item) => String(item.event_id) === String(eventId));
      if (!selected) return;
      fillWhatIfFromMatch(selected);
      if (whatIfSection) {
        whatIfSection.scrollIntoView({ behavior: "smooth", block: "start" });
      }
      setTimeout(() => whatIfForm.requestSubmit(), 120);
    });
  });
}

function compareTeamsFromCurrentData() {
  const teamA = compareTeamASelect.value;
  const teamB = compareTeamBSelect.value;
  if (!teamA || !teamB) {
    teamCompareResult.innerHTML = `<div class="history-empty">Select both teams to compare.</div>`;
    return;
  }
  if (teamA === teamB) {
    teamCompareResult.innerHTML = `<div class="history-empty">Select two different teams.</div>`;
    return;
  }

  const relevant = allUpcomingMatches.filter(
    (m) => m.home_team === teamA || m.away_team === teamA || m.home_team === teamB || m.away_team === teamB,
  );
  const summarize = (team) => {
    const teamMatches = relevant.filter((m) => m.home_team === team || m.away_team === team);
    const avgConfidence =
      teamMatches.length > 0
        ? teamMatches.reduce((acc, row) => acc + Number(row.confidence || 0), 0) / teamMatches.length
        : 0;
    const predictedWins = teamMatches.filter((m) => m.predicted_winner === team).length;
    return {
      matchCount: teamMatches.length,
      predictedWins,
      avgConfidence,
      nextGame:
        teamMatches
          .map((m) => ({ ...m, kickoffTs: Date.parse(m.kickoff_utc) || Number.MAX_SAFE_INTEGER }))
          .sort((a, b) => a.kickoffTs - b.kickoffTs)[0] || null,
    };
  };

  const a = summarize(teamA);
  const b = summarize(teamB);
  teamCompareResult.innerHTML = `
    <article class="compare-card">
      <h4>${teamA}</h4>
      <div class="compare-row"><span>Upcoming Matches</span><strong>${a.matchCount}</strong></div>
      <div class="compare-row"><span>Predicted Wins</span><strong>${a.predictedWins}</strong></div>
      <div class="compare-row"><span>Avg Confidence</span><strong>${pct(a.avgConfidence)}</strong></div>
      <div class="compare-row"><span>Next Match</span><strong>${a.nextGame ? `${a.nextGame.home_team} vs ${a.nextGame.away_team}` : "-"}</strong></div>
    </article>
    <article class="compare-card">
      <h4>${teamB}</h4>
      <div class="compare-row"><span>Upcoming Matches</span><strong>${b.matchCount}</strong></div>
      <div class="compare-row"><span>Predicted Wins</span><strong>${b.predictedWins}</strong></div>
      <div class="compare-row"><span>Avg Confidence</span><strong>${pct(b.avgConfidence)}</strong></div>
      <div class="compare-row"><span>Next Match</span><strong>${b.nextGame ? `${b.nextGame.home_team} vs ${b.nextGame.away_team}` : "-"}</strong></div>
    </article>
  `;
}

function renderModelSummary(payload) {
  if (!payload || payload.status !== "ok" || !payload.summary) {
    modelMetricsBody.innerHTML = `<tr><td colspan="2">Model summary not available yet.</td></tr>`;
    featureImportanceWrap.innerHTML = "";
    return;
  }
  const summary = payload.summary;
  const metricsRows = [
    ["Best Model", payload.summary.best_model || "-"],
    ["Trained At", payload.summary.trained_at_utc || "-"],
    ["CV Accuracy", pct(Number(summary.cv_metrics?.accuracy || 0))],
    ["CV Log Loss", Number(summary.cv_metrics?.log_loss || 0).toFixed(4)],
    ["CV Brier", Number(summary.cv_metrics?.brier || 0).toFixed(4)],
    ["CV ECE", Number(summary.cv_metrics?.ece || 0).toFixed(4)],
  ];
  modelMetricsBody.innerHTML = metricsRows.map((row) => `<tr><td>${row[0]}</td><td>${row[1]}</td></tr>`).join("");

  const top = summary.feature_importance || [];
  if (!top.length) {
    featureImportanceWrap.innerHTML = "";
    return;
  }
  const maxVal = Math.max(...top.map((v) => Number(v.importance || 0)), 0.0001);
  featureImportanceWrap.innerHTML = top
    .slice(0, 8)
    .map(
      (item) => `
      <div class="feature-row">
        <span>${item.feature}</span>
        <div class="feature-bar"><i style="width:${((Number(item.importance) / maxVal) * 100).toFixed(1)}%"></i></div>
        <em>${Number(item.importance).toFixed(4)}</em>
      </div>
    `,
    )
    .join("");
}

async function loadModelSummary({ silent = false } = {}) {
  if (!silent) {
    setModelMeta("Loading model summary...");
  }
  try {
    const sport = sportSelect.value;
    const response = await fetch(`/api/model-summary?sport=${encodeURIComponent(sport)}`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Failed to load model summary.");
    }
    renderModelSummary(payload);
    setModelMeta(payload.status === "ok" ? `Model summary loaded for ${sport.replace("_", " ")}.` : "Model summary not available yet.");
  } catch (error) {
    setModelMeta(error.message);
    renderModelSummary(null);
  }
}

async function loadUpcomingMatches({ force = false, silent = false } = {}) {
  const sport = sportSelect.value;
  if (!silent) {
    setUpcomingMeta("Loading upcoming matches...");
    loadMatchesBtn.disabled = true;
  }
  try {
    const response = await fetch(`/api/upcoming?sport=${encodeURIComponent(sport)}&force=${force ? "1" : "0"}`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Failed to load matches.");
    }

    allUpcomingMatches = payload.matches || [];
    updateLeagueFilterOptions(allUpcomingMatches, sport);
    renderUpcomingMatches();
    updateKpis(allUpcomingMatches, payload.training || {});

    if (payload.demo_mode) {
      demoBanner.classList.remove("hidden");
    } else {
      demoBanner.classList.add("hidden");
    }

    const filteredCount = filterUpcomingMatches().length;
    const updatedAt = asLocalTimestamp(payload.updated_at_utc);
    const training = payload.training || {};
    const injuryCount = Number(payload.injury_adjustments_count || 0);
    let trainingSummary = "Model: heuristic fallback";
    if (training.status === "ok") {
      trainingSummary = `Model: ${training.best_model} | Holdout: ${(Number(training.holdout_accuracy || 0) * 100).toFixed(1)}% | CV: ${(Number(training.cv_accuracy || 0) * 100).toFixed(1)}%`;
    } else if (training.status) {
      trainingSummary = `Model status: ${training.status}`;
    }

    setUpcomingMeta(
      `Showing ${filteredCount} of ${allUpcomingMatches.length}. Snapshot #${payload.snapshot_id ?? "-"} | Updated: ${updatedAt} | ${trainingSummary} | Injury rules: ${injuryCount} | Data mode: ${payload.data_mode}`,
    );
    whatIfSportSelect.value = sport;
    historySport.value = sport;
    injurySportSelect.value = sport;
    await loadHistory({ silent: true });
    await loadInjuries({ silent: true });
    await loadModelSummary({ silent: true });
  } catch (error) {
    setUpcomingMeta(error.message);
    upcomingBody.innerHTML = `<tr><td colspan="9">${error.message}</td></tr>`;
  } finally {
    if (!silent) {
      loadMatchesBtn.disabled = false;
    }
  }
}

function setAutoRefresh() {
  if (autoRefreshHandle) {
    clearInterval(autoRefreshHandle);
    autoRefreshHandle = null;
  }
  const minutes = Number(refreshIntervalSelect.value);
  if (!Number.isFinite(minutes) || minutes <= 0) return;
  autoRefreshHandle = setInterval(() => loadUpcomingMatches({ force: true, silent: true }), minutes * 60 * 1000);
}

function renderHistory(rows) {
  if (!rows.length) {
    historyBody.innerHTML = `<tr><td colspan="8">No saved predictions found.</td></tr>`;
    renderHistoryChart([]);
    return;
  }
  historyBody.innerHTML = rows
    .map((row) => `
        <tr>
          <td>${asUtcTimestamp(row.created_at_utc)}</td>
          <td>${asUtcTimestamp(row.kickoff_utc)}</td>
          <td>${row.sport.replace("_", " ")}</td>
          <td>${row.league || ""}</td>
          <td>${row.home_team || ""} vs ${row.away_team || ""}</td>
          <td class="winner">${row.predicted_winner || ""}</td>
          <td class="confidence">${pct(row.confidence || 0)}</td>
          <td>${row.predicted_score || ""}</td>
        </tr>
      `)
    .join("");
  renderHistoryChart(rows);
}

async function loadHistory({ silent = false } = {}) {
  if (!silent) {
    setHistoryMeta("Loading history...");
    loadHistoryBtn.disabled = true;
  }

  const params = new URLSearchParams();
  if (historySport.value) params.set("sport", historySport.value);
  if (historyTeam.value.trim()) params.set("team", historyTeam.value.trim());
  if (historyLeague.value.trim()) params.set("league", historyLeague.value.trim());
  params.set("limit", historyLimit.value);

  try {
    const response = await fetch(`/api/history?${params.toString()}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Failed to load history.");
    renderHistory(payload.rows || []);
    setHistoryMeta(`Loaded ${payload.count} rows.`);
  } catch (error) {
    setHistoryMeta(error.message);
    historyBody.innerHTML = `<tr><td colspan="8">${error.message}</td></tr>`;
    renderHistoryChart([]);
  } finally {
    if (!silent) loadHistoryBtn.disabled = false;
  }
}

async function clearHistory() {
  const targetSport = historySport.value;
  const scope = targetSport ? `for ${targetSport.replace("_", " ")}` : "for all sports";
  if (!window.confirm(`Clear saved prediction history ${scope}?`)) return;
  try {
    const response = await fetch("/api/history", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sport: targetSport || null }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Failed to clear history.");
    setHistoryMeta(`Deleted ${payload.deleted_rows} rows.`);
    await loadHistory({ silent: true });
  } catch (error) {
    setHistoryMeta(error.message);
  }
}

function renderWhatIfResult(result) {
  const confidence = Number(result.confidence || 0);
  const interpretation = confidence < 0.5 ? "Top pick in a close match" : "Strongest projected outcome";
  whatIfResult.classList.remove("hidden");
  whatIfResult.innerHTML = `
    <h3>What-If Result</h3>
    <div class="result-row">
      <div><strong>Winner:</strong> ${result.predicted_winner}</div>
      <div><strong>Confidence:</strong> ${pct(confidence)} (${interpretation})</div>
      <div><strong>Rough Score:</strong> ${result.predicted_score}</div>
      <div><strong>Home Win:</strong> ${pct(result.home_win_probability)}</div>
      <div><strong>Draw:</strong> ${pct(result.draw_probability)}</div>
      <div><strong>Away Win:</strong> ${pct(result.away_win_probability)}</div>
    </div>
  `;
}

async function submitWhatIf(event) {
  event.preventDefault();
  const payload = {
    sport: document.getElementById("whatif-sport").value,
    home_team: document.getElementById("home-team").value.trim(),
    away_team: document.getElementById("away-team").value.trim(),
    home_rating: readNumericInput("home-rating", 1500),
    away_rating: readNumericInput("away-rating", 1500),
    home_form: readNumericInput("home-form", 0.5),
    away_form: readNumericInput("away-form", 0.5),
    home_avg_scored: readNumericInput("home-scored", 20),
    away_avg_scored: readNumericInput("away-scored", 20),
    home_avg_allowed: readNumericInput("home-allowed", 20),
    away_avg_allowed: readNumericInput("away-allowed", 20),
    neutral_site: document.getElementById("neutral-site").checked,
    ignore_injuries: document.getElementById("ignore-injuries").checked,
  };

  try {
    const response = await fetch("/api/what-if", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "What-if simulation failed.");
    renderWhatIfResult(result);
  } catch (error) {
    whatIfResult.classList.remove("hidden");
    whatIfResult.innerHTML = `<h3>What-If Result</h3><p>${error.message}</p>`;
  }
}

function renderInjuries(rows) {
  if (!rows.length) {
    injuryBody.innerHTML = `<tr><td colspan="8">No injury adjustments saved.</td></tr>`;
    return;
  }
  injuryBody.innerHTML = rows
    .map(
      (row) => `
      <tr class="injury-row" data-sport="${row.sport}" data-team="${row.team}">
        <td>${asUtcTimestamp(row.updated_at_utc)}</td>
        <td>${row.sport}</td>
        <td>${row.team}</td>
        <td>${Number(row.rating_delta || 0).toFixed(1)}</td>
        <td>${Number(row.form_delta || 0).toFixed(2)}</td>
        <td>${Number(row.offense_delta || 0).toFixed(2)}</td>
        <td>${Number(row.defense_delta || 0).toFixed(2)}</td>
        <td>${row.notes || ""}</td>
      </tr>
    `,
    )
    .join("");

  injuryBody.querySelectorAll(".injury-row").forEach((rowEl) => {
    rowEl.addEventListener("click", () => {
      const sport = rowEl.dataset.sport || "";
      const team = rowEl.dataset.team || "";
      const row = rows.find((item) => item.sport === sport && item.team === team);
      if (row) fillInjuryFormFromRow(row);
    });
  });
}

async function loadInjuries({ silent = false } = {}) {
  if (!silent) setInjuryMeta("Loading injury adjustments...");
  const sport = injurySportSelect.value;
  try {
    const response = await fetch(`/api/injuries?sport=${encodeURIComponent(sport)}&limit=500`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Failed to load injury adjustments.");
    renderInjuries(payload.rows || []);
    setInjuryMeta(`Loaded ${payload.count} injury adjustment rows for ${sport.replace("_", " ")}.`);
  } catch (error) {
    setInjuryMeta(error.message);
    injuryBody.innerHTML = `<tr><td colspan="8">${error.message}</td></tr>`;
  }
}

async function saveInjury(event) {
  event.preventDefault();
  const sport = injurySportSelect.value;
  const team = injuryTeamInput.value.trim();
  if (!team) {
    setInjuryMeta("Team is required.");
    return;
  }
  const payload = {
    sport,
    team,
    rating_delta: readNumberFromElement(injuryRatingInput, 0),
    form_delta: readNumberFromElement(injuryFormInput, 0),
    offense_delta: readNumberFromElement(injuryOffenseInput, 0),
    defense_delta: readNumberFromElement(injuryDefenseInput, 0),
    notes: injuryNotesInput.value.trim(),
  };
  try {
    const response = await fetch("/api/injuries", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Failed to save injury adjustment.");
    setInjuryMeta(`Saved injury adjustment for ${team}.`);
    await loadInjuries({ silent: true });
    await loadUpcomingMatches({ force: true, silent: true });
  } catch (error) {
    setInjuryMeta(error.message);
  }
}

async function deleteCurrentTeamInjury() {
  const sport = injurySportSelect.value;
  const team = injuryTeamInput.value.trim();
  if (!team) {
    setInjuryMeta("Enter a team name to delete.");
    return;
  }
  if (!window.confirm(`Delete injury adjustment for ${team} in ${sport.replace("_", " ")}?`)) return;
  try {
    const response = await fetch("/api/injuries", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sport, team }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Failed to delete injury adjustment.");
    setInjuryMeta(`Deleted ${result.deleted_rows} row(s).`);
    await loadInjuries({ silent: true });
    await loadUpcomingMatches({ force: true, silent: true });
  } catch (error) {
    setInjuryMeta(error.message);
  }
}

async function clearSportInjuries() {
  const sport = injurySportSelect.value;
  if (!window.confirm(`Clear all injury adjustments for ${sport.replace("_", " ")}?`)) return;
  try {
    const response = await fetch("/api/injuries", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sport }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Failed to clear injury adjustments.");
    setInjuryMeta(`Deleted ${result.deleted_rows} row(s) for ${sport.replace("_", " ")}.`);
    await loadInjuries({ silent: true });
    await loadUpcomingMatches({ force: true, silent: true });
  } catch (error) {
    setInjuryMeta(error.message);
  }
}

loadMatchesBtn.addEventListener("click", () => loadUpcomingMatches({ force: true }));
leagueFilter.addEventListener("change", renderUpcomingMatches);
teamFilter.addEventListener("input", renderUpcomingMatches);
refreshIntervalSelect.addEventListener("change", setAutoRefresh);
upcomingViewModeSelect.addEventListener("change", applyUpcomingViewMode);

loadHistoryBtn.addEventListener("click", () => loadHistory({ silent: false }));
clearHistoryBtn.addEventListener("click", clearHistory);
historySport.addEventListener("change", () => loadHistory({ silent: true }));
historyLeague.addEventListener("input", () => loadHistory({ silent: true }));
historyTeam.addEventListener("input", () => loadHistory({ silent: true }));
historyLimit.addEventListener("change", () => loadHistory({ silent: true }));

whatIfForm.addEventListener("submit", submitWhatIf);
injuryForm.addEventListener("submit", saveInjury);
injurySportSelect.addEventListener("change", () => loadInjuries({ silent: false }));
deleteInjuryBtn.addEventListener("click", deleteCurrentTeamInjury);
clearSportInjuriesBtn.addEventListener("click", clearSportInjuries);
compareTeamsBtn.addEventListener("click", compareTeamsFromCurrentData);
compareTeamASelect.addEventListener("change", compareTeamsFromCurrentData);
compareTeamBSelect.addEventListener("change", compareTeamsFromCurrentData);

sportSelect.addEventListener("change", () => {
  whatIfSportSelect.value = sportSelect.value;
  historySport.value = sportSelect.value;
  injurySportSelect.value = sportSelect.value;
  loadUpcomingMatches({ force: true });
});

setAutoRefresh();
loadUpcomingMatches({ force: true });
loadHistory({ silent: false });
loadInjuries({ silent: false });
loadModelSummary({ silent: false });
applyUpcomingViewMode();
