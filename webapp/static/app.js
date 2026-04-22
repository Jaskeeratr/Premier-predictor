const sportSelect = document.getElementById("sport-select");
const loadMatchesBtn = document.getElementById("load-matches-btn");
const leagueFilter = document.getElementById("league-filter");
const teamFilter = document.getElementById("team-filter");
const refreshIntervalSelect = document.getElementById("refresh-interval");
const upcomingBody = document.getElementById("upcoming-body");
const upcomingMeta = document.getElementById("upcoming-meta");

const historySport = document.getElementById("history-sport");
const historyLeague = document.getElementById("history-league");
const historyTeam = document.getElementById("history-team");
const historyLimit = document.getElementById("history-limit");
const loadHistoryBtn = document.getElementById("load-history-btn");
const clearHistoryBtn = document.getElementById("clear-history-btn");
const historyBody = document.getElementById("history-body");
const historyMeta = document.getElementById("history-meta");

const whatIfForm = document.getElementById("what-if-form");
const whatIfResult = document.getElementById("what-if-result");
const whatIfSportSelect = document.getElementById("whatif-sport");
const whatIfSection = document.getElementById("what-if-section");

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

function setUpcomingMeta(text) {
  upcomingMeta.textContent = text;
}

function setHistoryMeta(text) {
  historyMeta.textContent = text;
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
      return;
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

function renderUpcomingMatches() {
  const matches = filterUpcomingMatches();
  if (!matches.length) {
    upcomingBody.innerHTML = `<tr><td colspan="8">No matches match the current filters.</td></tr>`;
    return;
  }

  const rows = matches.map((match) => {
    return `
      <tr>
        <td>${asLocalTimestamp(match.kickoff_utc)}</td>
        <td>${match.league}</td>
        <td>${match.home_team} vs ${match.away_team}</td>
        <td>${match.venue || "Unknown Venue"}</td>
        <td class="winner">${match.predicted_winner}</td>
        <td class="confidence">${pct(match.confidence)}</td>
        <td>${match.predicted_score}</td>
        <td><button type="button" class="use-match-btn" data-event-id="${match.event_id}">Use</button></td>
      </tr>
    `;
  });
  upcomingBody.innerHTML = rows.join("");

  const buttons = upcomingBody.querySelectorAll(".use-match-btn");
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const eventId = button.getAttribute("data-event-id");
      const selected = allUpcomingMatches.find((item) => item.event_id === eventId);
      if (selected) {
        fillWhatIfFromMatch(selected);
        if (whatIfSection) {
          whatIfSection.scrollIntoView({ behavior: "smooth", block: "start" });
        }
        setTimeout(() => {
          whatIfForm.requestSubmit();
        }, 120);
      }
    });
  });
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
    const filteredCount = filterUpcomingMatches().length;
    const updatedAt = asLocalTimestamp(payload.updated_at_utc);
    const training = payload.training || {};
    let trainingSummary = "Model: heuristic fallback";
    if (training.status === "ok") {
      trainingSummary = `Model: ${training.best_model} | Holdout Acc: ${(Number(training.holdout_accuracy || 0) * 100).toFixed(1)}% | Samples: ${training.sample_count}`;
    } else if (training.status) {
      trainingSummary = `Model status: ${training.status}`;
    }
    setUpcomingMeta(
      `Showing ${filteredCount} of ${allUpcomingMatches.length} matches. Snapshot #${payload.snapshot_id ?? "-"} | Updated: ${updatedAt} | ${trainingSummary}`,
    );
    whatIfSportSelect.value = sport;
    historySport.value = sport;
    await loadHistory({ silent: true });
  } catch (error) {
    setUpcomingMeta(error.message);
    upcomingBody.innerHTML = `<tr><td colspan="8">${error.message}</td></tr>`;
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
  if (!Number.isFinite(minutes) || minutes <= 0) {
    return;
  }
  autoRefreshHandle = setInterval(() => {
    loadUpcomingMatches({ force: true, silent: true });
  }, minutes * 60 * 1000);
}

function renderHistory(rows) {
  if (!rows.length) {
    historyBody.innerHTML = `<tr><td colspan="8">No saved predictions found.</td></tr>`;
    return;
  }
  historyBody.innerHTML = rows
    .map((row) => {
      const sportLabel = row.sport.replace("_", " ");
      return `
        <tr>
          <td>${row.created_at_utc}</td>
          <td>${row.kickoff_utc || ""}</td>
          <td>${sportLabel}</td>
          <td>${row.league || ""}</td>
          <td>${row.home_team || ""} vs ${row.away_team || ""}</td>
          <td class="winner">${row.predicted_winner || ""}</td>
          <td class="confidence">${pct(row.confidence || 0)}</td>
          <td>${row.predicted_score || ""}</td>
        </tr>
      `;
    })
    .join("");
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
    if (!response.ok) {
      throw new Error(payload.error || "Failed to load history.");
    }
    renderHistory(payload.rows || []);
    setHistoryMeta(`Loaded ${payload.count} rows.`);
  } catch (error) {
    setHistoryMeta(error.message);
    historyBody.innerHTML = `<tr><td colspan="8">${error.message}</td></tr>`;
  } finally {
    if (!silent) {
      loadHistoryBtn.disabled = false;
    }
  }
}

async function clearHistory() {
  const targetSport = historySport.value;
  const scope = targetSport ? `for ${targetSport.replace("_", " ")}` : "for all sports";
  const confirmed = window.confirm(`Clear saved prediction history ${scope}?`);
  if (!confirmed) return;

  try {
    const response = await fetch("/api/history", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sport: targetSport || null }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Failed to clear history.");
    }
    setHistoryMeta(`Deleted ${payload.deleted_rows} rows.`);
    await loadHistory({ silent: true });
  } catch (error) {
    setHistoryMeta(error.message);
  }
}

function renderWhatIfResult(result) {
  whatIfResult.classList.remove("hidden");
  whatIfResult.innerHTML = `
    <h3>What-If Result</h3>
    <div class="result-row">
      <div><strong>Winner:</strong> ${result.predicted_winner}</div>
      <div><strong>Confidence:</strong> ${pct(result.confidence)}</div>
      <div><strong>Rough Score:</strong> ${result.predicted_score}</div>
      <div><strong>Home Win:</strong> ${pct(result.home_win_probability)}</div>
      <div><strong>Draw:</strong> ${pct(result.draw_probability)}</div>
      <div><strong>Away Win:</strong> ${pct(result.away_win_probability)}</div>
    </div>
  `;
}

function readNumericInput(id, fallback) {
  const value = Number(document.getElementById(id).value);
  return Number.isFinite(value) ? value : fallback;
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
  };

  try {
    const response = await fetch("/api/what-if", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "What-if simulation failed.");
    }
    renderWhatIfResult(result);
  } catch (error) {
    whatIfResult.classList.remove("hidden");
    whatIfResult.innerHTML = `<h3>What-If Result</h3><p>${error.message}</p>`;
  }
}

loadMatchesBtn.addEventListener("click", () => loadUpcomingMatches({ force: true }));
leagueFilter.addEventListener("change", renderUpcomingMatches);
teamFilter.addEventListener("input", renderUpcomingMatches);
refreshIntervalSelect.addEventListener("change", setAutoRefresh);

loadHistoryBtn.addEventListener("click", () => loadHistory({ silent: false }));
clearHistoryBtn.addEventListener("click", clearHistory);
historySport.addEventListener("change", () => loadHistory({ silent: true }));
historyLeague.addEventListener("input", () => loadHistory({ silent: true }));
historyTeam.addEventListener("input", () => loadHistory({ silent: true }));
historyLimit.addEventListener("change", () => loadHistory({ silent: true }));

whatIfForm.addEventListener("submit", submitWhatIf);
sportSelect.addEventListener("change", () => {
  whatIfSportSelect.value = sportSelect.value;
  historySport.value = sportSelect.value;
  loadUpcomingMatches({ force: true });
});

setAutoRefresh();
loadUpcomingMatches({ force: true });
loadHistory({ silent: false });
