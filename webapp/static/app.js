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
      const selected = allUpcomingMatches.find((item) => String(item.event_id) === String(eventId));
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
    const injuryCount = Number(payload.injury_adjustments_count || 0);
    let trainingSummary = "Model: heuristic fallback";
    if (training.status === "ok") {
      trainingSummary = `Model: ${training.best_model} | Holdout Acc: ${(Number(training.holdout_accuracy || 0) * 100).toFixed(1)}% | Samples: ${training.sample_count}`;
    } else if (training.status) {
      trainingSummary = `Model status: ${training.status}`;
    }
    setUpcomingMeta(
      `Showing ${filteredCount} of ${allUpcomingMatches.length} matches. Snapshot #${payload.snapshot_id ?? "-"} | Updated: ${updatedAt} | ${trainingSummary} | Injury rules: ${injuryCount}`,
    );
    whatIfSportSelect.value = sport;
    historySport.value = sport;
    injurySportSelect.value = sport;
    await loadHistory({ silent: true });
    await loadInjuries({ silent: true });
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
          <td>${asUtcTimestamp(row.created_at_utc)}</td>
          <td>${asUtcTimestamp(row.kickoff_utc)}</td>
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
  const confidence = Number(result.confidence || 0);
  const interpretation =
    confidence < 0.5
      ? "Top pick in a close match (less certainty)"
      : "Strongest projected outcome";
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
    if (!response.ok) {
      throw new Error(result.error || "What-if simulation failed.");
    }
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

  const rowsEls = injuryBody.querySelectorAll(".injury-row");
  rowsEls.forEach((rowEl) => {
    rowEl.addEventListener("click", () => {
      const sport = rowEl.dataset.sport || "";
      const team = rowEl.dataset.team || "";
      const row = rows.find((item) => item.sport === sport && item.team === team);
      if (row) {
        fillInjuryFormFromRow(row);
      }
    });
  });
}

async function loadInjuries({ silent = false } = {}) {
  if (!silent) {
    setInjuryMeta("Loading injury adjustments...");
  }
  const sport = injurySportSelect.value;

  try {
    const response = await fetch(`/api/injuries?sport=${encodeURIComponent(sport)}&limit=500`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Failed to load injury adjustments.");
    }
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
    if (!response.ok) {
      throw new Error(result.error || "Failed to save injury adjustment.");
    }
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
  const confirmed = window.confirm(`Delete injury adjustment for ${team} in ${sport.replace("_", " ")}?`);
  if (!confirmed) return;

  try {
    const response = await fetch("/api/injuries", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sport, team }),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "Failed to delete injury adjustment.");
    }
    setInjuryMeta(`Deleted ${result.deleted_rows} row(s).`);
    await loadInjuries({ silent: true });
    await loadUpcomingMatches({ force: true, silent: true });
  } catch (error) {
    setInjuryMeta(error.message);
  }
}

async function clearSportInjuries() {
  const sport = injurySportSelect.value;
  const confirmed = window.confirm(`Clear all injury adjustments for ${sport.replace("_", " ")}?`);
  if (!confirmed) return;

  try {
    const response = await fetch("/api/injuries", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sport }),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "Failed to clear injury adjustments.");
    }
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
