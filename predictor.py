import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score

# 1. Load and Preprocess the Data
matches = pd.read_csv("matches.csv", parse_dates=["Date"])

matches["VENUE_CODE"] = matches["Venue"].astype("category").cat.codes
matches["OPP_CODE"] = matches["Opponent"].astype("category").cat.codes
matches["HOUR"] = matches["Time"].str.replace(":.+", "", regex=True).astype("int")
matches["DAY_CODE"] = matches["Date"].dt.dayofweek
matches["TARGET"] = (matches["Result"] == "W").astype("int")

# 2. Basic Prediction with GradientBoostingClassifier
predictors = ["VENUE_CODE", "OPP_CODE", "HOUR", "DAY_CODE"]
train = matches[matches["Date"] < '2024-03-01']
test = matches[matches["Date"] > '2024-03-01']

gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=1)
gbc.fit(train[predictors], train["TARGET"])
preds = gbc.predict(test[predictors])
acc = accuracy_score(test["TARGET"], preds)
print("Basic Prediction Accuracy (GradientBoosting):", acc)

combined_basic = pd.DataFrame(dict(ACTUAL=test["TARGET"], PREDICTED=preds))
print("Basic Prediction Crosstab (GradientBoosting):")
print(pd.crosstab(index=combined_basic["ACTUAL"], columns=combined_basic["PREDICTED"]))
prec = precision_score(test["TARGET"], preds)
print("Basic Prediction Precision (GradientBoosting):", prec)

# 3. Rolling Averages, Temporal & Interaction Features
def rolling_averages(group, cols, new_cols, short_form_window=5, long_form_window=10):
    group = group.sort_values("Date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group["FORM_SHORT"] = group["TARGET"].rolling(short_form_window, closed='left').mean()
    group["FORM_LONG"] = group["TARGET"].rolling(long_form_window, closed='left').mean()
    group = group.dropna(subset=new_cols + ["FORM_SHORT", "FORM_LONG"])
    return group

cols = ["GF", "GA", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]
new_cols = [f"{c}_ROLLING" for c in cols]

# Apply the function for each team
matches_rolling = matches.groupby("Team").apply(lambda x: rolling_averages(x, cols, new_cols, short_form_window=5, long_form_window=10))
matches_rolling = matches_rolling.droplevel('Team')
matches_rolling.index = range(matches_rolling.shape[0])

# Create interaction features:
# Interaction between VENUE_CODE and the rolling average of goals scored.
matches_rolling["VENUE_GF_INTERACTION"] = matches_rolling["VENUE_CODE"] * matches_rolling["GF_ROLLING"]
matches_rolling["VENUE_FORM_SHORT_INTERACTION"] = matches_rolling["VENUE_CODE"] * matches_rolling["FORM_SHORT"]
matches_rolling["VENUE_FORM_LONG_INTERACTION"] = matches_rolling["VENUE_CODE"] * matches_rolling["FORM_LONG"]

# 4. Predictions Using Rolling Averages with GradientBoostingClassifier
def make_predictions(data, predictors):
    train = data[data["Date"] < '2024-03-01']
    test = data[data["Date"] > '2024-03-01']
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=1)
    gbc.fit(train[predictors], train["TARGET"])
    preds = gbc.predict(test[predictors])
    combined = pd.DataFrame(dict(ACTUAL=test["TARGET"], PREDICTED=preds), index=test.index)
    precision = precision_score(test["TARGET"], preds)
    return combined, precision

# Update predictor list to include both the rolling metrics, both form features, and the interaction features.
rolling_predictors = (predictors + new_cols + 
                      ["FORM_SHORT", "FORM_LONG", 
                       "VENUE_GF_INTERACTION", "VENUE_FORM_SHORT_INTERACTION", "VENUE_FORM_LONG_INTERACTION"])
combined_rolling, precision_val = make_predictions(matches_rolling, rolling_predictors)
print("Rolling Averages Prediction Precision (GradientBoosting):", precision_val)

combined_rolling = combined_rolling.merge(
    matches_rolling[["Date", "Team", "Opponent", "Result"]],
    left_index=True, right_index=True
)

# 5. Additional Merging and Analysis
class MissingDict(dict):
    def __missing__(self, key):
        return key

map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}
mapping = MissingDict(**map_values)
combined_rolling["NEW_TEAM"] = combined_rolling["Team"].map(mapping)

merged = combined_rolling.merge(
    combined_rolling, left_on=["Date", "NEW_TEAM"],
    right_on=["Date", "Opponent"], suffixes=("_X", "_Y")
)

result_counts = merged[(merged["PREDICTED_X"] == 1) & (merged["PREDICTED_Y"] == 0)]["ACTUAL_X"].value_counts()
print("Final Merged Prediction Counts:")
print(result_counts)
