import sqlite3
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score

# 1. Load Data from Optimized Database
# Connect to the SQLite database that has your matches.csv data loaded
conn = sqlite3.connect('premier_league.db')

# Retrieve all matches from 2020 to 2024 (the full dataset)
query_full = "SELECT * FROM matches WHERE Date BETWEEN '2020-01-01' AND '2024-12-31';"
df = pd.read_sql(query_full, conn, parse_dates=["Date"])
conn.close()

# 2. Preprocess the Data
df["VENUE_CODE"] = df["Venue"].astype("category").cat.codes
df["OPP_CODE"] = df["Opponent"].astype("category").cat.codes
df["HOUR"] = df["Time"].str.replace(":.+", "", regex=True).astype("int")
df["DAY_CODE"] = df["Date"].dt.dayofweek
df["TARGET"] = (df["Result"] == "W").astype("int")

df = df.sort_values("Date")


# 3. Rolling Prediction Without Form Features

# (This simulates using "all the previous dataset" to predict wins.)
predictors = ["VENUE_CODE", "OPP_CODE", "HOUR", "DAY_CODE"]

# Define a cutoff date from which we begin predicting (e.g., March 1, 2024)
cutoff_date = pd.Timestamp("2024-03-15")

predictions = []
actuals = []

dates = df[df["Date"] >= cutoff_date]["Date"].unique()

for current_date in sorted(dates):
    test_data = df[df["Date"] == current_date]
    train_data = df[df["Date"] < current_date]
    
    if len(train_data) < 50:
        continue

    X_train = train_data[predictors]
    y_train = train_data["TARGET"]
    X_test = test_data[predictors]
    y_test = test_data["TARGET"]

    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=1)
    gbc.fit(X_train, y_train)
    
    # Predict for all matches on current_date
    preds = gbc.predict(X_test)
    
    predictions.extend(preds)
    actuals.extend(y_test)

# Compute overall accuracy and precision from the rolling predictions
acc = accuracy_score(actuals, predictions)
prec = precision_score(actuals, predictions)

print("Rolling Prediction Accuracy (using all previous data): {:.2%}".format(acc))
print("Rolling Prediction Precision (using all previous data): {:.2%}".format(prec))
