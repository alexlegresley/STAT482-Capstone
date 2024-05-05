# --------------------------------------------------------------------
# Imports

import pathlib
import os
import pandas as pd
import numpy as np
import functools
import operator

import scipy.stats as stats
from matplotlib import pyplot as plt

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegressionCV, LinearRegression
from sklearn.feature_selection import RFECV


# --------------------------------------------------------------------
# Utility functions

def computeWeightedAverage(row):
    combos = basicStats.loc[(basicStats["Season"] == row["Season"]) & (basicStats["Tm"] != "TOT") & (basicStats["Player"] == row["Player"])]
    rkKeys = zip(combos["Season"], combos["Tm"])
    weights = combos["MP"] / combos["MP"].sum()
    return np.average([rkMapping[k] for k in rkKeys], weights = weights)


def optimizeThreshold(model, X, y):
    best = []
    for trainIndex, testIndex in StratifiedKFold(n_splits = 5).split(X, y):
        mod = model.fit(X.iloc[trainIndex], y.iloc[trainIndex])
        res = []
        for th in np.arange(0.05, 1, 0.05):
            yPred = mod.predict_proba(X.iloc[testIndex])[:, 1] > th
            res.append((th, f1_score(y.iloc[testIndex], yPred)))
        best.append(max(res, key = operator.itemgetter(1)))

    return np.average([th for th, _ in best]).round(2), np.average([f1 for _, f1 in best])


def withDummyPos(X):
    posFeatures = pd.get_dummies(X.Pos, drop_first = True, prefix = "pos")
    return pd.concat([X, posFeatures], axis = 1).drop("Pos", axis = 1).apply(lambda x: x.fillna(x.median()) if pd.api.types.is_numeric_dtype(x) else x)


def createModelData(basic_stats, adv_stats, standings):
    basic = pd.read_csv(basic_stats)
    adv = pd.read_csv(adv_stats)
    stand = pd.read_csv(standings, header = 1)

    basic.loc[:, "Player"] = basic["Player"].map(lambda x: x.rstrip("*"))
    adv.loc[:, "Player"] = adv["Player"].map(lambda x: x.rstrip("*"))

    tmMapping = {
        "Atlanta Hawks": "ATL",
        "Boston Celtics": "BOS",
        "Brooklyn Nets": "BRK",
        "Charlotte Hornets": "CHO", 
        "Chicago Bulls": "CHI",
        "Cleveland Cavaliers": "CLE", 
        "Dallas Mavericks": "DAL",
        "Denver Nuggets": "DEN",
        "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW",
        "Houston Rockets": "HOU",
        "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC",
        "Los Angeles Lakers": "LAL",
        "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA",
        "Milwaukee Bucks": "MIL",
        "Minnesota Timberwolves": "MIN",
        "New Orleans Pelicans": "NOP",
        "New York Knicks": "NYK",
        "Oklahoma City Thunder": "OKC",
        "Orlando Magic": "ORL",
        "Philadelphia 76ers": "PHI",
        "Phoenix Suns": "PHO",
        "Portland Trail Blazers": "POR",
        "Sacramento Kings": "SAC",
        "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR",
        "Utah Jazz": "UTA",
        "Washington Wizards": "WAS"
    }
    stand["Tm"] = stand["Team"].map(tmMapping)
    rkMapping = dict(zip(stand["Tm"], stand["Rk"]))

    def computeWeightedAverage(row):
        combos = basic.loc[(basic["Player"] == row["Player"]) & (basic["Tm"] != "TOT")]
        weights = combos["MP"] / combos["MP"].sum()
        return np.average([rkMapping[k] for k in combos["Tm"]], weights = weights)

    basic["TeamRk"] = [rkMapping[rowKey] if (rowKey := row["Tm"]) in rkMapping.keys() else computeWeightedAverage(row) for _, row in basic.iterrows()]
    
    basic.drop_duplicates(subset = "Player", keep = "first", inplace = True)
    adv.drop_duplicates(subset = "Player", keep = "first", inplace = True)

    basic.drop(["Rk", "Tm", "Unnamed: 29", "Player-additional"], axis = 1, inplace = True)
    adv.drop(["Rk", "Pos", "Age", "Tm", "G", "MP", "Unnamed: 19", "Unnamed: 24", "Player-additional"], axis = 1, inplace = True)

    combined = pd.merge(basic, adv, how = "left", on = "Player")
    combined["Pos"] = combined["Pos"].astype("category")

    return combined


def predictVoteShare(data, stage1, threshold, stage2):
    probSomeShare = stage1.predict_proba(data.drop("Player", axis = 1))[:, 1]

    noShare = data.loc[~(probSomeShare > threshold)].copy()
    noShare["Vote_Share"] = 0
    
    someShare = data.loc[probSomeShare > threshold].copy()
    someShare["Vote_Share"] = stage2.predict(someShare.drop("Player", axis = 1))

    return pd.concat([someShare[["Player", "Vote_Share"]], noShare[["Player", "Vote_Share"]]], ignore_index = True).sort_values("Vote_Share", ascending = False).reset_index(drop = True)


def matchDummiesPos(X, XTrain):
    presentDummies = pd.get_dummies(X, prefix = "pos")
    res = pd.DataFrame()
    for d in [d for d in pd.get_dummies(XTrain, prefix = "pos").columns if d.startswith("pos_")]:
        if d in presentDummies.columns:
            res[d] = presentDummies[d]
        else:
            res[d] = False
    res.drop(columns = res.columns[0], axis = 1, inplace = True)
    return pd.concat([X, res], axis = 1).drop("Pos", axis = 1).apply(lambda x: x.fillna(x.median()) if pd.api.types.is_numeric_dtype(x) else x)


# --------------------------------------------------------------------
# Data cleaning

basicStats = pd.concat(
    (pd.read_csv(f).assign(Season = os.path.basename(f).removesuffix(".csv")) for f in (pathlib.Path()/"Data"/"Basic_Stats").glob("*.csv")), 
    ignore_index = True
)
advStats = pd.concat(
    (pd.read_csv(f).assign(Season = os.path.basename(f).removesuffix(".csv")) for f in (pathlib.Path()/"Data"/"Advanced_Stats").glob("*.csv")), 
    ignore_index = True
)
dpoyVoting = pd.concat(
    (pd.read_excel(f, header = 1).assign(Season = os.path.basename(f).removesuffix(".xlsx")) for f in (pathlib.Path()/"Data"/"DPOY_Voting").glob("*.xlsx")), 
    ignore_index = True
)
expStand = pd.concat(
    (pd.read_csv(f, header = 1).assign(Season = os.path.basename(f).removesuffix(".csv")) for f in (pathlib.Path()/"Data"/"Standings").glob("*.csv")), 
    ignore_index = True
)

teamMappings = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "Charlotte Bobcats": "CHA",
    "Charlotte Hornets": "CHO", 
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE", 
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Kansas City Kings": "KCK",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Jersey Nets": "NJN",
    "New Orleans Hornets" : "NOH",
    "New Orleans Pelicans": "NOP",
    "New Orleans/Oklahoma City Hornets": "NOK",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "San Diego Clippers": "SDC",
    "Seattle SuperSonics": "SEA",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Vancouver Grizzlies": "VAN",
    "Washington Bullets": "WSB",
    "Washington Wizards": "WAS"
}
expStand["Tm"] = expStand["Team"].map(teamMappings)
rkMapping = dict(zip(zip(expStand["Season"], expStand["Tm"]), expStand["Rk"]))

basicStats.loc[:, "Tm"] = basicStats["Tm"].apply(lambda x: "CHO" if x == "CHH" else x)
advStats.loc[:, "Tm"] = advStats["Tm"].apply(lambda x: "CHO" if x == "CHH" else x)

basicStats["TeamRk"] = [rkMapping[rowKey] if (rowKey := (row["Season"], row["Tm"])) in rkMapping.keys() else computeWeightedAverage(row) for _, row in basicStats.iterrows()]

basicStats.drop_duplicates(subset = ["Season", "Player"], keep = "first", inplace = True)
advStats.drop_duplicates(subset = ["Season", "Player"], keep = "first", inplace = True)

basicStats.loc[:, "Player"] = basicStats["Player"].map(lambda x: x.rstrip("*"))
advStats.loc[:, "Player"] = advStats["Player"].map(lambda x: x.rstrip("*"))

basicStats = basicStats.drop(["Rk", "Tm", "Unnamed: 29", "Player-additional"], axis = 1)
advStats = advStats.drop(["Rk", "Pos", "Age", "Tm", "G", "MP", "Unnamed: 19", "Unnamed: 24", "Player-additional"], axis = 1)
dpoyVoting = dpoyVoting[["Season", "Player", "Share"]]

combinedData = functools.reduce(
    lambda left, right: pd.merge(left, right, how = "left", on = ["Season", "Player"]),
    [basicStats, advStats, dpoyVoting]
)
combinedData["Share"] = combinedData["Share"].fillna(0)


# --------------------------------------------------------------------
# Stage 1

model1Data = combinedData.drop(columns = ["Season", "Player"])
model1Data["Pos"] = model1Data["Pos"].astype("category")
model1Data["ShareBinary"] = [1 if x != 0 else 0 for x in model1Data["Share"]]

X = model1Data.drop(["Share", "ShareBinary"], axis = 1)
y = model1Data["ShareBinary"]

classWeightRatio = y.value_counts()[0] / y.value_counts()[1]


# XGBoost classifier

mod = xgb.XGBClassifier(
    scale_pos_weight = classWeightRatio, 
    eval_metric = "aucpr", 
    enable_categorical = True
)
paramsDist = {
    "learning_rate": stats.uniform(0.01, 0.20), 
    "max_depth": stats.randint(3, 12), 
    "n_estimators": stats.randint(50, 350)
}
rsMod = RandomizedSearchCV(
    estimator = mod, 
    param_distributions = paramsDist, 
    scoring = "average_precision", 
    n_iter = 75, 
    cv = StratifiedKFold(n_splits = 5), 
    refit = True
)
rsMod.fit(X, y)
print(f"Best parameters: {rsMod.best_params_}")
print(f"Average cross validation score (AUCPR) from best model: {rsMod.best_score_}")
tunedMod = rsMod.best_estimator_

_, ax = plt.subplots(figsize = (10, 12))
xgb.plot_importance(tunedMod, ax = ax)

bestThXGB, bestF1XGB = optimizeThreshold(tunedMod, X, y)
print(f"Average best threshold: {bestThXGB}")
print(f"Average best F1 score: {bestF1XGB}")


# Logistic regression

modLR = LogisticRegressionCV(
    class_weight = "balanced",
    scoring = "average_precision",
    solver = "newton-cholesky",
    cv = StratifiedKFold(n_splits = 5),
    refit = True
)
modLR.fit(withDummyPos(X), y)
print(f"Average cross validation score (AUCPR): {np.average(modLR.scores_[1])}")

bestThLR, bestF1LR = optimizeThreshold(modLR, withDummyPos(X), y)
print(f"Average best threshold: {bestThLR}")
print(f"Average best F1 score: {bestF1LR}")


# --------------------------------------------------------------------
# Stage 2

model2Data = model1Data.drop(["ShareBinary"], axis = 1)
model2Data = model2Data.loc[model2Data["Share"].gt(0)]
model2Data.reset_index(drop = True, inplace = True)

X = model2Data.drop(["Share"], axis = 1)
y = model2Data["Share"]


# XGBoost regressor

mod2 = xgb.XGBRegressor(
    eval_metric = "rmse",
    enable_categorical = True
)
paramsDist = {
    "learning_rate": stats.uniform(0.01, 0.20), 
    "max_depth": stats.randint(3, 12), 
    "n_estimators": stats.randint(25, 300)
}
rsMod2 = RandomizedSearchCV(
    estimator = mod2, 
    param_distributions = paramsDist, 
    scoring = "neg_root_mean_squared_error",
    n_iter = 75, 
    cv = KFold(n_splits = 5), 
    refit = True
)
rsMod2.fit(X, y)
print(f"Best parameters: {rsMod2.best_params_}")
print(f"Best cross validation score (neg RMSE): {rsMod2.best_score_}")
tunedMod2 = rsMod2.best_estimator_
print(
    f"Average cross validation score (r2): "
    f"{np.average(cross_val_score(tunedMod2, X, y, scoring = "r2", cv = KFold(n_splits = 5)))}"
)

_, ax = plt.subplots(figsize = (10, 12))
xgb.plot_importance(tunedMod2, ax = ax)


# Linear regression

modLR2 = RFECV(
    estimator = LinearRegression(),
    min_features_to_select = 1,
    step = 1,
    scoring = "neg_root_mean_squared_error",
    cv = KFold(n_splits = 5)
)
modLR2.fit(withDummyPos(X), y)
print(
    f"Average cross validation score (negative RMSE): "
    f"{np.average(modLR2.cv_results_["mean_test_score"])}"
)
print(
    f"Average cross validation score (r2): "
    f"{np.average(cross_val_score(modLR2, withDummyPos(X), y, scoring = "r2", cv = KFold(n_splits = 5)))}"
)


# ----------------------------------------------------------------------------
# Current year prediction

currentYear = createModelData(
    basic_stats = (pathlib.Path()/"Data"/"Current_Year"/"2023-2024_Basic_Stats.csv").resolve(),
    adv_stats = (pathlib.Path()/"Data"/"Current_Year"/"2023-2024_Adv_Stats.csv").resolve(),
    standings = (pathlib.Path()/"Data"/"Current_Year"/"2023-2024_Standings.csv").resolve()
)


# XGBoost models

resultsXGB = predictVoteShare(data = currentYear, stage1 = tunedMod, threshold = bestThXGB, stage2 = tunedMod2)
resultsXGB.loc[resultsXGB["Vote_Share"].gt(0)]


# Linear models

resultsLR = predictVoteShare(
    data = matchDummiesPos(currentYear, model1Data), 
    stage1 = modLR, 
    threshold = bestThLR, 
    stage2 = modLR2
)
resultsLR.loc[resultsLR["Vote_Share"].gt(0)]