# Trains Lasso and Random Forest on the preprocessed data
# Run after preprocess_kaggle.py: python ml_features.py

import sys
import numpy as np
import sklearn
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from itertools import product

_logf = open("ml_results.txt", "w")
def log(msg=""):
    print(msg)
    sys.stdout.flush()
    _logf.write(msg + "\n")
    _logf.flush()

X_NV_INPUT = "data/preprocessed/X_nv_pp.npy"
Y_INPUT = "data/preprocessed/y_pp.npy"

TRAIN_PER = 0.75
DEV_PER = 0.125
TEST_PER = 0.125
RANDOM_STATE = 1

WHICH_Y = 0  # 0=yards, 1=play type, 2=positive play, 3=on track, 4=first down


X_nv = np.load(X_NV_INPUT, allow_pickle=True)
Y = np.load(Y_INPUT, allow_pickle=True).astype("float64")



np.random.seed(RANDOM_STATE)
m = X_nv.shape[0]
shuffled_indices = np.arange(m)
np.random.shuffle(shuffled_indices)

train_idxs = shuffled_indices[:int(m*TRAIN_PER)]
dev_idxs = shuffled_indices[(int(m*TRAIN_PER)):(int(m*TRAIN_PER)+int(m*DEV_PER))]
test_idxs = shuffled_indices[(int(m*TRAIN_PER)+int(m*DEV_PER)):]

X_train, Y_train = X_nv[train_idxs,:], Y[train_idxs, WHICH_Y]
X_dev, Y_dev = X_nv[dev_idxs,:], Y[dev_idxs, WHICH_Y]
X_test, Y_test= X_nv[test_idxs,:], Y[test_idxs, WHICH_Y]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)


benchmark1 = np.median(np.abs(np.median(Y_train)- Y_test))  # naive baseline: always predict median
benchmark2 = np.mean(np.abs(np.median(Y_train)- Y_test))

alphas_Lasso = np.logspace(-2, 2, 50)

log(f"Lasso: tuning over {len(alphas_Lasso)} alpha values via LassoCV...")
if WHICH_Y==0:
	lasso_cv = LassoCV(alphas=alphas_Lasso, cv=5, max_iter=10000, n_jobs=-1)
	lasso_cv.fit(X_train, Y_train)
	alpha_star = lasso_cv.alpha_
	log(f"  Best alpha: {alpha_star:.6f}")

	model_Lasso = Lasso(alpha=alpha_star, max_iter=10000)
	model_Lasso.fit(X_train, Y_train)
else:
	tune_scores_Lasso = np.zeros(len(alphas_Lasso))
	k=0
	for alpha in alphas_Lasso:
		tune_Lasso = Lasso(alpha=alpha, max_iter=10000)
		tune_Lasso.fit(X_train, Y_train)
		tune_scores_Lasso[k] = f1_score(Y_dev, (1.0*(tune_Lasso.predict(X_dev)>=0.5)))
		k+=1
	alpha_star = alphas_Lasso[np.where(tune_scores_Lasso==tune_scores_Lasso.max())][0]
	model_Lasso = Lasso(alpha=alpha_star, max_iter=10000)
	model_Lasso.fit(X_train, Y_train)

non0coeff = np.where(model_Lasso.coef_!=0)

if WHICH_Y==0:
	final_score_Lasso = np.median(np.abs(model_Lasso.predict(X_test)- Y_test))
	log(f"\n=== LASSO RESULTS ===")
	log(f"Best alpha: {alpha_star:.6f}")
	log(f"Median Absolute Error (test): {final_score_Lasso:.4f}")
	log(f"Non-zero coefficients: {len(non0coeff[0])}")
	log(f"Benchmark median|median-Y_test|: {benchmark1:.4f}")
	log(f"Benchmark mean|median-Y_test|: {benchmark2:.4f}")
else:
	test_fits_Lasso = (1.0*(model_Lasso.predict(X_test)>=0.5))
	final_score_Lasso =  f1_score(Y_test, test_fits_Lasso)
	print('F1 Score Lasso: ' + str(final_score_Lasso))
	accuracy_score_Lasso = accuracy_score(Y_test, test_fits_Lasso)
	print('Accuracy Score Lasso: '+str(accuracy_score_Lasso))
	print(test_fits_Lasso[0:20])


hp_grid_RF_n = np.unique(np.array(np.logspace(0,2,15, dtype="int64")))  # n_estimators: 1 to 100
hp_grid_RF_d = np.array([1,3,5,10,20])  # max_depth options
hp_grid_RF = np.array(list(product(hp_grid_RF_n, hp_grid_RF_d)))
tune_scores_RF = np.zeros(hp_grid_RF.shape[0])

log(f"\nRandom Forest: tuning over {hp_grid_RF.shape[0]} hyperparameter combos...")
if WHICH_Y==0:
	for j in range(hp_grid_RF.shape[0]):
		tune_RF = RandomForestRegressor(
			n_estimators = hp_grid_RF[j,0],
			max_depth=hp_grid_RF[j,1],
			criterion="absolute_error")
		tune_RF.fit(X_train, Y_train)
		tune_scores_RF[j] = tune_RF.score(X_dev, Y_dev)
		log(f"  RF combo {j+1}/{hp_grid_RF.shape[0]} (n_est={hp_grid_RF[j,0]}, depth={hp_grid_RF[j,1]}) R2={tune_scores_RF[j]:.4f}")
	hp_RF_star = hp_grid_RF[np.where(tune_scores_RF==tune_scores_RF.max())][0]
	model_RF = RandomForestRegressor(
		n_estimators = hp_RF_star[0],
			max_depth=hp_RF_star[1],
			criterion="absolute_error")
	model_RF.fit(X_train, Y_train)
	final_score_RF = np.median(np.abs(model_RF.predict(X_test)- Y_test))
	final_score_RF2 = np.mean(np.abs(model_RF.predict(X_test)- Y_test))
	log(f"\n=== RANDOM FOREST RESULTS ===")
	log(f"Best hyperparams: n_estimators={hp_RF_star[0]:.0f}, max_depth={hp_RF_star[1]:.0f}")
	log(f"Median Absolute Error (test): {final_score_RF:.4f}")
	log(f"Mean Absolute Error (test): {final_score_RF2:.4f}")
	log(f"\n=== SUMMARY ===")
	log(f"Benchmark (median prediction): MedAE={benchmark1:.4f}, MAE={benchmark2:.4f}")
	log(f"Lasso:         MedAE={final_score_Lasso:.4f}")
	log(f"Random Forest: MedAE={final_score_RF:.4f}, MAE={final_score_RF2:.4f}")
	_logf.close()
else:
	for j in range(hp_grid_RF.shape[0]):
		tune_RF = RandomForestClassifier(n_estimators = hp_grid_RF[j,0],
			max_depth=hp_grid_RF[j,1])
		tune_RF.fit(X_train, Y_train)
		tune_scores_RF[j] = f1_score(Y_dev, tune_RF.predict(X_dev))
	hp_RF_star = hp_grid_RF[np.where(tune_scores_RF==tune_scores_RF.max())][0]
	model_RF = RandomForestClassifier(n_estimators = hp_RF_star[0],
			max_depth=hp_RF_star[0])
	model_RF.fit(X_train, Y_train)
	test_fits_RF = model_RF.predict(X_test)
	final_score_RF = f1_score(Y_test, test_fits_RF)
	print(hp_RF_star)
	print('F1 score RF: ' + str(final_score_RF))
	accuracy_score_RF = accuracy_score(Y_test, test_fits_RF)
	print('Accuracy Score RF: '+str(accuracy_score_RF))
	print(test_fits_RF[0:20])



