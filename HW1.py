import numpy as np
import random
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import graphviz
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import *
import pandas as pd
from sample_code_unigram import get_tokens
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from sklearn.model_selection import train_test_split

data = pd.read_csv("news-train.csv")

category = {"sport": [],
			"business": [],
			"politics": [],
			"entertainment": [],
			"tech": []}

for idx, r in data.iterrows():
	category[r["Category"]].append(' '.join(str(elem) for elem in get_tokens(r["Text"])))

# print(category)

random.seed(0)

for cat in category:
    random.shuffle(category[cat])

s_y = np.zeros((len(category["sport"]),))
b_y = np.ones((len(category["business"]),))
p_y = 2 * np.ones((len(category["politics"]),))
e_y = 3 * np.ones((len(category["entertainment"]),))
t_y = 4 * np.ones((len(category["tech"]),)) 

num_s_train_val = int(len(category["sport"]) * 0.8)
num_b_train_val = int(len(category["business"]) * 0.8)
num_p_train_val = int(len(category["politics"]) * 0.8)
num_e_train_val = int(len(category["entertainment"]) * 0.8)
num_t_train_val = int(len(category["tech"]) * 0.8)

train_val_data = category["sport"][:num_s_train_val] + category["business"][:num_b_train_val] + category["politics"][:num_p_train_val] + category["entertainment"][:num_e_train_val] + category["tech"][:num_t_train_val]
test_data = category["sport"][num_s_train_val:] + category["business"][num_b_train_val:] + category["politics"][num_p_train_val:] + category["entertainment"][num_e_train_val:] + category["tech"][num_t_train_val:]

train_val_y = np.concatenate((s_y[:num_s_train_val], b_y[:num_b_train_val], p_y[:num_p_train_val], e_y[:num_e_train_val], t_y[:num_t_train_val]))
test_y = np.concatenate((s_y[num_s_train_val:], b_y[num_b_train_val:], p_y[num_p_train_val:], e_y[num_e_train_val:], t_y[num_t_train_val:]))

print(len(train_val_data), len(train_val_y))
print(len(test_data), len(test_y))

tfidf_vectorizer = TfidfVectorizer(use_idf=True, min_df=3, ngram_range=(1, 5))
X_train_tfidf = tfidf_vectorizer.fit_transform(train_val_data)
X_test_tfidf = tfidf_vectorizer.transform(test_data)

#2(a) uncomment 77 or 78 to choose from gini or entropy
def dtc_parameter_tune_1fold(train_val_X, train_val_y):
	depths = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
	train_acc_all = []
	val_acc_all = []

	for depth in depths:
		train_acc = []
		val_acc = []
		##########################
		train_X, val_X, train_y, val_y = train_test_split(train_val_X, train_val_y, test_size=0.2, random_state=1)

		dtc = tree.DecisionTreeClassifier(max_depth=depth, random_state=1)
		# dtc = tree.DecisionTreeClassifier(max_depth=depth, random_state=1, criterion="entropy")
		dtc.fit(train_X, train_y)
		train_acc.append(dtc.score(train_X, train_y))
		val_acc.append(dtc.score(val_X, val_y))
		##########################

		avg_train_acc = sum(train_acc) / len(train_acc)
		avg_val_acc = sum(val_acc) / len(val_acc)
		print("Depth: ", depth)
		print("Training accuracy: ", avg_train_acc * 100, "%")
		print("Validation accuracy: ", avg_val_acc * 100, "%")
		
		train_acc_all.append(avg_train_acc)
		val_acc_all.append(avg_val_acc)

	return depths, train_acc_all, val_acc_all

depths, train_acc_all, val_acc_all = dtc_parameter_tune_1fold(X_train_tfidf, train_val_y)

plt.plot(depths, train_acc_all, marker='.', label="Training accuracy")
plt.plot(depths, val_acc_all, marker='.', label="Validation accuracy")
plt.xlabel('Depth of tree')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 2(b) criterion, uncomment 124 or 126 to choose from gini or entropy
def dtc_parameter_tune(train_val_X, train_val_y):
	depths = [5, 10, 15, 25, 30, 35, 40, 50, 60, 80, 100, 120]

	train_acc_all = []
	val_acc_all = []

	kf = KFold(n_splits = 5)
	for depth in depths:
		train_acc = []
		val_acc = []
		for train_index, val_index in kf.split(train_val_X):
			##########################
			train_X = train_val_X[train_index,:]
			val_X = train_val_X[val_index,:]

			train_y = train_val_y[train_index]
			val_y = train_val_y[val_index]

			# gini
			dtc = tree.DecisionTreeClassifier(max_depth=depth, random_state=1)
			# entropy
			# dtc = tree.DecisionTreeClassifier(max_depth=depth, random_state=1, criterion="entropy")
			dtc.fit(train_X, train_y)
			train_acc.append(dtc.score(train_X, train_y))
			val_acc.append(dtc.score(val_X, val_y))
			##########################

		avg_train_acc = sum(train_acc) / len(train_acc)
		avg_val_acc = sum(val_acc) / len(val_acc)
		print("Depth: ", depth)
		print("Training accuracy: ", avg_train_acc * 100, "%")
		print("Validation accuracy: ", avg_val_acc * 100, "%")

		train_acc_all.append(avg_train_acc)
		val_acc_all.append(avg_val_acc)

	return depths, train_acc_all, val_acc_all

depths, train_acc_all, val_acc_all = dtc_parameter_tune(X_train_tfidf, train_val_y)

plt.plot(depths, train_acc_all, marker='.', label="Training accuracy")
plt.plot(depths, val_acc_all, marker='.', label="Validation accuracy")
plt.xlabel('Depth of tree')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

2(b) min_samples_leaf
def dtc_parameter_tune(train_val_X, train_val_y):
	min_samples_leaf = [1, 3, 5, 7, 9, 11, 13, 15]

	train_acc_all = []
	val_acc_all = []

	kf = KFold(n_splits = 5)
	for sample in min_samples_leaf:
		train_acc = []
		val_acc = []
		for train_index, val_index in kf.split(train_val_X):
			##########################
			train_X = train_val_X[train_index,:]
			val_X = train_val_X[val_index,:]

			train_y = train_val_y[train_index]
			val_y = train_val_y[val_index]

			# min_samples_leaf
			dtc = tree.DecisionTreeClassifier(min_samples_leaf=sample, max_depth=20, random_state=1)
			
			dtc.fit(train_X, train_y)
			train_acc.append(dtc.score(train_X, train_y))
			val_acc.append(dtc.score(val_X, val_y))
			##########################

		avg_train_acc = sum(train_acc) / len(train_acc)
		avg_val_acc = sum(val_acc) / len(val_acc)
		print("sample: ", sample)
		print("Training accuracy: ", avg_train_acc * 100, "%")
		print("Validation accuracy: ", avg_val_acc * 100, "%")

		train_acc_all.append(avg_train_acc)
		val_acc_all.append(avg_val_acc)

	return min_samples_leaf, train_acc_all, val_acc_all

min_samples_leaf, train_acc_all, val_acc_all = dtc_parameter_tune(X_train_tfidf, train_val_y)

plt.plot(min_samples_leaf, train_acc_all, marker='.', label="Training accuracy")
plt.plot(min_samples_leaf, val_acc_all, marker='.', label="Validation accuracy")
plt.xlabel('Min_samples_leaf')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 2(b) max_features
def dtc_parameter_tune(train_val_X, train_val_y):
	max_features = [1, 3, 5, 7, 9, 10, 15, 20, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400]

	train_acc_all = []
	val_acc_all = []

	kf = KFold(n_splits = 5)
	for feature in max_features:
		train_acc = []
		val_acc = []
		for train_index, val_index in kf.split(train_val_X):
			##########################
			train_X = train_val_X[train_index,:]
			val_X = train_val_X[val_index,:]

			train_y = train_val_y[train_index]
			val_y = train_val_y[val_index]

			# max_features
			dtc = tree.DecisionTreeClassifier(max_features=feature, max_depth=20, random_state=1)
			
			dtc.fit(train_X, train_y)
			train_acc.append(dtc.score(train_X, train_y))
			val_acc.append(dtc.score(val_X, val_y))
			##########################

		avg_train_acc = sum(train_acc) / len(train_acc)
		avg_val_acc = sum(val_acc) / len(val_acc)
		print("Max_features: ", feature)
		print("Training accuracy: ", avg_train_acc * 100, "%")
		print("Validation accuracy: ", avg_val_acc * 100, "%")

		train_acc_all.append(avg_train_acc)
		val_acc_all.append(avg_val_acc)

	return max_features, train_acc_all, val_acc_all

max_features, train_acc_all, val_acc_all = dtc_parameter_tune(X_train_tfidf, train_val_y)

plt.plot(max_features, train_acc_all, marker='.', label="Training accuracy")
plt.plot(max_features, val_acc_all, marker='.', label="Validation accuracy")
plt.xlabel('Max_features')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 3 random forest
def rf_parameter_tune(train_val_X, train_val_y):
	n_estimators = [1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 300, 500]
	train_acc_all = []
	val_acc_all = []
	
	kf = KFold(n_splits = 5)
	for estimator in n_estimators:
		train_acc = []
		val_acc = []
		
		for train_index, val_index in kf.split(train_val_X):
			train_X = train_val_X[train_index,:]
			val_X = train_val_X[val_index,:]

			train_y = train_val_y[train_index]
			val_y = train_val_y[val_index]
	            
			rf = RandomForestClassifier(n_estimators=estimator, max_depth=20, n_jobs=-1, random_state=1)
			rf.fit(train_X, train_y)
			train_acc.append(rf.score(train_X, train_y))
			val_acc.append(rf.score(val_X, val_y))
		
		avg_train_acc = sum(train_acc) / len(train_acc)
		avg_val_acc = sum(val_acc) / len(val_acc)
		print("estimator: ", estimator)
		print("Training accuracy: ", avg_train_acc * 100, "%")
		print("Validation accuracy: ", avg_val_acc * 100, "%")

		train_acc_all.append(avg_train_acc)
		val_acc_all.append(avg_val_acc)

	return n_estimators, train_acc_all, val_acc_all

n_estimators, train_acc_all, val_acc_all = rf_parameter_tune(X_train_tfidf, train_val_y)

best_estimator = n_estimators[np.argmax(val_acc_all)]
rf = RandomForestClassifier(n_estimators=best_estimator, max_depth=20)
rf.fit(X_train_tfidf, train_val_y)
train_acc = rf.score(X_train_tfidf, train_val_y)
test_acc = rf.score(X_test_tfidf, test_y)
##########################

plt.plot(n_estimators, train_acc_all, marker='.', label="Training accuracy")
plt.plot(n_estimators, val_acc_all, marker='.', label="Validation accuracy")
plt.xlabel('n_estimator')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 4 xgboost
def xgb_parameter_tune(train_val_X, train_val_y):
	rates = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
	train_acc_all = []
	val_acc_all = []
	
	kf = KFold(n_splits = 5)
	for rate in rates:
		train_acc = []
		val_acc = []
		
		for train_index, val_index in kf.split(train_val_X):
			train_X = train_val_X[train_index,:]
			val_X = train_val_X[val_index,:]

			train_y = train_val_y[train_index]
			val_y = train_val_y[val_index]
	            
			gbm = xgb.XGBClassifier(learning_rate=rate, n_estimators=200, max_depth=20, random_state=1)
			gbm.fit(train_X, train_y)
			train_acc.append(gbm.score(train_X, train_y))
			val_acc.append(gbm.score(val_X, val_y))
		
		avg_train_acc = sum(train_acc) / len(train_acc)
		avg_val_acc = sum(val_acc) / len(val_acc)
		print("rate: ", rate)
		print("Training accuracy: ", avg_train_acc * 100, "%")
		print("Validation accuracy: ", avg_val_acc * 100, "%")

		train_acc_all.append(avg_train_acc)
		val_acc_all.append(avg_val_acc)

	return rates, train_acc_all, val_acc_all

rates, train_acc_all, val_acc_all = xgb_parameter_tune(X_train_tfidf, train_val_y)

best_rate = rates[np.argmax(val_acc_all)]
gbm = xgb.XGBClassifier(learning_rate=best_rate, n_estimators=200, max_depth=20, random_state=1)
gbm.fit(X_train_tfidf, train_val_y)
train_acc = gbm.score(X_train_tfidf, train_val_y)
test_acc = gbm.score(X_test_tfidf, test_y)

plt.plot(rates, train_acc_all, marker='.', label="Training accuracy")
plt.plot(rates, val_acc_all, marker='.', label="Validation accuracy")
plt.xlabel('learning rate')
plt.ylabel('Accuracy')
plt.legend()
plt.show()