from tkinter import W
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import tree  
from sklearn.tree import plot_tree
from collections import OrderedDict


data=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data", header=None)
data.fillna(0) 


##### data info #####

# 48 continuous real [0,100] attributes of type word_freq_WORD
# = percentage of words in the e-mail that match WORD, i.e. 100 * (number of times the WORD appears in the e-mail) / total number of words in e-mail. A "word" in this case is any string of alphanumeric characters bounded by non-alphanumeric characters or end-of-string.

# 6 continuous real [0,100] attributes of type char_freq_CHAR]
# = percentage of characters in the e-mail that match CHAR, i.e. 100 * (number of CHAR occurences) / total characters in e-mail

# 1 continuous real [1,...] attribute of type capital_run_length_average
# = average length of uninterrupted sequences of capital letters

# 1 continuous integer [1,...] attribute of type capital_run_length_longest
# = length of longest uninterrupted sequence of capital letters

# 1 continuous integer [1,...] attribute of type capital_run_length_total
# = sum of length of uninterrupted sequences of capital letters
# = total number of capital letters in the e-mail

# 1 nominal {0,1} class attribute of type spam
# = denotes whether the e-mail was considered spam (1) or not (0), i.e. unsolicited commercial e-mail.



##### data preparation ######

# function to process and separate data as required (used a switch variable "i" to determine which kind of separation is requested)
def sep(data, i=2): # 1-separated data, 2-x, y data, 3-training and testing data
    word_freq=data.iloc[:,0:48]
    char_freq=data.iloc[:,48:54]
    avg_cap_ln=data.iloc[:,54]
    max_cap_ln=data.iloc[:,55]
    sum_cap_ln=data.iloc[:,56]
    spam_yn=data.iloc[:,57]
    if i==1:
        return word_freq, char_freq, avg_cap_ln, max_cap_ln, sum_cap_ln, spam_yn
    elif i==2:
        return data.iloc[:,0:57],spam_yn
    elif i==3:
        train, test = train_test_split(data, train_size=0.8, random_state=25)
        x_train=train.iloc[:,0:57]
        x_test=test.iloc[:,0:57]
        y_train=train.iloc[:,57]
        y_test=test.iloc[:,57]
        return x_train, x_test, y_train, y_test



# Part a: CART model
def part_a(x,y, i=1, criterion='gini', splitter='best', mdepth=None, clweight=None, minleaf=1000):
    x, y=sep(data)
    model=tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=mdepth, class_weight=clweight, min_samples_leaf=minleaf, random_state=0,)
    clf=model.fit(x,y)
    pred_labels=model.predict(x)
    # Tree summary and model evaluation metrics 
    print("------------Tree Summary------------")
    print("Classes: ", clf.classes_)
    print('Tree Depth: ', clf.tree_.max_depth)
    print("No. of leaves: ", clf.tree_.n_leaves)
    print("No. of features: ", clf.n_features_in_)
    print("------------------------------------\n") 
    print("------------Evaluation on Data------------")
    score=model.score(x,y)
    print("Accuracy Score: ",  score)
    print(classification_report(y, pred_labels))
    print("------------------------------------\n")
    fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(clf,  feature_names=x.columns, 
                                class_names=[str(list(clf.classes_)[0]), str(list(clf.classes_)[1])],
                                filled=True, 
                                rounded=True, 
                                #rotate=True,
                                 ) 
    fig.savefig(r"C:\Users\rishi\OneDrive\Desktop\Georgia Tech\Homeworks\Comp Stat\Homework5\decision_tree.png")
    return x, y, clf, fig



# part b: Random Forest Classification
def part_b():
    x,y=sep(data)
    rf=RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt')
    rf.fit(x,y)
    rf_predictions=rf.predict(x)
    rf_probs = rf.predict_proba(x)[:, 1]
    print(rf_predictions)
    print(rf_probs)
    from sklearn.metrics import roc_auc_score

    # Calculate roc auc
    roc_value = roc_auc_score(y, rf_probs)
    print(roc_value)

    # Random forest figure
    fig = plt.figure(figsize=(15, 10))
    plot_tree(rf.estimators_[0], 
            filled=True, impurity=True, 
            rounded=True)
    fig.savefig(r'C:\Users\rishi\OneDrive\Desktop\Georgia Tech\Homeworks\Comp Stat\Homework5\rfc_trees.png')



# part c-i) CART (Train-Test)
def part_c_i():
    x_train, x_test, y_train, y_test=sep(data, i=3)
    # part c-i: CART model
    model=tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, class_weight=None, min_samples_leaf=1000, random_state=0,)
    clf=model.fit(x_train,y_train)
    pred_labels=model.predict(x_test)
    # Tree summary and model evaluation metrics 
    print("------------Tree Summary------------")
    print("Classes: ", clf.classes_)
    print('Tree Depth: ', clf.tree_.max_depth)
    print("No. of leaves: ", clf.tree_.n_leaves)
    print("No. of features: ", clf.n_features_in_)
    print("------------------------------------\n") 
    print("------------Evaluation on Data------------")
    score=model.score(x_test,y_test)
    print("Accuracy Score: ",  score)
    print(classification_report(y_test, pred_labels))
    print("------------------------------------\n")
    fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(clf,  feature_names=x_train.columns, 
                                class_names=[str(list(clf.classes_)[0]), str(list(clf.classes_)[1])],
                                filled=True, 
                                rounded=True, 
                                #rotate=True,
                                ) 
    fig.savefig(r"C:\Users\rishi\OneDrive\Desktop\Georgia Tech\Homeworks\Comp Stat\Homework5\decision_tree_test_train.png")



# Part c-ii) Random Forest Classification (Train-Test)
def part_c_ii():

    x_train, x_test, y_train, y_test=sep(data, i=3)
    rf=RandomForestClassifier(n_estimators=100,warm_start=True, bootstrap=True, max_features='sqrt', oob_score=True)
    rf.fit(x_train,y_train)
    rf_predictions=rf.predict(x_test)
    rf_probs = rf.predict_proba(x_test)[:, 1]
    from sklearn.metrics import roc_auc_score

    # Calculate roc auc
    roc_value = roc_auc_score(y_test, rf_probs)
    print(roc_value)

    # # Random forest figure
    # fig = plt.figure(figsize=(15, 10))
    # plot_tree(rf.estimators_[0], 
    #         filled=True, impurity=True, 
    #         rounded=True)

    RANDOM_STATE=123
    ensemble_clfs = [
        (
            "RandomForestClassifier, max_features='sqrt'",
            RandomForestClassifier(
                warm_start=True,
                oob_score=True,
                max_features="sqrt",
                random_state=RANDOM_STATE,
            ),
        )]
    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
    # Range of `n_estimators` values to explore.
    min_estimators = 15
    max_estimators = 2000

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1, 5):
            clf.set_params(n_estimators=i)
            clf.fit(x_train, y_train)

            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))
    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()


def main():
    print("Part a: ")
    X, Y=sep(data)
    x,y,clf,fig=part_a(X,Y, 'gini', 'best', mdepth=None, clweight=None, minleaf=1000)
    print("\n\nPart b: ")
    part_b()
    print("\n\nPart c] - i:")
    part_c_i()
    print("\n\nPart c]-ii:")
    part_c_ii()


if __name__=="__main__":
    main()