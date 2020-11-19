import time
from math import *
from typing import Union, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence


# region Private


def __plot_bar_chart(x, y, feature: Union[int, str], axis=None,
                     legend=True, show_rate=True, show_y_values=True,
                     sort_cat: Callable[[list], None] = None):
    """
    create bar_chart
    :param x: the input data (names and values)
    :type x: pandas.core.frame.DataFrame
    :param y: the classifier column (name and values)
    :type y: pandas.core.series.Series
    :param feature: feature name
    :param axis: The axes of the subplot.
    :param legend: flag, indicate
    :param show_rate: flag, indicate
    :param show_y_values: flag, indicate
    :param sort_cat:
    :return:
    """
    if isinstance(ifeature, int):
        feature = list(x)[feature]
    col = x[feature]
    classes = list(dict.fromkeys(y))
    categories = list(dict.fromkeys(col))
    if sort_cat is not None:
        sort_cat(categories)
    bars = {cl: [0] * len(categories) for cl in classes}

    for cat, cl in zip(col, y):
        bars[cl][categories.index(cat)] += 1

    df = pd.DataFrame(bars, index=categories)
    ax = df.plot.bar(rot=0, title=feature, ax=axis, legend=False, sharey=True)
    handles, labels = ax.get_legend_handles_labels()

    if show_rate:
        rate = df.div(df.sum(1), axis=0)
        ax2 = ax.twinx()
        ax2.plot(range(len(categories)), rate[1], 'red', label='Rate', linestyle='--', linewidth=2.0)
        ax2.set_ylim(0, 1)
        for i, v in enumerate(rate[1]):
            ax2.text(i, v + 0.01, "%.1f%%" % (v * 100), ha="center")
        hn, lb = ax2.get_legend_handles_labels()
        handles += hn
        labels += lb

    if show_y_values:
        for p in ax.patches:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            ax.annotate(f'{height}', (x + width / 2, y + height * 1.02), ha='center')

    if legend:
        ax.legend(handles, labels)

    return handles, labels


def sort_range_strings(lst: list):
    """
    Score function - sorted the given list by the value of the range numbers
    :param lst: the list of range values
    """
    lst.sort()
    lst[:-2] = sorted(lst[:-2], key=lambda x: float(x.split('-')[0]))
    lst.insert(0, lst.pop(-2))


# endregion


# region Public


def plot_bar_chart(x, y, features: Union[int, str, list],
                   legend=True, show_rate=True, show_y_values=True,
                   sort_cat: Callable[[list], None] = sort_range_strings):
    """
    create bar_chart

    :param x: the input data (names and values)
    :type x: pandas.core.frame.DataFrame
    :param y: the classifier column (name and values)
    :type y: pandas.core.series.Series
    :param features: features name + score
    :param legend: flag, indicate
    :param show_rate: flag, indicate
    :param show_y_values: flag, indicate
    :param sort_cat:
    """
    if not isinstance(features, list):
        __plot_bar_chart(x, y, features, legend=legend,
                         show_rate=show_rate, show_y_values=show_y_values, sort_cat=sort_cat)
        return

    # create new figure with enough space for all bar chart:
    fig = plt.figure()
    subplots_num = len(features)
    cols = ceil(sqrt(subplots_num))
    rows = subplots_num // cols + int(bool(subplots_num % cols))

    for i, feature in enumerate(features):
        ax = fig.add_subplot(rows, cols, i + 1)
        # create bar_chart:
        handles, labels = __plot_bar_chart(x, y, feature, ax, legend=False,
                                           show_rate=show_rate, show_y_values=show_y_values, sort_cat=sort_cat)
        if legend and i == 0:  # adding legend
            fig.legend(handles, labels, loc='upper right', prop={'size': 10})


def calc_measures(classifier, data_set, target) -> tuple:
    """
    Helping function - gives an indicate about how much our calculations are accurate.

    :param classifier: a classifier function
    :param data_set: the input data to fit
    :type data_set: pandas.core.frame.DataFrame
    :param target: the classifier column - The target variable to try to predict
    :type target: pandas.core.series.Series
    :return: 4 accurate measures
    """
    from sklearn.model_selection import cross_val_score
    accuracy = cross_val_score(classifier, data_set, target, scoring='accuracy', cv=10)
    accuracy = accuracy.mean()
    precision = cross_val_score(classifier, data_set, target, scoring='precision_weighted', cv=10)
    precision = precision.mean()
    recall = cross_val_score(classifier, data_set, target, scoring='recall_weighted', cv=10)
    recall = recall.mean()
    f1 = cross_val_score(classifier, data_set, target, scoring='f1_weighted', cv=10)
    f1 = f1.mean()
    return accuracy, precision, recall, f1


def slope_rank(classifier, X, k: Union[str, int] = 'all', score=False) -> Union[tuple, list]:
    """
    select and return the `k` best features, according to the slop rank

    :param classifier: a classifier function
    :param X: the input data (names and values)
    :type X: pandas.core.frame.DataFrame
    :param k: amount of features to select. `all` means - select all features
    :param score: flag, indicate if to return the score of each value
    :return:
    """
    scores = []
    for feature in list(X):
        x, y = None, None
        try:
            y, x = partial_dependence(classifier, X, feature)
        except ValueError:
            # in the case that the slop too flat - it solve it
            y, x = partial_dependence(classifier, X, feature, percentiles=(0, 1))
        finally:
            # calc tha slop and add it to the scores
            if x is None or y is None:
                scores.append(0)
                continue
            x, y = x[0], y[0]
            line = np.polyfit(x, y, 1)
            scores.append(abs(line[0]))

    names = list(X)
    if k != 'all':
        # if user didn't select all of them - select the `k` best of them:
        ind = np.argpartition(scores, -k)[-k:]
        names = np.array(names)[ind]
        scores = np.array(scores)[ind]

    if score:
        return list(names), list(scores)
    else:
        return list(names)


def plot_feature_importance(x, y, score_func=None, classifier=None):
    """
    Show in figure the importance of each feature

    :param x: the input data (names and values)
    :type x: pandas.core.frame.DataFrame
    :param y: the classifier column (name and values)
    :type y: pandas.core.series.Series
    :param score_func: mathematical function for determining the quality of each feature
    :param classifier: a classifier function

    """
    # get the score and name of each feature, according to the classifier and/or score function:
    if classifier is not None:
        names, scores = select_k_best(x, y, classifier=classifier, score=True)
        res = [scores, names]
    else:
        names, scores = select_k_best(x, y, score_func=score_func, score=True)
        # associates between features name to there score:
        pairs = list(zip(scores, names))
        pairs.sort(key=lambda p: p[0])
        res = list(zip(*pairs))

    pd.DataFrame(res[0], res[1]).plot.barh()


def select_k_best(x, y, score_func=None, k: Union[int, str] = 'all',
                  classifier=None, score=False) -> Union[tuple, list]:
    """
    select the `k` best features, using a classifier object or score function.

    :param x: the input data (names and values)
    :type x: pandas.core.frame.DataFrame
    :param y: the classifier column (name and values)
    :type y: pandas.core.series.Series
    :param score_func: mathematical function for determining the quality of each feature
    :param k: amount of the best features. `all` means - return all features
    :param classifier: a classifier function
    :param score: flag, indicate if to return the score of each value
    :return:  k best features' names list. optional: score of each value
    """
    if classifier is not None:
        fs = classifier
    else:
        fs = SelectKBest(score_func, k=k)

    try:
        fs.fit(x, y)  # will work iff the features' values are numeric
    except ValueError:
        # for not-numeric(=range) values,
        # Encode categorical features as an integer array
        from sklearn.preprocessing import OrdinalEncoder
        enc = OrdinalEncoder()
        x_new = enc.fit_transform(x)
        fs.fit(x_new, y)

    if classifier is not None:
        # get all names and there scores:
        names = np.array(list(x))
        scores = fs.feature_importances_
        if k != 'all':
            # if user didn't select all of them - select the `k` best of them:
            ind = np.argpartition(scores, -k)[-k:]
            names = names[ind]
            scores = scores[ind]
    else:
        # select the `k` best features...
        names = x.columns.values[fs.get_support()]
        scores = fs.scores_[fs.get_support()]

    names = [str(val) for val in names]  # casting each value to str, in order to prevent exception
    if score:
        return names, list(scores)
    else:
        return names


def pdp(classifier, X, features: list, slope=True):
    """
    Show the pdp plot and calculate the slope of each variable

    :param classifier: a classifier function
    :param X: the input data (names and values)
    :type X: pandas.core.frame.DataFrame
    :param features: list of features
    :param slope: flag, indicate if we want to get the slope of each variable
    """
    if not slope:
        plot_partial_dependence(classifier, X, features)
        return  # if we dont need to calc the slop - not need anymore this function!

    # create new figure
    fig = plt.figure()
    subplots_num = len(features)
    cols = ceil(sqrt(subplots_num))
    rows = subplots_num // cols + int(bool(subplots_num % cols))

    def subplot(index, sharey=None):
        """
        Calculate the slop of each variable and show it in different subplot

        :param index: feature's index
        :type index: int
        :param sharey: the value to show in the shared Y axis
        :return: a new subplot
        """
        ax = fig.add_subplot(rows, cols, index + 1, sharey=sharey)
        y, x = partial_dependence(classifier, X, features[index])
        x, y = x[0], y[0]
        slope, intercept = np.polyfit(x, y, 1)
        ax.title.set_text(features[index] + (', slope = %.4f' % slope))
        ax.plot(x, y)
        slx = np.linspace(min(x), max(x))
        sly = slope * slx + intercept
        ax.plot(slx, sly)
        ax.plot(x, y)
        return ax

    # calculate each subplot. All shared the Y axis values:
    ax1 = subplot(0)
    for i in range(len(features) - 1):
        subplot(i + 1, ax1)


# endregion


def todo_split_it_to_what_you_need_to_split_to():
    # region Data Initializing
    start = time.time()
    data1 = pd.read_csv("spam.csv")  # numeric value in each cell
    X1 = data1.drop(labels=['class'], axis=1)
    y1 = data1['class']

    data2 = pd.read_csv("spam-1.csv")  # range value in each cell
    X2 = data2.drop(labels=['class', 'Selected'], axis=1)
    y2 = data2['class']
    # endregion

    # TODO: split this gigantic function to what you need to split to..

    # Classifier Initialization:
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X1, y1)

    # print(sorted(clf.feature_importances_))
    # print(list(X1))
    # Selector Initialization:
    # selector = SelectKBest(mutual_info_classif, k=6)
    # selector.fit(X1, y1)

    # Measurements for Selected Classifier:
    # print("\nMeasures:\n accuracy = %.4f\n precision = %.4f\n recall = %.4f\n f1 = %.4f" % calc_measures(clf, X1, y1))
    # print(slope_rank(clf, X1, k=6, score=True))
    # plot_partial_dependence(clf, X1, ['word_freq_edu', 'word_freq_meeting', 'char_freq_!', 'word_freq_000',
    #               'char_freq_$', 'word_freq_remove'])
    # print(line)
    # Plot Feature Importance for Categorical or Continuous Data
    # plot_feature_importance(X1, y1, classifier=clf)
    # plot_feature_importance(X2, y2, mutual_info_classif)

    # Professor, we save you the time to run the code yourself and tell you that those are the most influential words:)
    best_slope = ['word_freq_edu', 'word_freq_meeting', 'char_freq_!', 'word_freq_000', 'char_freq_$',
                  'word_freq_remove']
    best_forest = select_k_best(X1, y1, classifier=clf, k=6)
    pdp(clf, X1, best_slope)
    pdp(clf, X1, best_forest)

    # Get K Best Features Names for Categorical or Continuous Data
    # plot_features = select_k_best(X1, y1, classifier=clf, k=6)
    # print(plot_features)
    # plot_features = select_k_best(X2, y2, mutual_info_classif, 4)

    # Get K Best Features Names & Scores for Categorical or Continuous Data
    # names, scores = select_k_best(X1, y1, mutual_info_classif, 6, True)
    # names, scores = select_k_best(X2, y2, mutual_info_classif, 6, True)

    # PDP for Continuous Data only
    # plot_partial_dependence(clf, X1, [0, 1, 2, 3])
    # pdp(clf, X1, plot_features)
    # print(x[0])
    # print(y[0])
    # print(partial_dependence(clf, X1, ['word_freq_address']))
    # Plot Bar Chart for Categorical Data only
    # plot_bar_chart(X2, y2, plot_features)
    # slopes, features = slope_rank(clf, X1)
    # pd.DataFrame(slopes, features).plot.barh()
    print("\nTime Elapsed: %.4f seconds." % (time.time() - start))
    plt.show()


if __name__ == "__main__":
    todo_split_it_to_what_you_need_to_split_to()
