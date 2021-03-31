"""
DO NOT IMPORT ANYTHING GLOBALLY.
DO NOT DECLARE ANY GLOBAL VARIABLE, UNLESS IT STARTS WITH '__'
ANY FUNCTION WITHOUT THIS SIGNATURE:
def <name>(x, y, k, score) -> Tuple[list, list] (returns (names, scores) when score=True),
SHOULD BE PRIVATE (starts with '__'), OR SHOULD BE DECLARED INSIDE ANOTHER FUNCTION.
"""


# TODO:
#   1. implement more efficient way for "slope rank" (and test)
#   2. implement slope rank with normalization
#   3. implement another slope rank with a different classifier (optional)
#   4. RUN


def slope_rank(x, y, k='all', score=False):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100).fit(x, y)
    return __slope_rank(clf, x, k, score)


def info_gain(x, y, k='all', score=False):
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(criterion='entropy')
    return __select_k_best(x, y, classifier=clf, k=k, score=score)


def chi2(x, y, k='all', score=False):
    from sklearn.feature_selection import chi2 as chisq
    from sklearn.feature_selection import chi2
    return __select_k_best(x, y, score_func=chisq, k=k, score=score)


def gini(x, y, k='all', score=False):
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(criterion='gini')
    return __select_k_best(x, y, classifier=clf, k=k, score=score)


def random_forest(x, y, k='all', score=False):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100)
    return __select_k_best(x, y, classifier=clf, k=k, score=score)


def __select_k_best(x, y, *, score_func=None, k='all',
                    classifier=None, score=False):
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
    import numpy as np
    if classifier is not None:
        fs = classifier
    else:
        from sklearn.feature_selection import SelectKBest
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


def __score(clf, x, feature):
    from sklearn.inspection import partial_dependence
    import numpy as np

    try:
        y, x = partial_dependence(clf, x, feature)
    except ValueError:
        # in the case that the slope is too flat - it solve it
        # y, x = partial_dependence(classifier, given_x, feature, percentiles=(0, 1))
        return 0
    else:
        # calc tha slope and add it to the scores
        # if x is None or y is None:
        #     return 0
        x, y = x[0], y[0]
        line = np.polyfit(x, y, 1)
        return abs(line[0])


def __work(group, score):
    from joblib import parallel_backend
    with parallel_backend('threading', n_jobs=-1):
        ret = {feature: score(feature=feature) for feature in group}
    return ret


def __slope_rank(classifier, given_x, k='all', score=False, n_jobs=4,normalized=False):
    """
    select and return the `k` best features, according to the slop rank
    :param classifier: a classifier function
    :param given_x: the input data (names and values)
    :type given_x: pandas.core.frame.DataFrame
    :param k: amount of features to select. `all` means - select all features
    :param score: flag, indicate if to return the score of each value
    :return:
    """
    from multiprocessing import Pool
    from functools import partial
    import numpy as np

    if normalized:
        given_x = preprocessing.StandardScaler().fit_transform(given_x)

    scores = []
    features = list(given_x)
    s = partial(__score, clf=classifier, x=given_x)
    with Pool(n_jobs) as p:
        a = [(map(str, chunk), s) for chunk in np.array_split(features, n_jobs)]
        results = p.starmap(__work, a)

    result = {}
    for r in results:
        result = {**result, **r}

    names, scores = zip(*result.items())

    if k != 'all':
        # if user didn't select all of them - select the `k` best of them:
        ind = np.argpartition(scores, -k)[-k:]
        names = np.array(names)[ind]
        scores = np.array(scores)[ind]

    if score:
        return list(names), list(scores)
    else:
        return list(names)
