"""
DO NOT IMPORT ANYTHING GLOBALLY.
DO NOT DECLARE ANY GLOBAL VARIABLE, UNLESS IT STARTS WITH '__'
ANY FUNCTION WITHOUT THIS SIGNATURE:
def <name>(x, y, k, score) -> Tuple[list, list] (returns (names, scores) when score=True),
SHOULD BE PRIVATE (starts with '__'), OR SHOULD BE DECLARED INSIDE ANOTHER FUNCTION.
"""


def slope_rank(x, y, k='all', score=False):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(x, y)
    return __slope_rank(clf, x, k, score)


def info_gain(x, y, k='all', score=False):
    from sklearn.feature_selection import mutual_info_classif
    return __select_k_best(x, y, mutual_info_classif, k, score=score)


def chi2(x, y, k='all', score=False):
    from sklearn.feature_selection import chi2 as chisq
    return __select_k_best(x, y, chisq, k, score=score)


def gini(x, y, k='all', score=False):
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(criterion='gini')
    return __select_k_best(x, y, classifier=clf, k=k, score=score)


def random_forest(x, y, k='all', score=False):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(x, y)
    return __select_k_best(x, y, classifier=clf, k=k, score=score)


def __select_k_best(x, y, score_func=None, k='all',
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


def __slope_rank(classifier, given_x, k='all', score=False, normalized=False):
    """
    select and return the `k` best features, according to the slop rank
    :param classifier: a classifier function
    :param given_x: the input data (names and values)
    :type given_x: pandas.core.frame.DataFrame
    :param k: amount of features to select. `all` means - select all features
    :param score: flag, indicate if to return the score of each value
    :return:
    """
    from sklearn.inspection import partial_dependence
    import numpy as np

    if normalized:
        given_x = preprocessing.StandardScaler().fit_transform(given_x)

    scores = []
    features = list(given_x)
    for feature in features:
        x, y = None, None
        try:
            y, x = partial_dependence(classifier, given_x, feature)
        except ValueError:
            # in the case that the slope is too flat - it solve it
            y, x = partial_dependence(classifier, given_x, feature, percentiles=(0, 1))
        finally:
            # calc tha slope and add it to the scores
            if x is None or y is None:
                scores.append(0)
                continue
            x, y = x[0], y[0]
            line = np.polyfit(x, y, 1)
            scores.append(abs(line[0]))
    names = list(given_x)
    if k != 'all':
        # if user didn't select all of them - select the `k` best of them:
        ind = np.argpartition(scores, -k)[-k:]
        names = np.array(names)[ind]
        scores = np.array(scores)[ind]

    if score:
        return list(names), list(scores)
    else:
        return list(names)
