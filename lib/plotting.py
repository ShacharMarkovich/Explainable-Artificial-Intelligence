from typing import Union, Callable
import pandas as pd
from lib import sort_range_strings
from math import ceil, sqrt
import numpy as np
from sklearn.inspection import plot_partial_dependence, partial_dependence


def pdp(classifier, given_x, features: list, slope=True, fig_name: str = ""):
    """
    Show the pdp plot and calculate the slope of each variable

    :param fig_name: the output figure title
    :param classifier: a classifier function
    :param given_x: the input data (names and values)
    :type given_x: pandas.core.frame.DataFrame
    :param features: list of features
    :param slope: flag, indicate if we want to get the slope of each variable
    """
    if not slope:
        plot_partial_dependence(classifier, given_x, features)
        return  # if we don't need to calc the slope - no need for this function anymore!

    # create new figure
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.suptitle(fig_name, fontsize=16)
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
        y, x = partial_dependence(classifier, given_x, features[index])
        x, y = x[0], y[0]
        # noinspection PyTupleAssignmentBalance
        the_slope, intercept = np.polyfit(x, y, 1)
        ax.title.set_text(features[index] + (', slope = %.4f' % the_slope))
        ax.plot(x, y)
        slx = np.linspace(min(x), max(x))
        sly = the_slope * slx + intercept
        ax.plot(slx, sly)
        ax.plot(x, y)
        return ax

    # calculate each subplot. All shared the Y axis values:
    ax1 = subplot(0)
    for i in range(len(features) - 1):
        subplot(i + 1, ax1)

    fig.savefig(f"{fig_name}.png", dpi=300)


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
    if isinstance(feature, int):
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
    import matplotlib.pyplot as plt
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
    from lib.feature_selection import __select_k_best
    # get the score and name of each feature, according to the classifier and/or score function:
    if classifier is not None:  # features' value are numeric
        names, scores = __select_k_best(x, y, classifier=classifier, score=True)
        res = [scores, names]
    else:  # features' value are range
        names, scores = __select_k_best(x, y, score_func=score_func, score=True)
        # associates between features name to there score:
        pairs = list(zip(scores, names))
        pairs.sort(key=lambda p: p[0])
        res = list(zip(*pairs))

    pd.DataFrame(res[0], res[1]).plot.barh()
