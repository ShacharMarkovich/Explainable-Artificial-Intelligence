import numpy as np
import pandas as pd
from math import ceil, sqrt
import matplotlib.pyplot as plt
from lib import sort_range_strings
from typing import Union, Callable
from sklearn.inspection import plot_partial_dependence, partial_dependence


class InteractivePlots:
    def __init__(self, fig):
        if not fig or not fig.axes:
            return

        self.fig = fig
        hn, lb = fig.axes[0].get_legend_handles_labels()
        leg = fig.legend(hn, lb, 'center right', fontsize=18)

        lines = zip(*[ax.get_lines() for ax in fig.axes])
        self.lined = {}
        for legline, origlines in zip(leg.get_lines(), lines):
            legline.set_picker(True)  # Enable picking on the legend line.
            legline.set_pickradius(15)
            for origline in origlines:
                origline.set_picker(True)  # Enable picking on the plot line.
                origline.set_pickradius(10)
            self.lined[legline] = origlines

        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_pick(self, event):
        self.__update(line=event.artist)

    def on_click(self, event):
        if event.button == 3:
            visible = False
        elif event.button == 2:
            visible = True
        else:
            return

        self.__update(value=visible)

    def __update(self, *, line=None, value=None):
        if not line and value is None:
            return

        if not line or line in self.lined:  # general click or click on legend
            cur_lines = self.lined if not line else {line: self.lined[line]}
            visible = line.get_alpha() != 1 if value is None else value

            for ll, ols in cur_lines.items():
                ll.set_alpha(1.0 if visible else 0.2)
                for ol in ols:
                    ol.set_visible(visible)

        else:  # click on plotted line
            legline, origlines = next(filter(lambda item: line in item[1], self.lined.items()))
            visible = value if value is not None else not line.get_visible()
            line.set_visible(visible)
            if all(ol.get_visible() == visible for ol in origlines):
                legline.set_alpha(1.0 if visible else 0.2)
        self.fig.canvas.draw()


def pdp(classifier, given_x, features: list, slope=None, interactive=None, fig_name: str = "fig", mode=None):
    """
    Show the pdp plot and calculate the slope of each variable

    :param fig_name: the output figure title
    :param classifier: a classifier function
    :param given_x: the input data (names and values)
    :type given_x: pandas.core.frame.DataFrame
    :param features: list of features
    :param slope: flag, indicate if we want to get the slope of each variable
    :param mode
    :param interactive
    """
    # TODO: Add option for a Normalizer (after 'core')
    multi = len(classifier.classes_) > 2
    slope = slope if slope is not None else not multi
    interactive = interactive if interactive is not None else multi

    fig = plt.figure()
    fig.suptitle(fig_name, fontsize=16)
    subplots_num = len(features)
    cols = ceil(sqrt(subplots_num))
    d, r = divmod(subplots_num, cols)
    rows = d + int(bool(r))

    for i, feature in enumerate(features):
        ax = fig.add_subplot(rows, cols, i + 1, title=feature)
        ys, x = partial_dependence(classifier, given_x, feature, grid_resolution=125)  # kind='average' to get a Bunch
        x = x[0]
        for y, cls in zip(ys, classifier.classes_):
            label = cls if multi else 'pdp'
            ax.plot(x, y, label=label)

            if slope:
                the_slope, intercept = np.polyfit(x, y, 1)
                if not multi:
                    ax.text(0.01, 0.01, f'slope = {the_slope:.4f}', fontsize=12, transform=ax.transAxes)

                slx = np.linspace(min(x), max(x))
                sly = the_slope * slx + intercept
                ax.plot(slx, sly, label='slope')

    ret = fig if not interactive else InteractivePlots(fig)

    if mode:
        if 'save' in mode:
            fig.savefig(f"{fig_name}.png", dpi=300)
        if 'show' in mode:
            plt.show()
    return ret


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
