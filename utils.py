import pandas as pd


class ProgressBar:
    def __init__(self, total, prefix='', suffix='', length=100):
        self.total = total
        self.iteration = 0
        self._prefix = prefix
        self._suffix = suffix
        self.length = length
        self._warned = False
        self.print_bar()

    def get_bar(self):
        percent = "{0:.1f}".format(self.length * (self.iteration / float(self.total)))
        filled_length = int(self.length * self.iteration // self.total)
        bar = '█' * filled_length + '-' * (self.length - filled_length) \
            if filled_length <= self.length else '█' * self.length
        full_bar = f'\r{self._prefix} |{bar}| {percent}% {self._suffix}'
        return full_bar

    def print_bar(self, end='\r'):
        print(self.get_bar(), end=end)
        # Print New Line on Complete

    def increment(self, num=1):
        self.iteration += num
        if not self._warned and self.iteration > self.total:
            self.__warn()
        else:
            self.print_bar()

    def note(self, note):
        num_space = len(self.get_bar()) - len(note)
        print(note + ' ' * num_space)
        self.print_bar()

    @property
    def prefix(self):
        return self._prefix

    @prefix.setter
    def prefix(self, prefix):
        curr_len = len(self.get_bar())
        new_len = curr_len - len(self._prefix) + len(prefix)
        self._prefix = prefix
        self.print_bar(' ' * (curr_len - new_len) + '\r')

    @property
    def suffix(self):
        return self._suffix

    @suffix.setter
    def suffix(self, suffix):
        curr_len = len(self.get_bar())
        new_len = curr_len - len(self._suffix) + len(suffix)
        self._suffix = suffix
        self.print_bar(' ' * (curr_len - new_len) + '\r')

    def __del__(self):
        if self.iteration == self.total:
            print()
        else:
            self.__warn()
        del self

    def __warn(self):
        if not self._warned:
            self.suffix += self._suffix + ' ! ⚠ !'
            self._warned = True


def get_data(file, drop=None):
    drop = drop or []
    data = pd.read_csv(file)
    data.rename(columns={'Class': 'class'}, inplace=True)
    x = data.drop(labels=['class'] + drop, axis=1)
    y = data['class']
    return x, y


def calc_measures(classifier, data_set, target):
    """
    Helping function - gives an indicate about how much our calculations are accurate.

    :param classifier: a classifier function
    :param data_set: the input data to fit
    :type data_set: pandas.core.frame.DataFrame
    :param target: the classifier column - The target variable to try to predict
    :type target: pandas.core.series.Series
    :return: 4 accurate measures
    """
    from sklearn.model_selection import cross_validate
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    result = cross_validate(classifier, data_set, target, scoring=scoring, cv=10)
    result = [result['test_' + score].mean() for score in scoring]
    scoring = [score.split('_')[0] for score in scoring]
    return dict(zip(scoring, result))


def sort_range_strings(lst: list):
    """
    Score function - sorted the given list by the value of the range numbers
    :param lst: the list of range values
    """
    lst.sort()
    lst[:-2] = sorted(lst[:-2], key=lambda x: float(x.split('-')[0]))
    lst.insert(0, lst.pop(-2))
