import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier  # gini
from utils import get_data, ProgressBar, calc_measures
from feature_selection import __select_k_best
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.model_selection import train_test_split

# temp_warns = 0
# def warn(*args, **kwargs):
#     global temp_warns
#     temp_warns += 1
# import warnings
# warnings.warn = warn
files = ['spam.csv', 'sonar.csv', 'urbanLandCover.csv', 'semeion.csv', 'sceneCsvBeach.csv', 'madelon.csv']
ranks = ['random_forest', 'gini', 'info_gain', 'chi2']


def intersection(lst1, lst2):
    lst3 = [list(filter(lambda x: x in lst1, sublist)) for sublist in lst2]
    return lst3


k = 20
for file in files:
    df = pd.read_csv(f'calculated/{file}')
    slope = df.sort_values('slope_rank', ascending=False)['feature'][:k].to_list()
    bests = []
    for rank in ranks:
        best = df.sort_values(rank, ascending=False)['feature'][:k].to_list()
        bests.append(best)
    print(f"File: {file}\nIntersections with slope rank for each rank:")
    for rank, inter in zip(ranks, intersection(slope, bests)):
        print(f'  {rank}, {len(inter)} intersections: {inter}')

    X, y = get_data(f'numeric_db/{file}')
    X = X[slope]  # get best feature columns according to slope rank
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = RandomForestClassifier(n_estimators=100)
    # clf.fit(X_train, y_train)
    # mes = calc_measures(clf, X_test, y_test)
    clf.fit(X, y)
    mes = calc_measures(clf, X, y)
    print('Measures of training by best slope features only (Random Forest):')
    for measure, value in mes.items():
        print(f'  {measure}: %.4f' % value)
    print()
