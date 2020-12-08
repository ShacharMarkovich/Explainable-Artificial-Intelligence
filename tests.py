import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier  # gini
from utils import get_data, ProgressBar
from feature_selection import select_k_best
from sklearn.feature_selection import mutual_info_classif, chi2
# import warnings
# warnings.filterwarnings('ignore')
files = ['spam.csv', 'sonar.csv', 'urbanLandCover.csv', 'semeion.csv', 'sceneCsvBeach.csv', 'madelon.csv']


def intersection(lst1, lst2):
    lst3 = [list(filter(lambda x: x in lst1, sublist)) for sublist in lst2]
    return lst3


for file in files:
    df = pd.read_csv(f'calculated/{file}')
    slope = df.sort_values('slope_rank', ascending=False)['feature'][:7].to_list()
    bests = []
    for rank in df.columns[2:]:
        best = df.sort_values(rank, ascending=False)['feature'][:7].to_list()
        bests.append(best)

    print(intersection(slope, bests))
