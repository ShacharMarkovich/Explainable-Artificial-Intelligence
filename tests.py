import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier  # gini
from utils import get_data, ProgressBar, calc_measures
from feature_selection import __select_k_best
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.model_selection import train_test_split
import argparse

files = os.listdir('calculated')
k = 20
slope_rank = 'slope_rank'


def parse():
    global files
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', nargs='*', help='Files in "calculated" to evaluate')

    args = parser.parse_args()
    if args.files:
        files = args.files


def intersection(lst1, lst2):
    lst3 = [list(filter(lambda x: x in lst1, sublist)) for sublist in lst2]
    return lst3


def kbest(df, rank):
    return df.sort_values(rank, ascending=False)['feature'][:k].to_list()


def main():
    parse()
    mes_sum = []
    for file in files:
        # init DataFrame
        df = pd.read_csv(f'calculated/{file}')
        ranks = list(df)[1:]
        if slope_rank not in ranks:
            continue
        ranks.remove(slope_rank)

        # calc bests for each rank
        slope = kbest(df, slope_rank)
        bests = {slope_rank: slope}
        for rank in ranks:
            best = kbest(df, rank)
            bests[rank] = best

        # print intersections
        # print(f"File: {file}\nIntersections with slope rank for each rank:")
        # for rank in ranks:
        #     inter = intersection(slope, bests[rank])
        #     print(f'  {rank}, {len(inter)} intersections: {inter}')

        # calc measures for slope's bests
        X, y = get_data(f'numeric_db/{file}')
        # scoring = ['accuracy', 'precision', 'recall', 'f1']
        mes_df = pd.DataFrame()
        for rank in bests:
            clf = RandomForestClassifier(n_estimators=100)
            best_data = X[bests[rank]]  # get best feature columns according to slope rank
            # X_train, X_test, y_train, y_test = train_test_split(best_data, y, test_size=0.33, random_state=42)
            # clf.fit(X_train, y_train)
            # mes = calc_measures(clf, X_test, y_test)
            clf.fit(best_data, y)
            mes = calc_measures(clf, best_data, y)
            mes_df.loc[rank, mes.keys()] = mes.values()
        mes_sum.append(mes_df)
        print(f'finished {file}')

    final = pd.concat(mes_sum, axis=0, keys=files)
    final.to_csv('tests.csv')


if __name__ == '__main__':
    main()
