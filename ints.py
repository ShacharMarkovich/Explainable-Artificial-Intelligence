import os

import pandas as pd
from pandas.core.groupby.generic import SeriesGroupBy
from sklearn.ensemble import RandomForestClassifier

from lib import get_data, calc_measures, k_best, intersection

k = 20
tested: SeriesGroupBy = None


def calc_intersections_measures(file: str) -> pd.DataFrame:
    """
    For a given file, return a DataFrame which contains the
    MinMax slop_rank with the other ranks (who not kind of slope_rank) features intersections' measures calculation,
    and for the other ranks, calc the measures and add it to the DataFrame too.

    :param file: the input file name
    :return:  measures' DataFrame
    """
    df = pd.read_csv(f'calculated/{file}')
    x, y = get_data(f'numeric_db/{file}')

    ranks = list(df)[1:]  # drop attributes name's column
    slope = 'slope_rank_minmax'
    others = [rank for rank in ranks if not rank.startswith('slope_rank')]

    ranks = others + [slope]
    bests = {rank: k_best(df, rank, k) for rank in ranks}  # calc bests for each rank

    group = tested.get_group(file)  # get fit group of file's measures
    cur = group.loc[group['rank'].isin(ranks)]  # select ranks which appear in `ranks`
    ret = pd.DataFrame(cur)
    ret.insert(2, '#features', k)

    # calc the measures for each other rank, and add it to DataFrame:
    for other in others:
        b = intersection(bests[slope], bests[other])
        clf = RandomForestClassifier(n_estimators=100)
        mes = calc_measures(clf, x[b], y)

        mes['file'] = file
        mes['rank'] = f'{slope} inter. {other}'
        mes['#features'] = len(b)
        ret = ret.append(mes, ignore_index=True)

    print(ret)
    return ret


def main():
    global tested
    files = os.listdir('calculated')
    tested = pd.read_csv(f'tested/mes_k{k}.csv').groupby('file')

    df_sum = []
    for file in files:
        df_sum.append(calc_intersections_measures(file))

    # Concatenate the data and write it to csv file
    final = pd.concat(df_sum)
    final.set_index(['file', 'rank'], inplace=True)
    final.to_csv(f'ints.csv')


if __name__ == '__main__':
    main()
