import os
import pandas as pd
from pandas.core.groupby.generic import SeriesGroupBy
from sklearn.ensemble import RandomForestClassifier
from lib import get_data, calc_measures, k_best, intersection

k = 20
tested: SeriesGroupBy = None


def a(file):
    df = pd.read_csv(f'calculated/{file}')
    X, y = get_data(f'numeric_db/{file}')

    ranks = list(df)[1:]
    slope = 'slope_rank_minmax'
    others = [r for r in ranks if not r.startswith('slope_rank')]

    ranks = others + [slope]
    bests = {rank: k_best(df, rank, k) for rank in ranks}

    group = tested.get_group(file)
    cur = group.loc[group['rank'].isin(ranks)]
    ret = pd.DataFrame(cur)
    ret.insert(2, '#features', k)

    for other in others:
        b = intersection(bests[slope], bests[other])
        clf = RandomForestClassifier(n_estimators=100)
        mes = calc_measures(clf, X[b], y)

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
        df_sum.append(a(file))

    final = pd.concat(df_sum)
    final.set_index(['file', 'rank'], inplace=True)
    final.to_csv(f'ints.csv')


if __name__ == '__main__':
    main()
