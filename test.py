import argparse
import os
from typing import List

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from lib import ProgressBar, get_data, calc_measures, k_best, intersection

files = os.listdir('calculated')
ks = {7}
bar: ProgressBar = None
do_mes = True
do_int = True


# region Parse Args
def get_files(input_files: List[str]):
    """
    Parse file name into `files` global var.

    :param input_files: input flies name
    """
    global files
    if input_files:
        files = []
        for file in input_files:
            if not file.endswith('.csv'):
                file = file + '.csv'
            if not os.path.exists(f'calculated/{file}'):
                print(f'{file} was not found in "calculated" directory')
                continue
            files.append(file)


def get_ks(input_ks):
    """
    Parse the wanted "k best features" amount to `ks` global var.

    :param input_ks: the ks features amount.
    """
    global ks
    try:
        if input_ks:
            ks = set(sum(((list(range(*[int(b) + c for c, b in enumerate(a.split('-'))]))
                           if '-' in a else [int(a)]) for a in input_ks.split(',')), []))
    except Exception:
        print("'-k' was not in the right format. aborting.")
        exit()


def parse():
    """
    parse the line arguments.
    """
    global do_int, do_mes
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', nargs='*', help='Files in "calculated" to evaluate')
    parser.add_argument('-k', '--Ks', help="all Ks to evaluate. single numbers, ranges separated by '-' (a-b)"
                                           " or any combination separated by ',' are accepted")
    parser.add_argument('-m', '--mes', action='store_true', help='use this flag in case you want to calculate only'
                                                                 ' measurements')
    parser.add_argument('-i', '--int', action='store_true', help='use this flag in case you want to calculate only'
                                                                 ' intersections')

    args = parser.parse_args()

    get_files(args.files)
    get_ks(args.Ks)

    # set flags
    if args.mes and not args.int:
        do_int = False
    if not args.mes and args.int:
        do_mes = False

    # announce for a problem (no input files name / k best features amount
    if not files or not ks:
        if not files:
            print('No valid files found.')
        if not ks:
            print('No valid K found')
        print('Aborting')
        exit()


# endregion


# region Helper Functions

def set_bar():
    """
    Initialize the ProgressBar
    """
    global bar
    bar_len = 0
    for file in files:
        df = pd.read_csv(f'calculated/{file}')
        num_ranks = len(list(df)[1:])  # count amount of features selection algorithms
        if do_mes:
            bar_len += num_ranks
        if do_int:
            bar_len += sum(range(num_ranks))

    bar = ProgressBar(bar_len * len(ks), length=80)


# endregion


def handle_k(k: int):
    """
    Calc best rank for each file.
    Than, measures/calc intersections all the best rank
    And finally write it to file.

    :param k: best features amount to select
    """
    mes_sum = [] if do_mes else None
    int_sum = [] if do_int else None

    for file in files:
        bar.prefix = f'k{k}, {file[:-4]}'  # remove the .csv from file name
        df = pd.read_csv(f'calculated/{file}')
        # calc bests for each rank
        bests = {rank: k_best(df, rank, k) for rank in list(df)[1:]}  # remove the attribute name

        if do_mes:
            # calc measures for all rank's bests
            x, y = get_data(f'numeric_db/{file}')
            mes_df = pd.DataFrame()
            for rank, best in bests.items():
                bar.suffix = f'{rank} mes'
                clf = RandomForestClassifier(n_estimators=100)
                mes = calc_measures(clf, x[best], y)
                mes_df.loc[rank, mes.keys()] = mes.values()
                bar.increment()

            mes_sum.append(mes_df)

        if do_int:
            # calc intersections for all rank's bests
            int_df = pd.DataFrame()
            for i, rank1 in enumerate(bests):
                bar.suffix = f'{rank1} int'
                int_df.loc[rank1, rank1] = k
                for rank2 in list(bests)[i + 1:]:
                    val = len(intersection(bests[rank1], bests[rank2]))
                    int_df.loc[rank1, rank2] = int_df.loc[rank2, rank1] = val
                    bar.increment()

            int_sum.append(int_df)

    # write to file:
    if do_mes:
        final = pd.concat(mes_sum, axis=0, keys=files)
        final.index.set_names(['file', 'rank'], inplace=True)
        final.to_csv(f'tested/mes_k{k}.csv')

    if do_int:
        final = pd.concat(int_sum, axis=0, keys=files)
        final.index.set_names(['file', 'rank'], inplace=True)
        final.to_csv(f'tested/int_k{k}.csv')


# region Reports
def report(kind_pref: str):
    """
    Write the report in `tested` folder.

    :param kind_pref: test kind prefix
    """
    lst = [f for f in os.listdir('tested') if f.startswith(kind_pref)]  # get fit files names
    df_sum = []
    # for each k best selected file,
    for file in lst:
        df = pd.read_csv(f'tested/{file}')
        # get the mean intersection/measure of each rank
        df_sum.append(df.groupby('rank').mean())

    final = pd.concat(df_sum, axis=0, keys=[x[4:-4] for x in lst])
    final.index.set_names('k', level=0, inplace=True)
    final.to_csv(f'tested/report_{kind_pref}.csv')


# endregion


def main():
    global bar

    parse()
    set_bar()

    for k in ks:
        handle_k(k)

    if do_mes:
        report('mes')

    if do_int:
        report('int')

    bar.prefix = ''
    bar.suffix = 'Completed'
    del bar


if __name__ == '__main__':
    main()
