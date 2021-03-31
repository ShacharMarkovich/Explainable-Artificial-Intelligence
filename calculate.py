import os
import sys
import argparse
import numpy as np
import pandas as pd
from lib import get_data, ProgressBar

files = []
score_funcs = []
brute = False
summery = pd.DataFrame()


class Status:
    NONE = np.nan
    NEW = 'new'
    BRUTE = 'changed'
    SAME = 'same'
    ERROR = 'error'


def parse_flags():
    parser = argparse.ArgumentParser(description='Run this to calculate scores for features from databases in '
                                                 '"numeric_db", using score functions from "feature_selection" module, '
                                                 'and save the results to matching files in "calculated".')
    parser.add_argument('-f', '--files', nargs='*', help='The files to add or modify. Files must be found in '
                                                         '"numeric_db" directory and be CSV files.')
    parser.add_argument('-sf', '--score_funcs', nargs='+', help='The score functions for the features.\n'
                                                                'All functions MUST appear in "feature_selection"'
                                                                'module. Signature: '
                                                                'def <name>(X: DataFrame, y:Series, k: Union[int, str]'
                                                                ', score: bool) ->  Tuple[list, list]')
    parser.add_argument('-a', '--add', action='store_true', help='Use this flag with "--files" flag '
                                                                 'to calculate for all files in "numeric_db" that'
                                                                 ' were not calculated, or without "--files" to force '
                                                                 'recalculation of a calculated file (empty flag means '
                                                                 'the whole "calculated" directory). NOT RECOMMENDED.')
    args = parser.parse_args()
    get_files(args.files, args.add)
    get_score_funcs(args.score_funcs)
    if not files or not score_funcs:
        if not files:
            print('No valid files found.')
        if not score_funcs:
            print('No valid score functions found')
        print('Aborting')
        sys.exit()
    global brute
    brute = args.add


def get_files(input_files, add):
    global files
    if input_files:
        summery['files'] = input_files
        for file in input_files:
            if not file.endswith('.csv'):
                file = file + '.csv'
            if not os.path.exists(f'numeric_db/{file}'):
                print(f'{file} was not found in "numeric_db" directory')
                continue
            files.append(file)
    elif add and input_files is not None:
        files = os.listdir('numeric_db')
        summery['files'] = files
    else:
        files = os.listdir('calculated')
        summery['files'] = files
    summery.set_index('files', drop=True, inplace=True)


def get_score_funcs(input_funcs):
    global score_funcs
    from lib import feature_selection

    if not input_funcs:
        input_funcs = [func for func in dir(feature_selection) if not func.startswith('__')]

    for func in input_funcs:
        summery[func] = Status.NONE
        try:
            sfunc = getattr(feature_selection, func)
            score_funcs.append(sfunc)
        except AttributeError:
            print(f'{func} was not found in "feature_selection" module.')
        except Exception as e:
            print(f'Unknown error importing {func}: {str(e)}')


def handle(file, bar):
    x, y = get_data(f'numeric_db/{file}')
    exists = os.path.exists(f'calculated/{file}')
    df = pd.read_csv(f'calculated/{file}') if exists else pd.DataFrame()
    df['feature'] = df['feature'] if exists else list(x)
    df.set_index('feature', drop=True, inplace=True)
    for score_func in score_funcs:
        bar.suffix = score_func.__name__
        if (isin := score_func.__name__ in list(df)) and not brute:
            summery.loc[file, score_func.__name__] = Status.SAME
        else:
            try:
                names, scores = score_func(x, y, k='all', score=True)
                df.loc[names, score_func.__name__] = scores
                summery.loc[file, score_func.__name__] = Status.NEW if not isin else Status.BRUTE
            except Exception as e:
                bar.note(f'Error calculating {score_func.__name__} for file {file}:  {str(e)}')
                summery.loc[file, score_func.__name__] = Status.ERROR
        bar.increment()
    df.to_csv(f'calculated/{file}')


def main():
    parse_flags()
    bar = ProgressBar(len(files)*len(score_funcs))
    for file in files:
        bar.prefix = file + ': '
        handle(file, bar)
    bar.prefix = ''
    bar.suffix = 'Completed'
    del bar
    print('\nSummery:')
    print(summery)


if __name__ == '__main__':
    main()
