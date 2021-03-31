# Run here whatever

import warnings
from sklearn.ensemble import RandomForestClassifier
from lib.feature_selection import __slope_rank
from multiprocessing import Pool, freeze_support
import time
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
from lib import get_data

warnings.filterwarnings('ignore')
x, y = get_data('numeric_db/spam.csv')
fs = list(x)
clf = RandomForestClassifier(n_estimators=100).fit(x, y)


if __name__ == '__main__':
    pass
    # func()
    s = time.time()
    res = __slope_rank(clf, x, score=True)
    print(f'time: {time.time() - s} sec')
    print(res)
    # 87.90728330612183
    # best result - 4 processes, 4 chunks + parallel_backend
