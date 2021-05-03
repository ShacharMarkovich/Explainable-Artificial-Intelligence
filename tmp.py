# Run here whatever
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.inspection import plot_partial_dependence, partial_dependence
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from lib import pdp, get_data
from lib.plotting import pdp2

X, Y = get_data('numeric_db/spam.csv')
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, Y)
# ['V162', 'V178']
r = pdp2(clf, X, ['char_freq_$', 'word_freq_make', 'word_freq_address', 'word_freq_all'])

plt.show()

# print(r)
# scaled_x = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=list(X))
# scaled_clf = RandomForestClassifier(n_estimators=100).fit(scaled_x, Y)
#
# pdp(clf, X, ['V162', 'V178'], mode='show')
# Multiclass -- average of absolute values of different slopes
# check that target is being consider for multiclass
# plot multiclass
# create intervals for slope to deal with "dips"
