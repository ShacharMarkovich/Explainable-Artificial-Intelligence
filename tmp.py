# Run here whatever
from sklearn.ensemble import RandomForestClassifier

from lib import get_data
from lib.plotting import pdp


# TODO
#   check that target is being consider for multiclass    V
#   plot multiclass                                       V
#   plot normalized data                                  X
#   organize - make a 'core' file with declaration of:    V
#     Classifier (with a base classifier and normalizer)  X (possibly handle categorical data with encoder)
#     function for pdp + slope                            X
#   score by average of absolute values of all slopes     X
#   create intervals for slope to deal with "dips"        X


def main():
    X, Y = get_data('numeric_db/urbanLandCover.csv')
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, Y)
    print(clf.classes_)
    # ['V162', 'V75', 'V250', 'V16']['word_freq_make']

    r = pdp(clf, X, ['BrdIndx', 'Area', 'Round', 'Bright'], mode='show', )


if __name__ == '__main__':
    main()
