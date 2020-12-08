from multiprocessing import Pool as ProcessPool
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

K = 6

problems = {}


# def explain(files_list: list):
#     for file in files_list:
#         data1 = pd.read_csv(f"../numeric_db/numeric/{file}")  # numeric value in each cell
#         try:
#             x = data1.drop(labels=['class'], axis=1)
#             y = data1['class']
#         except KeyError:
#             x = data1.drop(labels=['Class'], axis=1)
#             y = data1['Class']
#
#         clf = RandomForestClassifier(n_estimators=100)
#         clf.fit(x, y)
#         try:
#             best_slope = slope_rank(clf, x, K, True)
#             print(f"finish slope_rank file {file}")
#             best_forest = select_k_best(x, y, classifier=clf, k=K)
#             print(f"finish select_k_best file {file}")
#
#             pdp(clf, x, best_slope, fig_name=f"{file}: best slopes")
#             print(f"finish best_slope {file}")
#             pdp(clf, x, best_forest, fig_name=f"{file}: best random forest classifier")
#             print(f"finish with {file}")
#         except Exception as error:
#             problems[file] = [type(error), error]

#
# def run_all_files():
#     # get all files' name in db:
#     print("execute numeric dbs, it's gonna take a while... ")
#     files_name = os.listdir("../numeric_db/numeric")
#     n_jobs = 6
#     work_items = np.array_split(files_name, n_jobs) # split files to n parts
#     with ProcessPool(n_jobs) as pool:
#         work_results = pool.map(explain, work_items)
#     # t1 = threading.Thread(target=explain, args=(f1,))
#     # t2 = threading.Thread(target=explain, args=(f2,))
#     # t3 = threading.Thread(target=explain, args=(f3,))
#     # t4 = threading.Thread(target=explain, args=(f4,))
#     #
#     # t1.start()
#     # t2.start()
#     # t3.start()
#     # t4.start()
#     #
#     # t1.join()
#     # t2.join()
#     # t3.join()
#     # t4.join()
#     plt.show()
#     print("\n\nthe maniac are:\n", problems)
#     with open("problems.txt", "w") as problem_files:
#         for file_name, val in problems.items():
#             problem_files.write(f"In `{file_name}`:\n")
#             problem_files.write(f"{str(val[0])}\n")
#             problem_files.write(f"{str(val[1])}\n\n\n")


def todo_split_it_to_what_you_need_to_split_to():
    # region Data Initializing
    start = time.time()
    X, y = get_data('numeric_db/spam.csv')
    # endregion

    # TODO: split this gigantic function to what you need to split to..

    # Classifier Initialization:
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)

    # Measurements for Selected Classifier:
    # print(calc_measures(clf, x1, y1))
    # print(slope_rank(clf, X, k=6, score=True))
    v = partial_dependence(clf, X, ['char_freq_!', 'word_freq_000', 'char_freq_$',
                  'word_freq_remove'])
    print(v)
    # plot_partial_dependence(clf, x1, ['word_freq_edu', 'word_freq_meeting', 'char_freq_!', 'word_freq_000',
    #               'char_freq_$', 'word_freq_remove'])
    # print(line)
    # Plot Feature Importance for Categorical or Continuous Data
    # plot_feature_importance(X, y, classifier=clf)
    # plot_feature_importance(x2, y2, mutual_info_classif)

    # Professor, we save you the time to run the code yourself and tell you that those are the most influential words:)
    best_slope = ['word_freq_edu', 'word_freq_meeting', 'char_freq_!', 'word_freq_000', 'char_freq_$',
                  'word_freq_remove']
    # best_forest = select_k_best(x1, y1, classifier=clf, k=6)
    # pdp(clf, x1, best_slope, fig_name="best features by slopes")
    # pdp(clf, x1, best_forest, fig_name="best features by Random Forest Classifier")

    # Get K Best Features Names for Categorical or Continuous Data
    # plot_features = select_k_best(x1, y1, classifier=clf, k=6)
    # print(plot_features)
    # plot_features = select_k_best(x2, y2, mutual_info_classif, 4)

    # Get K Best Features Names & Scores for Categorical or Continuous Data
    # names, scores = select_k_best(x1, y1, mutual_info_classif, 6, True)
    # names, scores = select_k_best(x2, y2, mutual_info_classif, 6, True)

    # PDP for Continuous Data only
    # plot_partial_dependence(clf, x1, [0, 1, 2, 3])
    # pdp(clf, x1, plot_features)
    # print(x[0])
    # print(y[0])
    # print(partial_dependence(clf, x1, ['word_freq_address']))
    # Plot Bar Chart for Categorical Data only
    # plot_bar_chart(x2, y2, plot_features)
    # slopes, features = slope_rank(clf, x1)
    # pd.DataFrame(slopes, features).plot.barh()
    print("\nTime Elapsed: %.4f seconds." % (time.time() - start))
    # plt.show()


if __name__ == "__main__":
    todo_split_it_to_what_you_need_to_split_to()
    # run_all_files()
