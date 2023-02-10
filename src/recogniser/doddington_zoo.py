# plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from scipy.ndimage.filters import gaussian_filter1d


class Doddigton:

    def __init__(self, features, y_labels):

        self.features = features
        self.y_label = y_labels
        


    def __init__(self, features, y_labels):

        self.features = features
        self.y_label = y_labels

        # treshold lists
        self.decimal_tresholds = np.arange(0, 50, 0.05)
        self.distance_matrix = self.compute_distance_matrix().to_numpy()

    def compute_distance_matrix(self, metric='seuclidean'):
        return pd.DataFrame(squareform(pdist(np.array(self.features), metric=metric)))

    def verification(self, distance_matrix=None, thresholds=None):

        if distance_matrix is None:
            distance_matrix = self.distance_matrix

        if thresholds is None:
            thresholds = self.decimal_tresholds

        # dictionary that will contain all the results
        results = {user: dict() for user in set(self.y_label)}
        rows, cols = distance_matrix.shape

        t = 5.7

        for y in range(rows):
            fa, fr, gr, ga = 0, 0, 0, 0
            row_label = self.y_label[y]

            # results are grouped by label (here we're doing verification with multiple templates)
            grouped_results = dict()

            for x in range(cols):
                # same probe => skip
                if x == y:
                    continue
                # column label
                col_label = self.y_label[x]
                # storing results
                grouped_results.setdefault(col_label, []).append(distance_matrix[y, x])

            for label in grouped_results:
                # take minimum distance between templates
                d = min(grouped_results[label])
                if d <= t:
                    # we have an acceptance, genuine or false?
                    if row_label == label:
                        ga += 1
                    else:
                        fa += 1
                else:
                    # we have a rejection, genuine or false?
                    if row_label == label:
                        fr += 1
                    else:
                        gr += 1

            results[row_label] = {'fa': results[row_label]['fa'] + fa, 'fr': results[row_label]['fr'] + fr}

        #if index % (len(thresholds) // 10) == 0:
        #    print(
        #        f"gar: {results[t]['gar']}\t "
        #        f"far: {results[t]['far']}\t "
        #        f"frr: {results[t]['frr']}\t "
        #        f"grr: {results[t]['grr']}")

        return results

    