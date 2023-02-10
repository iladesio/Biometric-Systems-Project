# plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from scipy.ndimage.filters import gaussian_filter1d


class Evaluation:

    def __init__(self, features, y_labels, current_metric="correlation"):

        self.features = features
        self.y_label = y_labels
        self.current_metric = current_metric

        # treshold lists
        self.distance_matrix = self.compute_distance_matrix().to_numpy()
        max_treshold = self.distance_matrix.max() * 0.66
        # self.decimal_tresholds = np.arange(0, 1.5, 0.01)
        self.decimal_tresholds = np.arange(0, max_treshold, max_treshold / 500)

    def compute_distance_matrix(self):
        return pd.DataFrame(squareform(pdist(np.array(self.features), metric=self.current_metric)))

    def verification(self, distance_matrix=None, thresholds=None):

        if distance_matrix is None:
            distance_matrix = self.distance_matrix

        if thresholds is None:
            thresholds = self.decimal_tresholds

        # dictionary that will contain all the results
        results = {t: dict() for t in thresholds}
        rows, cols = distance_matrix.shape

        for index, t in enumerate(thresholds):
            # For each treshold we test the method
            ga, fa, fr, gr = 0, 0, 0, 0

            for y in range(rows):
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

            results[t] = {'gar': ga / rows, 'far': fa / (rows * (len(grouped_results) - 1)), 'frr': fr / rows,
                          'grr': gr / (rows * (len(grouped_results) - 1))}

            if index % (len(thresholds) // 10) == 0:
                print(
                    f"gar: {results[t]['gar']}\t "
                    f"far: {results[t]['far']}\t "
                    f"frr: {results[t]['frr']}\t "
                    f"grr: {results[t]['grr']}")

        return results

    def plot_verification_results(self, results):

        fig, (axERR, axROC, axDET) = plt.subplots(ncols=3)
        fig.set_size_inches(10, 5)
        thresholds = []
        fars = []
        frrs = []

        for t in results:
            thresholds += [t]
            fars += [results[t]['far']]
            frrs += [results[t]['frr']]


        #fars = gaussian_filter1d(fars, sigma=2)
        #frrs = gaussian_filter1d(frrs, sigma=2)

        axERR.plot(thresholds, fars, 'r', label='FAR')
        axERR.plot(thresholds, frrs, 'g', label='FRR')
        axERR.legend(loc='center right', shadow=True, fontsize='x-large')
        axERR.set_xlabel('Threshold')
        axERR.title.set_text('FAR vs FRR')

        axROC.plot(fars, list(map(lambda frr: 1 - frr, frrs)))
        axROC.set_ylabel('GAR=1-FRR')
        axROC.set_xlabel('FAR')
        axROC.title.set_text('ROC Curve')

        axDET.plot(fars, frrs)
        axDET.set_ylabel('FRR')
        axDET.set_xlabel('FAR')
        axDET.set_yscale('log')
        axDET.set_xscale('log')
        axDET.title.set_text('DET Curve')

        plt.show()

        # plt.savefig("metrics_plot/" + self.current_metric + "_verification.png")
        # plt.clf()

    def similar_to(self, probe_idx, distance_matrix=None):

        if distance_matrix is None:
            distance_matrix = self.distance_matrix

        similarities = list(
            sorted([i for i in range(distance_matrix[probe_idx].shape[0])],
                   key=lambda i: distance_matrix[probe_idx][i]))

        similarities.remove(probe_idx)

        return similarities

    def identification(self, distance_matrix=None, thresholds=None):

        if distance_matrix is None:
            distance_matrix = self.distance_matrix

        if thresholds is None:
            thresholds = self.decimal_tresholds

        # dictionary that will contain all the results
        results = {t: dict() for t in thresholds}
        rows, cols = distance_matrix.shape
        di = {t: [0 for _ in range(rows)] for t in thresholds}
        _dir = {t: [0 for _ in range(rows)] for t in thresholds}

        for index, t in enumerate(thresholds):
            fa, gr = 0, 0

            for y in range(rows):
                row_label = self.y_label[y]
                similar_index = self.similar_to(y, distance_matrix)

                if distance_matrix[y][similar_index[0]] <= t:
                    if row_label == self.y_label[similar_index[0]]:
                        di[t][0] += 1

                        # find impostor case
                        for k in similar_index:
                            if row_label != self.y_label[k] and distance_matrix[y][k] <= t:
                                fa += 1
                                break
                    else:
                        for k, index_at_rank_k in enumerate(similar_index):
                            if row_label == self.y_label[index_at_rank_k] and distance_matrix[y][index_at_rank_k] <= t:
                                fa += 1
                                di[t][k] += 1
                                break
                else:
                    gr += 1

            _dir[t][0] = di[t][0] / rows

            for i in range(1, rows):
                _dir[t][i] = di[t][i] / rows + _dir[t][i - 1]

            results[t] = {'far': fa / rows, 'grr': gr / rows, 'dir': _dir[t], 'frr': 1 - _dir[t][0]}

            if index % (len(thresholds) // 10) == 0:
                print(
                    f"dir({t}, 1): {results[t]['dir'][0]}\t "
                    f"far: {results[t]['far']}\t "
                    f"frr: {results[t]['frr']}\t "
                    f"grr: {results[t]['grr']}")

        return results

    def cumulative_matching_score(self, distance_matrix=None):

        if distance_matrix is None:
            distance_matrix = self.distance_matrix

        rows, cols = distance_matrix.shape
        cms = [0 for _ in range(rows)]

        for y in range(rows):
            row_label = self.y_label[y]
            similar_index = self.similar_to(y)

            for k, indexAtRankK in enumerate(similar_index):
                if row_label == self.y_label[indexAtRankK]:
                    cms[k] += 1
                    break
        # rr
        cms[0] = cms[0] / rows
        for k in range(1, rows):
            cms[k] = cms[k] / rows + cms[k - 1]
        return cms

    def plot_identification_results(self, results, cms):

        fig, (axERR, axROC, axDET, axDIR, axCMS) = plt.subplots(ncols=5)
        fig.set_size_inches(20, 5)
        thresholds = []
        fars = []
        frrs = []
        dir1 = []

        for t in results:
            thresholds += [t]
            fars += [results[t]['far']]
            frrs += [results[t]['frr']]
            dir1 += [results[t]['dir'][0]]

        #fars = gaussian_filter1d(fars, sigma=2)
        #frrs = gaussian_filter1d(frrs, sigma=2)
        #dir1 = gaussian_filter1d(dir1, sigma=2)
        #cms = gaussian_filter1d(cms, sigma=2)


        # FAR vs FRR
        axERR.plot(thresholds, fars, 'r', label='FAR')
        axERR.plot(thresholds, frrs, 'g', label='FRR')
        axERR.set_xlabel('Threshold')
        axERR.legend(loc='lower right', shadow=True, fontsize='x-large')
        axERR.title.set_text('FAR and FRR')
        axDET.plot(fars, frrs)
        axDET.set_ylabel('FRR')
        axDET.set_xlabel('FAR')
        axDET.set_yscale('log')
        axDET.set_xscale('log')
        axDET.title.set_text('DET Curve')

        axROC.plot(fars, list(map(lambda frr: 1 - frr, frrs)))
        axROC.set_ylabel('GAR=1-FRR')
        axROC.set_xlabel('FAR')
        axROC.title.set_text('ROC Curve')

        # DIR(t, 1)
        axDIR.plot(thresholds, dir1)
        axDIR.set_xlabel('Threshold')
        axDIR.set_ylabel('DIR at rank 1')
        axDIR.title.set_text('DIR(t, 1)')

        # CMS
        axCMS.plot(range(1, len(cms) + 1), cms)
        axCMS.set_xlabel('Rank')
        axCMS.set_ylabel('Probability of identification')
        axCMS.title.set_text('CMC')

        plt.show()

        # plt.savefig("metrics_plot/" + self.current_metric + "_identification.png")
        # plt.clf()

    def eval_verification(self):

        # verification
        print("Verification:")
        verification_results = self.verification()
        self.plot_verification_results(verification_results)

    def eval_identification(self):

        # identification
        print("Identification:")
        identification_results = self.identification()
        cms = self.cumulative_matching_score()
        self.plot_identification_results(identification_results, cms)
