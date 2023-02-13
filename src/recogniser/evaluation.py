import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from shapely.geometry import LineString

from src.utilities import config

TITLE_SIZE = 20
LABEL_SIZE = 16


class Evaluation:

    def __init__(self, x_features, y_labels, current_metric="euclidean"):

        self.x_features = x_features
        self.y_label = y_labels
        self.current_metric = current_metric

        # treshold lists
        self.distance_matrix = self.compute_distance_matrix().to_numpy()
        max_treshold = self.distance_matrix.max()
        self.decimal_tresholds = np.arange(0, max_treshold, max_treshold / 1000)

    def compute_distance_matrix(self):
        return pd.DataFrame(squareform(pdist(np.array(self.x_features), metric=self.current_metric)))

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

    @staticmethod
    def plot_verification_results(results):

        fig, axERR = plt.subplots(nrows=1, ncols=1, figsize=(8.5, 6.5))

        thresholds = []
        fars = []
        frrs = []

        for t in results:
            thresholds += [t]
            fars += [results[t]['far']]
            frrs += [results[t]['frr']]

        # ERR
        axERR.plot(thresholds, fars, 'r', label='FAR')
        axERR.plot(thresholds, frrs, 'g', label='FRR')
        axERR.legend(loc='center right', shadow=True, fontsize='x-large')
        axERR.set_xlabel('Threshold', fontsize=LABEL_SIZE)
        axERR.set_ylabel('Error Rate', fontsize=LABEL_SIZE)

        line_1 = LineString(np.column_stack((thresholds, fars)))
        line_2 = LineString(np.column_stack((thresholds, frrs)))
        intersection = line_1.intersection(line_2)
        x, y = intersection.xy
        x = float("{:.2f}".format(x[0]))
        y = float("{:.2f}".format(y[0]))

        axERR.plot(*intersection.xy, 'ro')
        axERR.annotate("(" + str(x) + "," + str(y) + ")", xy=(x, y), xytext=(x + 0.3, y + 0.01))

        axERR.set_title('FAR and FRR', size=TITLE_SIZE)
        axERR.figure.savefig(config.BASE_PLOT_PATH + "verification_err.png", dpi=400)
        plt.clf()

        # ROC
        fig, axROC = plt.subplots(nrows=1, ncols=1, figsize=(8.5, 6.5))

        axROC.plot(fars, list(map(lambda frr: 1 - frr, frrs)))
        axROC.set_ylabel('GAR=1-FRR', fontsize=LABEL_SIZE)
        axROC.set_xlabel('FAR', fontsize=LABEL_SIZE)
        axROC.set_title('ROC Curve', size=TITLE_SIZE)
        axROC.figure.savefig(config.BASE_PLOT_PATH + "verification_roc.png", dpi=400)
        plt.clf()

        # DET
        fig, axDET = plt.subplots(nrows=1, ncols=1, figsize=(8.5, 6.5))

        axDET.plot(fars, frrs)
        axDET.set_ylabel('FRR', fontsize=LABEL_SIZE)
        axDET.set_xlabel('FAR', fontsize=LABEL_SIZE)
        axDET.set_yscale('log')
        axDET.set_xscale('log')
        axDET.set_title('DET Curve', size=TITLE_SIZE)
        axDET.figure.savefig(config.BASE_PLOT_PATH + "verification_det.png", dpi=400)
        plt.clf()

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

    @staticmethod
    def plot_identification_results(results, cms):

        thresholds = []
        fars = []
        frrs = []
        dir1 = []

        for t in results:
            thresholds += [t]
            fars += [results[t]['far']]
            frrs += [results[t]['frr']]
            dir1 += [results[t]['dir'][0]]

        # FAR and FRR
        fig, axERR = plt.subplots(nrows=1, ncols=1, figsize=(8.5, 6.5))

        axERR.plot(thresholds, fars, 'r', label='FAR')
        axERR.plot(thresholds, frrs, 'g', label='FRR')
        axERR.set_ylabel('Error Rate', fontsize=LABEL_SIZE)
        axERR.set_xlabel('Threshold', fontsize=LABEL_SIZE)
        axERR.legend(loc='lower right', shadow=True, fontsize='x-large')
        axERR.set_title('FAR and FRR', size=TITLE_SIZE)

        line_1 = LineString(np.column_stack((thresholds, fars)))
        line_2 = LineString(np.column_stack((thresholds, frrs)))
        intersection = line_1.intersection(line_2)
        x, y = intersection.xy
        x = float("{:.2f}".format(x[0]))
        y = float("{:.2f}".format(y[0]))

        axERR.plot(*intersection.xy, 'ro')
        axERR.annotate("(" + str(x) + "," + str(y) + ")", xy=(x, y), xytext=(x + 0.3, y + 0.01))
        axERR.figure.savefig(config.BASE_PLOT_PATH + "identification_err.png", dpi=400)
        plt.clf()

        # ROC
        fig, axROC = plt.subplots(nrows=1, ncols=1, figsize=(8.5, 6.5))
        axROC.plot(fars, list(map(lambda frr: 1 - frr, frrs)))
        axROC.set_ylabel('GAR=1-FRR', fontsize=LABEL_SIZE)
        axROC.set_xlabel('FAR', fontsize=LABEL_SIZE)
        axROC.set_title('ROC Curve', size=TITLE_SIZE)
        axROC.set_ylim(0, 1)
        axROC.figure.savefig(config.BASE_PLOT_PATH + "identification_roc.png", dpi=400)
        plt.clf()

        # DIR(t, 1)
        fig, axDIR = plt.subplots(nrows=1, ncols=1, figsize=(8.5, 6.5))
        axDIR.plot(thresholds, dir1)
        axDIR.set_xlabel('Threshold', fontsize=LABEL_SIZE)
        axDIR.set_ylabel('DIR at rank 1', fontsize=LABEL_SIZE)
        axDIR.set_title('DIR(t, 1)', size=TITLE_SIZE)
        axDIR.figure.savefig(config.BASE_PLOT_PATH + "identification_dir.png", dpi=400)
        plt.clf()

        # CMS
        fig, axCMS = plt.subplots(nrows=1, ncols=1, figsize=(8.5, 6.5))
        axCMS.plot(range(1, len(cms) // 6), cms[1:len(cms) // 6])
        axCMS.set_xlabel('Rank', fontsize=LABEL_SIZE)
        axCMS.set_ylabel('Probability of identification', fontsize=LABEL_SIZE)
        axCMS.set_title('CMC', size=TITLE_SIZE)
        axCMS.figure.savefig(config.BASE_PLOT_PATH + "identification_cms.png", dpi=400)
        plt.clf()

        # FAR and DIR
        fig, axTRS = plt.subplots(nrows=1, ncols=1, figsize=(8.5, 6.5))

        axTRS.plot(thresholds, fars, 'r', label='FAR')
        axTRS.plot(thresholds, dir1, label='DIR')
        axTRS.set_xlabel('Threshold', fontsize=LABEL_SIZE)
        axTRS.set_title('FAR and DIR', size=TITLE_SIZE)
        axTRS.axvline(x=1.6, color="orange", linestyle='--')
        axTRS.legend(loc='lower right', shadow=True, fontsize='x-large')

        trsh = np.linspace(0, max(thresholds), 1001)
        line_1 = LineString(np.column_stack((trsh, fars)))
        line_2 = LineString(np.column_stack((trsh, dir1)))
        line_t = LineString([(1.6, 0), (1.6, 1.0)])

        intersection_1 = line_1.intersection(line_t)
        intersection_2 = line_2.intersection(line_t)

        x1, y1 = intersection_1.xy
        x1 = float("{:.2f}".format(x1[0]))
        y1 = float("{:.2f}".format(y1[0]))

        axTRS.plot(*intersection_1.xy, 'g', marker="x")

        axTRS.annotate("(" + str(x1) + "," + str(y1) + ")", xy=(x1, y1), xytext=(x1 + 0.3, y1 - 0.01))

        x2, y2 = intersection_2.xy
        x2 = float("{:.2f}".format(x2[0]))
        y2 = float("{:.2f}".format(y2[0]))

        axTRS.plot(*intersection_2.xy, 'g', marker="x")
        axTRS.annotate("(" + str(x2) + "," + str(y2) + ")", xy=(x2, y2), xytext=(x2 + 0.3, y2 - 0.01))
        axTRS.figure.savefig(config.BASE_PLOT_PATH + "identification_dir_far.png", dpi=400)

        plt.clf()

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
