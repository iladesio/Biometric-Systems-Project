# plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist

from src.utilities import config

LABEL_SIZE = 16
TITLE_SIZE = 20


class Doddigton:

    def __init__(self, features, y_labels):

        self.features = features
        self.y_label = y_labels

        # treshold lists
        self.distance_matrix = self.compute_distance_matrix().to_numpy()
        # threshold chosen after evaluation
        self.selected_threshold = 1.6

    def compute_distance_matrix(self, metric='euclidean'):
        return pd.DataFrame(squareform(pdist(np.array(self.features), metric=metric)))

    def verification(self, distance_matrix=None):

        if distance_matrix is None:
            distance_matrix = self.distance_matrix

        # dictionary that will contain all the results
        results = {user: dict({'fa': 0, 'fr': 0, 'ga': 0, 'gr': 0, 'ctr': 0, 'ctrImpostor': 0}) for user in
                   set(self.y_label)}
        rows, cols = distance_matrix.shape

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
                if d <= self.selected_threshold:
                    # we have an acceptance, genuine or false?
                    if row_label == label:
                        results[row_label]['ga'] += 1
                    else:
                        results[row_label]['fa'] += 1
                else:
                    # we have a rejection, genuine or false?
                    if row_label == label:
                        results[row_label]['fr'] += 1
                    else:
                        results[row_label]['gr'] += 1

            results[row_label]['ctrImpostor'] = len(grouped_results) - 1
            results[row_label]['ctr'] += 1

        return results

    @staticmethod
    def plot_verification_results(results):

        fig, (axDDGT) = plt.subplots(ncols=1)
        fig.set_size_inches(15, 5)
        grr = []
        frr = []
        gar = []
        far = []

        for t in results:
            grr += [results[t]['gr'] / (results[t]['ctr'] * results[t]['ctrImpostor'])]
            frr += [results[t]['fr'] / results[t]['ctr']]
            gar += [results[t]['ga'] / results[t]['ctr']]
            far += [results[t]['fa'] / (results[t]['ctr'] * results[t]['ctrImpostor'])]

        axDDGT.scatter(far, frr, c=np.random.rand(len(frr), 3))

        for i, txt in enumerate(results):
            axDDGT.annotate(txt.split('_')[0], (far[i] + 0.003, frr[i] + 0.003))

        axDDGT.set_xlabel('FAR', fontsize=LABEL_SIZE)
        axDDGT.set_ylabel('FRR', fontsize=LABEL_SIZE)
        axDDGT.set_title('Doddington Zoo', size=TITLE_SIZE)

        plt.savefig(config.BASE_PLOT_PATH + "doddington_zoo_zoom.png", dpi=400)
        plt.clf()

        fig, (axDDGT) = plt.subplots(ncols=1)
        fig.set_size_inches(15, 5)

        axDDGT.scatter(far, frr, c=np.random.rand(len(frr), 3))

        for i, txt in enumerate(results):
            axDDGT.annotate(txt.split('_')[0], (far[i] + 0.003, frr[i] + 0.003))

        axDDGT.set_xlabel('FAR', fontsize=LABEL_SIZE)
        axDDGT.set_ylabel('FRR', fontsize=LABEL_SIZE)
        axDDGT.set_title('Doddington Zoo', size=TITLE_SIZE)

        axDDGT.set_ylim(0, 1)
        axDDGT.set_xlim(0, 1)
        axDDGT.axvspan(0, 0.33, 0.66, 1, color='orange', alpha=0.1)
        axDDGT.axvspan(0.33, 0.66, 0.66, 1, color='red', alpha=0.1)
        axDDGT.axvspan(0.66, 1, 0.66, 1, color='purple', alpha=0.1)
        axDDGT.axvspan(0, 0.66, 0, 0.66, color='green', alpha=0.1)
        axDDGT.axvspan(0, 0.3, 0, 0.3, color='lime', alpha=0.1)
        axDDGT.axvspan(0.66, 1, 0.33, 0.66, color='blue', alpha=0.1)
        axDDGT.axvspan(0.66, 1, 0, 0.33, color='cyan', alpha=0.1)

        axDDGT.annotate('Phantoms', (0.01, 0.95), weight="bold")
        axDDGT.annotate('Goats', (0.34, 0.95), weight="bold")
        axDDGT.annotate('Worms', (0.67, 0.95), weight="bold")
        axDDGT.annotate('Sheeps', (0.35, 0.35), weight="bold")
        axDDGT.annotate('Lambs/Wolves', (0.67, 0.61), weight="bold")
        axDDGT.annotate('Chameleons', (0.67, 0.28), weight="bold")
        axDDGT.annotate('Doves', (0.25, 0.25), weight="bold")

        plt.savefig(config.BASE_PLOT_PATH + "doddington_zoo.png", dpi=400)
        plt.clf()

    def eval_verification(self):

        # verification
        print("Verification Doddington Zoo")
        verification_results = self.verification()
        self.plot_verification_results(verification_results)
