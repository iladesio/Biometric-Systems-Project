# plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist

from src.utilities import config



class Doddigton:

    def __init__(self, features, y_labels):

        self.features = features
        self.y_label = y_labels
        


    def __init__(self, features, y_labels):

        self.features = features
        self.y_label = y_labels

        # treshold lists
        self.distance_matrix = self.compute_distance_matrix().to_numpy()

    def compute_distance_matrix(self, metric='euclidean'):
        return pd.DataFrame(squareform(pdist(np.array(self.features), metric=metric)))

    def verification(self, distance_matrix=None):

        if distance_matrix is None:
            distance_matrix = self.distance_matrix
            

        # dictionary that will contain all the results
        results = {user: dict({'fa': 0, 'fr': 0, 'ga': 0, 'gr': 0, 'ctr': 0}) for user in set(self.y_label)}
        rows, cols = distance_matrix.shape

        t = 4.66

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
                        results[row_label]['ga'] += 1 
                    else:
                        results[row_label]['fa'] += 1 
                else:
                    # we have a rejection, genuine or false?
                    if row_label == label:
                        results[row_label]['fr'] += 1 
                    else:
                        results[row_label]['gr'] += 1 

            results[row_label]['ctr'] += 1 
                

        return results

    def plot_verification_results(self, results):

        fig, (axDDGT) = plt.subplots(ncols=1)
        fig.set_size_inches(15, 5)
        grr = []
        frr = []
        gar = []
        far = []

        for t in results:
            grr += [results[t]['gr']/(results[t]['ctr']*23)]
            frr += [results[t]['fr']/results[t]['ctr']]
            gar += [results[t]['ga']/results[t]['ctr']]
            far += [results[t]['fa']/(results[t]['ctr']*23)]

        
        axDDGT.scatter(far, frr, c=np.random.rand(len(frr),3))
        #axDDGT.legend(loc='center right', shadow=True, fontsize='x-large')
        for i, txt in enumerate(results):
            axDDGT.annotate(txt, (far[i], frr[i]))

        axDDGT.set_xlabel('FAR')
        axDDGT.set_ylabel('FRR')

        plt.savefig(config.BASE_PLOT_PATH+"doddington_zoo.png", dpi=400)
        plt.clf()

    def eval_verification(self):

        # verification
        print("Verification Doddington Zoo:")
        verification_results = self.verification()
        self.plot_verification_results(verification_results)
    