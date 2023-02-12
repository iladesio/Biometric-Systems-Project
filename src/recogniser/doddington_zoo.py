# plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist


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
        results = {user: dict({'fa': 0, 'fr': 0, 'ga': 0, 'gr': 0}) for user in set(self.y_label)}
        rows, cols = distance_matrix.shape

        #t = 5.02
        #t = 10
        t = 20

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
                break

            results[row_label] = {'fa': results[row_label]['fa'] + fa, 'fr': results[row_label]['fr'] + fr, 'ga': results[row_label]['ga'] + ga, 'gr': results[row_label]['gr'] + gr}

        return results, rows

    def plot_verification_results(self, results, rows):

        fig, (axDDGT) = plt.subplots(ncols=1)
        fig.set_size_inches(15, 5)
        fars = []
        frrs = []
        ags = []
        aip = []

        for t in results:
            fars += [results[t]['fa']/rows]
            frrs += [results[t]['fr']/rows]
            ags += [(results[t]['ga'] + results[t]['gr'])/rows]
            aip += [(results[t]['fa'] + results[t]['fr'])/rows]

        
        #axDDGT.scatter(fars, frrs, c=np.random.rand(len(fars),3))
        axDDGT.scatter(ags, aip)
        #axDDGT.legend(loc='center right', shadow=True, fontsize='x-large')
        for i, txt in enumerate(results):
            axDDGT.annotate(txt, (ags[i], aip[i]))

        axDDGT.set_xlabel('Average Genuine Score')
        axDDGT.set_ylabel('Average Imposter Score')

        plt.show()

    def eval_verification(self):

        # verification
        print("Verification Doddington Zoo:")
        verification_results, rows = self.verification()
        self.plot_verification_results(verification_results, rows)
    