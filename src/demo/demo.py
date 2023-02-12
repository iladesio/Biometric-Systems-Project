import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from src.recogniser.wbb_recogniser import WBBRecogniser
from src.utilities import config


class IdentityVerifier:
    def __init__(self, wbb_recognizer: WBBRecogniser, acceptance_threshold: float = 4.94):
        self.at = acceptance_threshold
        self.wbb_recognizer = wbb_recognizer

    def verify(self, pathname, claimed_id) -> bool:

        # Read sample data from file
        sample = np.array(np.loadtxt(pathname, dtype=float))

        # features extraction from time series
        ts = self.wbb_recognizer.normalize_sample_to_timeseries(sample, id="demo_sample", counter=1)
        probe_features = self.wbb_recognizer.extract_features_from_timeseries(ts)

        # scale features with fitted scaler
        scaled_features = self.wbb_recognizer.scaler.transform(probe_features["template"])

        # filter the calculated filter
        df = pd.DataFrame(scaled_features)
        df.columns = probe_features["features_name"]

        filtered_features_sample = df[self.wbb_recognizer.features_name].to_numpy()

        """ Compute distances between probe's template and claim's templates """

        gallery_template = self.wbb_recognizer.extracted_features[
            self.wbb_recognizer.extracted_features.index == claimed_id]

        # filter gallery_template with the current features list
        gallery_template = gallery_template[self.wbb_recognizer.features_name]

        # no user found with that claimed id
        if len(df) == 0:
            return False

        # Compute distance with every template associated to that claimed id
        min_distance = sys.float_info.max
        for row in gallery_template.to_numpy():
            row_features = row.reshape(1, -1)
            stacked = np.vstack((row_features, filtered_features_sample.reshape(1, -1)))
            min_distance = min(min_distance, pdist(stacked, metric='euclidean')[0])

        return min_distance <= self.at


class SubjectIdentifier:
    def __init__(self, wbb_recognizer: WBBRecogniser, acceptance_threshold: float = 4.94):
        self.at = acceptance_threshold
        self.wbb_recognizer = wbb_recognizer

    def identify(self, pathname, maxRank: int = 1):
        # Read sample data from file
        sample = np.array(np.loadtxt(pathname, dtype=float))

        # features extraction from time series
        ts = self.wbb_recognizer.normalize_sample_to_timeseries(sample, id="demo_sample", counter=1)
        probe_features = self.wbb_recognizer.extract_features_from_timeseries(ts)

        # scale features with fitted scaler
        scaled_features = self.wbb_recognizer.scaler.transform(probe_features["template"])

        # filter the calculated filter
        df = pd.DataFrame(scaled_features)
        df.columns = probe_features["features_name"]

        filtered_features_sample = df[self.wbb_recognizer.features_name].to_numpy()

        """ Compute distances between probe's template and all the other templates """

        gallery_template = self.wbb_recognizer.extracted_features

        # filter gallery_template with the current features list
        gallery_template = gallery_template[self.wbb_recognizer.features_name]

        # no user found with that claimed id
        if len(df) == 0:
            return []

        # Compute distance with every template associated to that claimed id
        distances = []
        for idx, row in enumerate(gallery_template.to_numpy()):
            row_features = row.reshape(1, -1)
            stacked = np.vstack((row_features, filtered_features_sample.reshape(1, -1)))
            curr_label = self.wbb_recognizer.extracted_features.index[idx]
            distances += [(pdist(stacked, metric='euclidean')[0], curr_label)]

        distances.sort(key=lambda el: el[0])
        if type(self.at) == type(float()):
            distances = list(filter(lambda x: x[0] < self.at, distances))
        return distances[:maxRank]


if __name__ == '__main__':

    wbb_recognizer = WBBRecogniser()

    x = "i"

    if x == "v":
        identity_verifier = IdentityVerifier(wbb_recognizer)
        claimed_identity = "S6_F_23_DX_S"

        if identity_verifier.verify(config.TEST_DEMO_PATH, claimed_identity):
            print(f"Identity verified, you are user {claimed_identity}")
        else:
            print("Access denied.")

    else:
        subjectIdentifier = SubjectIdentifier(wbb_recognizer)

        users = subjectIdentifier.identify(config.TEST_DEMO_PATH)
        if len(users) == 0:
            print(f"No match in the dataset")
        else:
            print(f"You are user {users[0][1]}")
