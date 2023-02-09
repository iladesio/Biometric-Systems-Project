import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from src.recogniser.wbb_recogniser import WBBRecogniser
from src.utilities import config


class IdentityVerifier:
    def __init__(self, wbb_recognizer: WBBRecogniser, acceptance_threshold: float = 1.7):
        self.at = acceptance_threshold
        self.wbb_recognizer = wbb_recognizer

        self.features_name = [] # todo calolcare da wbb rec

    def verify(self, pathname, claimed_id="AB_M_65_DX_R") -> bool:

        # Read sample data from file
        sample = np.array(np.loadtxt(pathname, dtype=float))

        # Features extraction from time series
        ts = self.wbb_recognizer.normalize_sample_to_timeseries(sample, id="demo_sample")
        probe_features = self.wbb_recognizer.extract_features_from_timeseries(ts)

        # filter the calculated filter
        df = pd.DataFrame(probe_features["template"])
        df.columns = probe_features["features_name"]

        filtered_features_sample = df[self.wbb_recognizer.features_name].to_numpy()

        # Compute distances between probe's template and claim's templates
        distance_matrix = pd.DataFrame()
        gallery_template = self.wbb_recognizer.extracted_features[
            self.wbb_recognizer.extracted_features.index == claimed_id]  # 'AB_M_65_DX_R'

        # no user found with that claimed id
        if len(df) == 0:
            return False

        # Compute distance with every template associated to that claimed id
        min_distance = sys.float_info.max
        for row in gallery_template.to_numpy():
            row_features = row.reshape(1, -1)
            stacked = np.vstack((row_features, filtered_features_sample.reshape(1, -1)))
            min_distance = min(min_distance, pdist(stacked, metric='euclidean')[0])
        return min_distance < self.at


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Verification demo')
    # parser.add_argument('--in', metavar='img_path', type=str, nargs=1, required=True,
    #                     help='Input image to verify')
    # parser.add_argument('--id', metavar='claimed_identity', type=int, nargs=1, required=True,
    #                     help='Claimed identity')
    # parser.add_argument('--dataset', metavar='dataset_filepath', type=str, nargs=1, required=True,
    #                     help='Dataset csv file')

    # args = vars(parser.parse_args())
    # inputImagePath = args['in'][0]
    # claimedIdentity = args['id'][0]
    # datasetPath = args['dataset'][0]

    # featNet = FeatNet(pretrainedName="featNetTriplet_100e_1e-4lr.pth").eval()
    # vggfe = VGGFE(pretrainedName='vggfe_lr0001_100e.pth')

    # featureExtractor = FeatureExtractor(featNet)
    # dataset = Dataset(datasetPath, featureExtractor)

    wbb_recognizer = WBBRecogniser()

    identity_verifier = IdentityVerifier(wbb_recognizer)
    claimed_identity = "AB_M_65_DX_R"

    if identity_verifier.verify(config.TEST_DEMO_PATH, claimed_identity):
        print(f"Identity verified, you are user#{claimed_identity}")
    else:
        print("Access denied.")
