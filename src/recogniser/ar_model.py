import os

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from tqdm.auto import tqdm
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

from src.utilities import config


class AR_Model:

    def __init__(self):

        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []

    def get_datas(self):

        # get samples directory list name
        list_dir = os.listdir(config.SAMPLES_DIR_PATH)
        print(list_dir)

        templates = []

        for directory in list_dir:
            # get samples file list name
            file_list = os.listdir(config.SAMPLES_DIR_PATH + "/" + directory + "")

            for filename in file_list:
                ts = {"id": [], "time": [], "m_x": [], "m_y": []}
                sample = np.array(
                    np.loadtxt(config.SAMPLES_DIR_PATH + "/" + directory + "/" + filename + "", dtype=float))

                for temp in sample:
                    ts["m_x"].append(temp[5])
                    ts["m_y"].append(temp[6])
                    ts["time"].append(temp[0])
                    ts["id"].append(directory)

                if len(sample) > 0:
                    templates.append(pd.DataFrame(ts))

        data = {"label": [], "template": []}

        for id, elem in enumerate(templates):
            print("fe cycle id: " + str(id))

            extracted_features = extract_features(
                elem,
                column_id="id",
                column_sort="time",
                n_jobs=1,
                show_warnings=False,
                disable_progressbar=False,
                profile=False,
                impute_function=impute
            )

            data["label"].append(elem["id"][0])
            ext_feat_list = []

            for e in extracted_features.to_numpy()[0]:
                ext_feat_list.append(e)
            data["template"].append(ext_feat_list)

    # find the best lag for one time series
    def _get_best_lag(self, x, maxlag):
        return np.max(ar_select_order(x, maxlag, ic="aic").ar_lags)

    # find the average best lag
    # for the time series
    def find_best_lag(self, x, maxlag=100):
        lags = []
        for i in tqdm(range(len(x))):
            lags.append(self._get_best_lag(x[i], maxlag))
        return int(round(np.mean(lags)))

    # extract coefficients
    def get_ar_coefficients(self, x, lags):
        features = []
        for i in tqdm(range(len(x))):
            ar_model = AutoReg(x[i], lags).fit()
            features.append(ar_model.params)
        return pd.DataFrame(features)

    def get_features(self):
        # sample every 10th to reduce computation time
        best_lag = self.find_best_lag(self.x_train[::10, :])
        print("Chosen lag", best_lag)
        # => "Chosen lag 45"
        train_features = self.get_ar_coefficients(self.x_train, best_lag)
        test_features = self.get_ar_coefficients(self.x_test, best_lag)

        return train_features, test_features
