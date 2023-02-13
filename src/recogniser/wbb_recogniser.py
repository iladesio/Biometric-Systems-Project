import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.utilities.dataframe_functions import impute

from src.recogniser.doddington_zoo import Doddigton
from src.recogniser.evaluation import Evaluation
from src.utilities import config


class WBBRecogniser:
    def __init__(self):

        # templates data
        self.data = {}
        self.features_name = []

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.gallery = None

        self.scaler = None

        """ init """
        if config.EXTRACT_FEATURE_FROM_SAMPLES:
            self.__extract_feature_from_samples()

        self.__split_train_test()

    def __read_datas(self):
        with open(config.TEMPLATES_PATH) as json_file:
            self.data = json.load(json_file)

    # Split dataset in train and test set
    def __split_train_test(self):

        print("Loading templates data")
        self.__read_datas()

        print("Scaling features")
        scaler = MinMaxScaler()
        scaler.fit(self.data["template"])
        scaled_features = scaler.transform(self.data["template"])
        self.scaler = scaler

        print("Splitting dataset")
        x_train, x_test, y_train, y_test = train_test_split(scaled_features, self.data["label"], test_size=0.5,
                                                            random_state=42, stratify=self.data["label"])

        features_name = self.__select_feature_extracted_train(
            x_feature=x_train,
            y_label=y_train,
            features_name=self.data["features_name"],
        )

        # update features list starting from train dataset
        self.features_name = features_name

        print("Splitting Dataset")

        df_test = pd.DataFrame(index=y_test, data=x_test)
        df_test.columns = self.data["features_name"]

        # filter test features with the trained ones
        df_test = df_test[self.features_name]

        self.x_test = df_test.to_numpy()
        self.y_test = y_test

        self.gallery = df_test

    def __extract_feature_from_samples(self):

        print("Samples processing in progress...")

        # get samples directory list name
        list_dir = os.listdir(config.SAMPLES_DIR_PATH)

        ts = {"id": [], "time": [], "m_x": [], "m_y": []}

        for directory in list_dir:
            # get samples file list name
            file_list = os.listdir(config.SAMPLES_DIR_PATH + "/" + directory + "")

            for ctr, filename in enumerate(file_list):
                sample = np.array(
                    np.loadtxt(config.SAMPLES_DIR_PATH + "/" + directory + "/" + filename + "", dtype=float))

                self.normalize_sample_to_timeseries(sample, id=directory, ts=ts, counter=ctr)

        print("Samples processing completed!")

        data = self.extract_features_from_timeseries(ts)

        print("Dumping templates data")
        with open(config.TEMPLATES_PATH, "w") as convert_file:
            convert_file.write(json.dumps(data))

    @staticmethod
    def normalize_sample_to_timeseries(sample, id, counter, ts=None):
        if ts is None:
            ts = {"id": [], "time": [], "m_x": [], "m_y": []}

        for temp in sample:
            ts["m_x"].append(temp[5])
            ts["m_y"].append(temp[6])
            ts["time"].append(temp[0])
            ts["id"].append(id + "_" + str(counter))

        return ts

    @staticmethod
    def extract_features_from_timeseries(timeseries):
        print("Feature extraction in progress...")

        data = {"label": [], "template": [], "features_name": []}

        settings = ComprehensiveFCParameters()

        if "matrix_profile" in settings.keys():
            del settings["matrix_profile"]

        extracted_features = extract_features(
            pd.DataFrame(timeseries),
            column_id="id",
            column_sort="time",
            n_jobs=8,
            show_warnings=False,
            disable_progressbar=False,
            profile=False,
            impute_function=impute,
            default_fc_parameters=settings
        )

        features_name = extracted_features.columns.tolist()

        # remove index at the end of the label if it is present and setting label list
        for idx, e in enumerate(extracted_features.iterrows()):
            if "_" in e[0]:
                label = "_".join(e[0].split("_")[:-1])
            else:
                label = e[0]
            data["label"].append(label)

        for features_list in extracted_features.to_numpy():
            ext_feat_list = []
            for feature in features_list:
                ext_feat_list.append(feature)

            data["template"].append(ext_feat_list)

        # save current features name list
        data["features_name"] = features_name

        print("Feature extraction completed!")
        return data

    def __select_feature_extracted_train(self, x_feature, y_label, features_name):

        df = pd.DataFrame(index=y_label, data=x_feature)
        df.columns = features_name

        print("Feature selection in progress...")

        # selected_feature evaluates the importance of the different extracted features
        selected_feature = select_features(df, pd.Series(data=y_label, index=y_label))

        # sort features based on p_values
        # relevance_table is the feature list
        relevance_table = calculate_relevance_table(selected_feature, pd.Series(data=y_label, index=y_label))
        relevance_table = relevance_table[relevance_table.relevant]
        relevance_table.sort_values("p_value", inplace=True)

        # filter the first N_RELEVANT_FEATURES ordered by p_value
        rel_features_name = relevance_table["feature"][:config.N_RELEVANT_FEATURES]

        print("Feature selection completed!")

        return rel_features_name

    def perform_evaluation(self):

        doddington = Doddigton(features=self.x_test, y_labels=self.y_test)

        # doddington zoo verification
        doddington.eval_verification()

        evaluation = Evaluation(x_features=self.x_test, y_labels=self.y_test, current_metric="euclidean")

        # verification
        evaluation.eval_verification()

        # identification
        evaluation.eval_identification()
