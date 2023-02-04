import json
import os

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute

from src.utilities import config


class WBBRecogniser:
    def __init__(self,
                 load_model_from_file=config.LOAD_DUMPS):

        self.load_model_from_file = load_model_from_file

        # templates data
        self.data = {}

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        # models
        self.standard_scaler = None
        self.lr_model = None
        self.kneighbors_classifier = None
        self.svm_model = None

        """ init """
        if config.EXTRACT_FEATURE_FROM_SAMPLES:
            self.__extract_feature_from_samples()
        self.__setup_models()

    def __read_datas(self):
        with open(config.TEMPLATES_PATH) as json_file:
            self.data = json.load(json_file)

    # Split dataset in train and test set
    def __split_train_test(self):
        self.__read_datas()

        x_feature = self.data["template"]
        y_label = self.data["label"]
        features_name = self.data["features_name"]

        df = pd.DataFrame(index=y_label, data=x_feature)
        df.columns = features_name

        # selected_feature evaluates the importance of the different extracted features
        selected_feature = select_features(df, pd.Series(data=y_label, index=y_label))

        x_train, x_test, y_train, y_test = train_test_split(selected_feature.to_numpy(), y_label, test_size=0.1,
                                                            random_state=42)

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def __setup_models(self):

        # setup train and test datas
        self.__split_train_test()

        if self.load_model_from_file:
            self.standard_scaler = load(config.STANDARD_SCALER_DUMP_PATH)
            self.lr_model = load(config.LR_MODEL_DUMP_PATH)
            self.kneighbors_classifier = load(config.KNEIGHBORS_CLASSIFIER_DUMP_PATH)
            self.svm_model = load(config.SVM_MODEL_DUMP_PATH)

        else:
            self.standard_scaler = StandardScaler()
            self.lr_model = LogisticRegression(max_iter=10000)
            self.svm_model = svm.SVC()

        # Don't cheat - fit only on training data
        self.standard_scaler.fit(self.x_train)

        # apply same transformation
        transformed_x_train = self.standard_scaler.transform(self.x_train)
        transformed_x_test = self.standard_scaler.transform(self.x_test)

        self.lr_model.fit(self.x_train, self.y_train)
        self.svm_model.fit(self.x_train, self.y_train)

        k_range = range(1, 8)

        for k in tqdm(k_range):
            self.kneighbors_classifier = KNeighborsClassifier(n_neighbors=k)  # todo capire perché
            self.kneighbors_classifier.fit(transformed_x_train, self.y_train)

        print("Model accuracy for Logistic Regression: ", self.lr_model.score(self.x_test, self.y_test))
        print("Model accuracy for SVM: ", self.svm_model.score(transformed_x_test, self.y_test))

        if config.SAVE_DUMPS:
            dump(self.standard_scaler, config.STANDARD_SCALER_DUMP_PATH)
            dump(self.lr_model, config.LR_MODEL_DUMP_PATH)
            dump(self.kneighbors_classifier, config.KNEIGHBORS_CLASSIFIER_DUMP_PATH)
            dump(self.svm_model, config.SVM_MODEL_DUMP_PATH)

    @staticmethod
    def __extract_feature_from_samples():
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

        data = {"label": [], "template": [], "features_name": []}

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
            features_name = extracted_features.columns.tolist()

            for e in extracted_features.to_numpy()[0]:
                ext_feat_list.append(e)

            data["template"].append(ext_feat_list)
            # save just one feature name list
            data["features_name"] = features_name

        with open(config.TEMPLATES_PATH, "w") as convert_file:
            convert_file.write(json.dumps(data))

    def test_sample(self, pathname):

        ts = {"id": [], "time": [], "m_x": [], "m_y": []}
        sample = np.array(np.loadtxt(pathname, dtype=float))

        for temp in sample:
            ts["m_x"].append(temp[5])
            ts["m_y"].append(temp[6])
            ts["time"].append(temp[0])
            ts["id"].append(pathname)

        ts = pd.DataFrame(ts)

        templates = []

        extracted_features = impute(
            extract_features(ts,
                             column_id="id",
                             column_sort="time",
                             n_jobs=5,
                             chunksize=5,
                             show_warnings=False,
                             disable_progressbar=True,
                             profile=False))

        list = []
        for e in extracted_features.to_numpy()[0]:
            list.append(e)

        templates.append(list)

        scaled_templates = self.standard_scaler.transform(templates)

        print()
        print("lrModel: ", self.lr_model.predict(scaled_templates))
        print("neigh: ", self.kneighbors_classifier.predict(scaled_templates))
        print("svm: ", self.svm_model.predict(scaled_templates))

    def run_test(self, pathname):
        self.test_sample(pathname)
