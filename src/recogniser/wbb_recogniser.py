import json
import os

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tsfresh import extract_features
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
        self.mlp_classifier = None
        self.lr_model = None
        self.kneighbors_classifier = None

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
        x_feature = self.data['template']
        y_label = self.data['label']

        x_train, x_test, y_train, y_test = train_test_split(x_feature, y_label, test_size=0.1, random_state=42)

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def __setup_models(self):

        # setup train and test datas
        self.__split_train_test()

        if self.load_model_from_file:
            self.standard_scaler = load(config.STANDARD_SCALER_DUMP_PATH)
            self.mlp_classifier = load(config.MLP_CLASSIFIER_DUMP_PATH)
            self.lr_model = load(config.LR_MODEL_DUMP_PATH)
            self.kneighbors_classifier = load(config.KNEIGHBORS_CLASSIFIER_DUMP_PATH)

        else:
            self.standard_scaler = StandardScaler()
            self.mlp_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1,
                                                max_iter=10000)
            self.lr_model = LogisticRegression(max_iter=10000)

        # Don't cheat - fit only on training data
        self.standard_scaler.fit(self.x_train)

        # apply same transformation
        self.x_train = self.standard_scaler.transform(self.x_train)
        self.x_test = self.standard_scaler.transform(self.x_test)

        self.mlp_classifier.fit(self.x_train, self.y_train)

        self.lr_model.fit(self.x_train, self.y_train)

        k_range = range(1, 8)

        for k in tqdm(k_range):
            self.kneighbors_classifier = KNeighborsClassifier(n_neighbors=k)
            self.kneighbors_classifier.fit(self.x_train, self.y_train)

        if config.SAVE_DUMPS:
            dump(self.standard_scaler, config.STANDARD_SCALER_DUMP_PATH)
            dump(self.mlp_classifier, config.MLP_CLASSIFIER_DUMP_PATH)
            dump(self.lr_model, config.LR_MODEL_DUMP_PATH)
            dump(self.kneighbors_classifier, config.KNEIGHBORS_CLASSIFIER_DUMP_PATH)

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

        data = {"label": [], "template": []}

        for elem in templates:
            extracted_features = impute(
                extract_features(
                    elem,
                    column_id="id",
                    column_sort="time",
                    n_jobs=1,
                    show_warnings=False,
                    disable_progressbar=False,
                    profile=False,
                )
            )

            data["label"].append(elem["id"][0])
            ext_feat_list = []

            for e in extracted_features.to_numpy()[0]:
                ext_feat_list.append(e)
            data["template"].append(ext_feat_list)

        with open(config.TEMPLATES_PATH, 'w') as convert_file:
            convert_file.write(json.dumps(data))

    def test_sample(self, pathname):

        ts = {'id': [], 'time': [], 'm_x': [], 'm_y': []}
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
                             n_jobs=1,
                             show_warnings=False,
                             disable_progressbar=True,
                             profile=False))

        list = []
        for e in extracted_features.to_numpy()[0]:
            list.append(e)

        templates.append(list)

        templates_clf = self.standard_scaler.transform(templates)

        print()
        print("clf: ", self.mlp_classifier.predict(templates_clf))
        print("lrModel: ", self.lr_model.predict(templates_clf))
        print("neigh: ", self.kneighbors_classifier.predict(templates))

    def run_test(self, pathname):
        self.test_sample(pathname)
