import json
import os

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tsfresh import extract_features, select_features
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.utilities.dataframe_functions import impute
from sklearn.preprocessing import MinMaxScaler

from src.recogniser.evaluation import Evaluation
from src.recogniser.doddington_zoo import Doddigton
from src.utilities import config


class WBBRecogniser:
    def __init__(self,
                 load_model_from_file=config.LOAD_DUMPS):

        self.load_model_from_file = load_model_from_file

        # templates data
        self.data = {}
        self.features_name = []

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.impostors = None

        # models
        self.standard_scaler = None
        self.lr_model = None
        self.kneighbors_classifier = None
        self.svm_model = None

        self.extracted_features = None

        """ init """
        if config.EXTRACT_FEATURE_FROM_SAMPLES:
            self.__extract_feature_from_samples()
        self.__setup_models()

    def __read_datas(self):
        with open(config.TEMPLATES_PATH) as json_file:
            self.data = json.load(json_file)
            self.features_name = self.data["features_name"]

    # Split dataset in train and test set
    def __split_train_test(self):

        print("Loading templates data")
        self.__read_datas()

        print("Scaling features")
        scaler = MinMaxScaler()
        scaler.fit(self.data["template"])
        scaled_features = scaler.transform(self.data["template"])

        print("Splitting dataset")
        x_train, x_test, y_train, y_test = train_test_split(scaled_features, self.data["label"], test_size=0.5, random_state=42)

        rel_features, y_label, features_name = self.__select_feature_extracted(
            x_feature=x_train,
            y_label=y_train,
            features_name=self.data["features_name"],
        )

        print("Splitting Dataset")

        df_test = pd.DataFrame(index=y_test, data=x_test)
        df_test.columns = self.data["features_name"]

        self.x_train = rel_features
        self.x_test = (df_test[features_name]).to_numpy()
        self.y_train = y_train
        self.y_test = y_test

        self.features_name = features_name.tolist()

    def __setup_models(self):

        # setup train and test datas
        self.__split_train_test()

        if self.load_model_from_file:
            print("Loading models data")
            self.standard_scaler = load(config.STANDARD_SCALER_DUMP_PATH)
            self.lr_model = load(config.LR_MODEL_DUMP_PATH)
            self.kneighbors_classifier = load(config.KNEIGHBORS_CLASSIFIER_DUMP_PATH)
            self.svm_model = load(config.SVM_MODEL_DUMP_PATH)

        else:
            print("Initializing models")
            self.standard_scaler = StandardScaler()
            self.lr_model = LogisticRegression(max_iter=10000)
            self.svm_model = svm.SVC(probability=True)

        print("Models training in progress...")

        # Don't cheat - fit only on training data
        self.standard_scaler.fit(self.x_train)

        # apply same transformation
        transformed_x_train = self.standard_scaler.transform(self.x_train)
        transformed_x_test = self.standard_scaler.transform(self.x_test)

        self.lr_model.fit(transformed_x_train, self.y_train)
        self.svm_model.fit(transformed_x_train, self.y_train)

        k_range = range(1, 8)

        for k in tqdm(k_range):
            self.kneighbors_classifier = KNeighborsClassifier(n_neighbors=k)  # todo capire perché
            self.kneighbors_classifier.fit(transformed_x_train, self.y_train)

        print("Model accuracy for Logistic Regression: ", self.lr_model.score(transformed_x_test, self.y_test))
        print("Model accuracy for SVM: ", self.svm_model.score(transformed_x_test, self.y_test))

        if config.SAVE_DUMPS:
            print("Dumping models data")
            dump(self.standard_scaler, config.STANDARD_SCALER_DUMP_PATH)
            dump(self.lr_model, config.LR_MODEL_DUMP_PATH)
            dump(self.kneighbors_classifier, config.KNEIGHBORS_CLASSIFIER_DUMP_PATH)
            dump(self.svm_model, config.SVM_MODEL_DUMP_PATH)

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
    def extract_features_from_timeseries(timeseries, features_filter=None):
        print("Feature extraction in progress...")

        data = {"label": [], "template": [], "features_name": []}

        extracted_features = extract_features(
            pd.DataFrame(timeseries),
            column_id="id",
            column_sort="time",
            n_jobs=8,
            show_warnings=False,
            disable_progressbar=False,
            profile=False,
            impute_function=impute
        )
        
        # filter features if required
        if features_filter is not None:
            extracted_features = extracted_features[features_filter]
            features_name = features_filter
        # set features_name list with the extracted ones
        else:
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

        # save just one feature name list
        data["features_name"] = features_name

        print("Feature extraction completed!")
        return data

    def __select_feature_extracted(self, x_feature, y_label, features_name):

        df = pd.DataFrame(index=y_label, data=x_feature)
        df.columns = features_name

        print("Feature selection in progress...")
        # selected_feature evaluates the importance of the different extracted features
        selected_feature = select_features(df, pd.Series(data=y_label, index=y_label))

        if self.extracted_features is None:
            self.extracted_features = selected_feature

        relevance_table = calculate_relevance_table(selected_feature, pd.Series(data=y_label, index=y_label))
        relevance_table = relevance_table[relevance_table.relevant]
        relevance_table.sort_values("p_value", inplace=True)

        # update current features list
        self.features_name = relevance_table["feature"]        

        rel_features = df[relevance_table["feature"]].to_numpy()

        print("Feature selection completed!")

        return rel_features, y_label, relevance_table["feature"]

    def test_sample(self, pathname, directory="/"):

        sample = np.array(np.loadtxt(pathname, dtype=float))

        ts = self.normalize_sample_to_timeseries(sample, id=directory)
        data = self.extract_features_from_timeseries(pd.DataFrame(ts), self.features_name)

        scaled_templates = self.standard_scaler.transform(data["template"])

        print()
        print("lrModel: ", self.lr_model.predict(scaled_templates))
        print("neigh: ", self.kneighbors_classifier.predict(scaled_templates))
        print("svm: ", self.svm_model.predict(scaled_templates))

    def run_test(self, pathname):
        self.test_sample(pathname, directory="Test")

    def run_all_test(self):

        file_list = os.listdir(config.TEST_DIR_PATH + "/")
        probabilities = []

        for ctr, filename in enumerate(file_list):
            probabilities.append(self.test_sample("../data/Test/" + filename, directory="Test"))

    def perform_evaluation(self):

        #rel_features, y_label, features_name = self.__select_feature_extracted(
        #    x_feature=self.x_test,
        #    y_label=self.y_test,
        #    features_name=self.features_name,
        #)

        #scaler = MinMaxScaler()
        #scaler.fit(rel_features)
        #rel_features = scaler.transform(rel_features)

        # metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean',
        #            'hamming', 'jaccard', 'jensenshannon', 'matching', 'minkowski',
        #            'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']

        doddington = Doddigton(features=self.x_test, y_labels=self.y_test)

        # doddington zoo verification
        doddington.eval_verification()

        metrics = ["euclidean"]

        for metric in metrics:
            evaluation = Evaluation(features=self.x_test, y_labels=self.y_test, current_metric=metric)

            # verification
            evaluation.eval_verification()

            # identification
            evaluation.eval_identification()

