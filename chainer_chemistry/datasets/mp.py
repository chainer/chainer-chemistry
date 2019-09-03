import os
import json
import pickle
import ast
import numpy as np
import pandas as pd


import chainer


class MPDataset(chainer.dataset.DatasetMixin):
    """
    """
    data_range_dict = {
        "poisson_ratio": {"min": -1, "max": 1/2.},
        "band_gap": {"min": 0.},
    }

    def __len__(self):
        """
        """
        return len(self.data)

    def save_pickle(self, path):
        """
        """
        print("saving dataset into {}".format(path))
        with open(path, "wb") as file_:
            pickle.dump(self.data, file_)

    def load_pickle(self, path):
        """
        """
        print("loading dataset from {}".format(path))
        if os.path.exists(path) is False:
            print("Fail.")
            return False
        with open(path, "rb") as file_:
            self.data = pickle.load(file_)

        return True

    def _load_data_list(self, data_dir, target_list):
        """
        ここでCSVの読み込みをする
        基本的にmp-idと正解ラベルのCSVを読み込む
        """
        # load csv
        id_prop_data = pd.read_csv(os.path.join(data_dir, "property_data.csv"), index_col=0)
        stability_data = pd.read_csv(os.path.join(data_dir, "stability_data.csv"),
                                     index_col=0, converters={3: ast.literal_eval})
        id_prop_data = id_prop_data.merge(stability_data, on="material_id")
        # drop data which has warnings
        n_warns = np.array([len(d) for d in id_prop_data["warnings"]])
        mask = n_warns == 0
        id_prop_data = id_prop_data[mask]
        # drop data which doesn't have fermi energy data
        id_prop_data = id_prop_data[~np.isnan(id_prop_data["efermi"].values)]

        if "band_gap" in target_list:
            id_prop_data = id_prop_data[id_prop_data["band_gap"].values > 0]

        if "K_VRH" in target_list or "G_VRH" in target_list or "poisson_ratio" in target_list:
            id_prop_data = id_prop_data[id_prop_data["K_VRH"] >= 1]
            id_prop_data = id_prop_data[id_prop_data["G_VRH"] >= 1]

        self.id_prop_data = id_prop_data
