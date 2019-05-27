import numpy as np
import os
import pandas as pd
from sklearn.model_selection import KFold
from Dataset import Dataset
class UCI(Dataset):
    def __init__(self, folder_path='../data/uci',seed=0,split=None,folds=5):
        Dataset.__init__(self)
        np.random.seed(seed)
        self.trn_data           = []
        self.trn_labels         = []
        data_list      = os.listdir(folder_path)
        np.random.shuffle(data_list)
        if split is None:
            self.trn_names = data_list
        else:
            assert(type(split)==int and split >-1 and split < folds)
            KF = KFold(folds)
            KF.get_n_splits(data_list)
            for i,indices in enumerate(KF.split(data_list)):
                if i==split:
                    self.trn_names     = [data_list[_] for _ in indices[0]]
                    self.tst_names     = [data_list[_] for _ in indices[1]]
                    break

            self.tst_data   = []
            self.tst_labels = []
            for file in self.tst_names:
                self.tst_data.append(np.asarray(pd.read_csv(os.path.join(folder_path,file,file+'_py.dat'),header=None)))
                self.tst_labels.append(np.asarray(pd.read_csv(os.path.join(folder_path,file,'labels_py.dat'),header=None)))

        for file in self.trn_names:
            self.trn_data.append(np.asarray(pd.read_csv(os.path.join(folder_path,file,file+'_py.dat'),header=None)))
            self.trn_labels.append(np.asarray(pd.read_csv(os.path.join(folder_path,file,'labels_py.dat'),header=None)))