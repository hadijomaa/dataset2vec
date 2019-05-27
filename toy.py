import numpy as np
from sklearn import datasets
from sklearn.model_selection import KFold
from Dataset import Dataset
class Toy(Dataset):
    def __init__(self,number_of_data_sets=int(1e4),seed=0,split=None,folds=5):
        np.random.seed(seed)
        self.create_set(number_of_data_sets)
        perm = np.arange(number_of_data_sets)
        np.random.shuffle(perm)
        
        self.trn_data   = [self.trn_data[_] for _ in perm]
        self.trn_labels = [self.trn_labels[_] for _ in perm]
        self.trn_names  = [self.trn_names[_] for _ in perm]
        
        if split is None:
            pass
        else:
            assert(type(split)==int and split >-1 and split < folds)
            KF = KFold(folds)
            KF.get_n_splits(self.trn_data)
            for i,indices in enumerate(KF.split(self.trn_data)):
                if i==split:
                    self.tst_names     = [self.trn_names[_] for _ in indices[1]]
                    self.trn_names     = [self.trn_names[_] for _ in indices[0]]
                    
                    self.tst_labels     = [self.trn_labels[_] for _ in indices[1]]
                    self.trn_labels     = [self.trn_labels[_] for _ in indices[0]]
                    
                    self.tst_data     = [self.trn_data[_] for _ in indices[1]]
                    self.trn_data     = [self.trn_data[_] for _ in indices[0]]                    
                    break
            
    def _generate_circles(self):
        n_samples     = 2**np.random.randint(low=11,high=14,size=1)[0]
        random_state  = np.random.randint(low=0,high=100,size=1)[0]
        factor        = np.random.rand(1)[0]
        noise         = np.random.randint(low=0,high=2,size=1)[0]
        if noise==1:
            noise = np.random.rand(1)[0]
        X,y                   = datasets.make_circles(n_samples=n_samples,
                                                      factor=factor,noise=noise,
                                                      random_state=random_state)
        return X,y
        
    def _generate_blobs(self):
        n_samples     = 2**np.random.randint(low=11,high=14,size=1)[0]
        random_state  = np.random.randint(low=0,high=100,size=1)[0]
        n_features    = 2
        centers       = np.random.randint(low=2,high=8,size=1)[0]
        X,y           = datasets.make_blobs(n_samples=n_samples,n_features=n_features,
                                            random_state=random_state,centers=centers)
        return X,y
    
    def _generate_moons(self):
        n_samples     = 2**np.random.randint(low=11,high=14,size=1)[0]
        random_state  = np.random.randint(low=0,high=100,size=1)[0]
        noise         = np.random.randint(low=0,high=2,size=1)[0]
        if noise==1:
            noise = np.random.rand(1)[0]        
        X,y           = datasets.make_moons(n_samples=n_samples,
                                            random_state=random_state,noise=noise)
        return X,y        
        
    def create_set(self,n_datasets):
        self.trn_data = []
        self.trn_labels = []
        self.trn_names = []
        for i in range(n_datasets):
            choice = np.random.randint(low=0,high=3,size=1)[0]
            if choice==0:
                data,labels = self._generate_blobs()
            elif choice==1:
                data,labels = self._generate_circles()
            elif choice==2:
                data,labels = self._generate_moons()
                
            self.trn_data.append(data)
            self.trn_labels.append(labels)
            self.trn_names.append(choice)