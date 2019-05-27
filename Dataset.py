from sklearn import preprocessing
import numpy as np

class Dataset(object):
    def __init__(self):
        pass
    def sample_batch(self,data,labels,Ns=None,Ls=None,Ms=None):
        ohc           = preprocessing.OneHotEncoder(n_values = len(np.unique(labels)),sparse=False)
        labels        = ohc.fit_transform(labels.reshape(-1,1))
        Ms            = np.random.choice(np.arange(2,data.shape[1]+1),size=1)[0]  if Ms is None else Ms
        Ls            = np.random.choice(np.arange(1,labels.shape[1]+1),size=1)[0] if Ls is None else Ls
        Ns            = np.minimum(2**np.random.choice(np.arange(4,9),size=1)[0],data.shape[0]) if Ns is None else Ns
        L_hat         = np.random.choice(np.arange(0,labels.shape[1]),size=Ls,replace=False)
        M_hat         = np.random.choice(np.arange(0,data.shape[1]),size=Ms,replace=False)
        N_hat         = np.random.choice(np.arange(0,data.shape[0]),size=Ns,replace=False)
        
        data          = data[N_hat]
        data          = data[:,M_hat]
        labels        = labels[N_hat]
        labels        = labels[:,L_hat]
        return data,labels

    def flatten(self,data,labels):
        first_element = []
        for c in range(labels.shape[1]):
            c_label = np.tile(labels[:,c],reps=[data.shape[1]]).transpose().reshape(-1,1)
            first_element.append(np.concatenate([data.transpose().reshape(-1,1),c_label],axis=-1))
        return np.vstack(first_element)
    
    def sample_batch_pairs(self,positive=False,first_element=None,test=False):
        if test:
            data    = self.tst_data
            labels  = self.tst_labels
        else:
            data = self.trn_data
            labels = self.trn_labels
            
        first_element = np.random.choice(np.arange(0,len(data)), 1,replace=False)[0] if first_element is None else first_element
        if not positive:
            done = False
            while not done:
                second_element = np.random.choice(np.arange(0,len(data)), 1,replace=False)[0]
                if first_element != second_element:
                    done = True
            related = 0
        else:
            second_element = first_element
            related = 1
        info = []
        
        X,Y = self.sample_batch(data=data[first_element],labels=labels[first_element])
        info.append(X.shape+(Y.shape[1],)+(first_element,))
        first_element = self.flatten(X,Y)
        del X,Y
        
        X,Y = self.sample_batch(data=data[second_element],labels=labels[second_element])    
        info.append(X.shape+(Y.shape[1],)+(second_element,))
        second_element = self.flatten(X,Y)
        del X,Y
        return np.vstack([first_element,second_element]),np.vstack(info),related
    
    def get_batch(self,batch_size,stratification_pos_ratio = 0.5):
        X = []
        I = []
        num_pos = int(batch_size*stratification_pos_ratio)
        for i in range(num_pos):
            batch_pair, batch_pair_info,batch_label = self.sample_batch_pairs(positive=True)
            X.append(batch_pair)
            I.append(batch_pair_info)
        for i in range(batch_size-num_pos):
            batch_pair, batch_pair_info,batch_label = self.sample_batch_pairs(positive=False)
            X.append(batch_pair)
            I.append(batch_pair_info)
            
        return np.vstack(X),np.vstack(I),int(num_pos)