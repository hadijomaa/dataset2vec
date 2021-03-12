# Dataset2Vec: Learning Dataset Meta-Features
We provide here the source code for our paper "Dataset2Vec: Learning Dataset Meta-Features".

## Usage
To train the metafeature extractor, run the d2v.py file.
```
python d2v.py 
```

To extract metafeatures from new datasets, run the extract_meta_features.py file. Please make sure
that your data follows the same format as the existing datasets, i.e. separate predictor and labels .dat 
files with no headers/indices.

```
python extract_meta_features.py --file "abalone" 
```

## Citing Dataset2Vec
-----------

To cite Dataset2Vec please reference our arxiv paper:


```
@article{jomaa2019dataset2vec,
  title={Dataset2vec: Learning dataset meta-features},
  author={Jomaa, Hadi S and Schmidt-Thieme, Lars and Grabocka, Josif},
  journal={Accepted at DAMI 2020},
  year={2019}
}
```

## How to get ISMLLDataset for python
You can install the ISMLLDataset package directly by

        pip install ismlldataset
        
### Examples
The following example shows how to read data:
```python
import ismlldataset

dataset_id = 31 # (between 0-119)

dataset = ismlldataset.datasets.get_dataset(dataset_id=dataset_id)

# get data
x,y     = dataset.get_data()

# get specific split
x,y = dataset.get_folds(split=1,return_valid=True)

train_x,valid_x,test_x = x

train_y,valid_y,test_y = y

```

We read metadata similarly:

```python
import ismlldataset

dataset_id = 31 # (between 0-119)

metadataset = ismlldataset.datasets.get_metadataset(dataset_id=dataset_id)

# get configurations and response
x,y = metadataset.get_meta_data()

# normalize response (optional)

metadataset.normalize_response()

# find configuration space
cs = metadataset.get_configuration_space()

# find response of a paricular configuration
sample = cs.sample_configuration()
response,is_valid = metadataset.objective_function(sample)

# get all loss curves
loss = metadataset.get_all_loss_curves()

# get all gradient information
gradients = metadataset.get_gradient_curve(sample)
```


**Under development**: The tasks of this package so far include hyperparameter optimization:
```python
import ismlldataset

dataset_id = 31

metadataset = ismlldataset.datasets.get_metadataset(dataset_id=dataset_id)

# select HPO approach
task = ismlldataset.tasks.Random(metadataset=metadataset,evaluation='acc')

# run and observe results
task.run(return_results=True)

```
## Disclaimer
The package is currently under development.
