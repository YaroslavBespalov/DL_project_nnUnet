# Global pipeline for all

Requirements:
- Cuda v9.0+
- PyTorch v0.4.1.post2
- albumentations

Before running the training or inference process you should specify configuration files (`configs/config.yml`, `configs/config_inference.yml`, `configs/path.yml`) 
and create a logic for loading the data (`src/dataset.py`). Moreover you can customise you own loss or metric functions in modules `src/losses.py`, `src/metrics.py` respectively.
All custom models (and contains some existing zoo models) implemented in `src/models.py`.

The core of this project - `src/youtrain/` [train loop](https://github.com/amirassov/youtrain).

The main wrappers for the core are `src/train.py` and `src/inference.py`.

Running all modules support multi-gpu. Please specify the `CUDA_VISIBLE_DEVICES` in `train.sh` and `inference.sh` files.

##### Run train:
```
sh train.sh
```

##### Run inference:
```
sh inference.sh
```

##### Below you can see the main fields in configuration files:
```
data_params:                    # all params related to data
  batch_size: 16                # batch size
  num_workers: 16               # num of process which will use for training
  augmentation_params:          # all augmentation params needed for you 
    resize: 320                

train_params:                   # all params related to training
  name: rn18                    # name of model for creating the logs and weights dirs
  model: models.MultiResnet18   # model which use for training
  model_params:                 # all model params
    num_filters: 16
    pretrained: True
    num_classes: 2
  loss: losses.BCELoss2d        # loss function 
  loss_params: {}               # loss parameters if there are no please specify the empty dict
  metrics: [metrics.MAP3]       # list of all metrics functions use for monitoring
  steps_per_epoch: 2500         # number of batches per epoch
  new_save: True                # create new weights for each training run
#  weights:                     # path to weights if any
stages:                         # stages of training
-
  load_best: False              # if set True - use best weights from the previous stage
  optimizer: Adam               
  optimizer_params:             
    lr: 0.001
  scheduler: ReduceLROnPlateau  # standard pytorch scheduler or any other specified in schedulers
  scheduler_params:             # schedule parameters
    patience: 5
    factor: 0.5
    min_lr: 0.000001
  epochs: 300                   # number of epochs
  augmentation: mix_transform   # name of augmentation function determined in src/transforms.py

```