# mws
code for "Learning to Detect Salient Object with Multi-source Weak Supervision"

# training SNet
The SNet is supervised by the natural image dataset with noisy pseudo ground-truth predicted by CNet and PNet on unlabeled saliency images and a synthetic image dataset created by composing detected salient objects and background images.

1. To generate the saliency masks of the above two dataset, run 
```
to be added XXXXX
``` 
The averaged map is processed with CRF and then binarized.

2. To create the synthetic image
dataset, run 
```
python test_crf.py
python syn_msk.py
```
3. train SNet, run 
```
python train_sal.py --name save_folder_name --model SalModel --batchSize 24
```
4. test the results of the SNet, run
```
python test_sal.py --name save_folder_name --model SalModel --batchSize 24
```
