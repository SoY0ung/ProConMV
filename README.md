# ProConMV

Official repository of ProConMV: Provenance-Enabled Conceptual Framework for Interpretable Multi-View Diabetic Retinopathy Diagnosis(ICML 26)  


## Code

Run `python train.py` and `python test.py` for MFIDDR, or `python train_drtid.py` and `python test_drtid.py` for DRTiD.

### Datasets

Download the datasets from their official repositories:

- [MFIDDR](https://github.com/mfiddr/MFIDDR): a four-view dataset.
- [DRTiD](https://github.com/fdu-vts/drtid): a two-view dataset.

After downloading and preparing a dataset, update the dataset paths in the corresponding training and testing scripts. 

| Variable | Meaning | Current value in `train.py` |
| --- | --- | --- |
| `DATA_PATH` | Dataset root directory. | `./data/mfiddr` |
| `TRAIN_PATH` | Training image directory. | `./data/mfiddr/train_process` |
| `MASK_PATH` | Training mask directory. | `./data/mfiddr/train_mask_cleaned` |
| `TEST_PATH` | Validation and test image directory. | `./data/mfiddr/test_process` |
| `MASK_PATH2` | Validation and test mask directory. | `./data/mfiddr/test_mask_cleaned` |
| `train_csv_path` | CSV containing training sample IDs, lesion labels, and grading labels. | `./train.csv` |
| `test_csv_path` | CSV containing validation and test sample IDs, lesion labels, and grading labels. | `./test.csv` |

`test.py` defines its own `DATA_PATH`, `TEST_PATH`, `MASK_PATH2`, and `test_csv_path`; update these as well before testing. 



```bibtex
@inproceedings{luo2026proconmv,
  title={ProCon{MV}: Provenance-Enabled Conceptual Framework for Interpretable Multi-View Diabetic Retinopathy Diagnosis},
  author={Xiaoling Luo and Shuo Yang and Qihao Xu and Jiansong Zhang and Zhuoqin Yang and Zhihui Lai and Linlin Shen and Chengliang Liu},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026}
}
```
