# AutoAttention


## Requirements
Please use `pip install -r requirements.txt` to setup the operating environment in `python3.5`.  
Note that we use [DeepCTR](https://github.com/shenweichen/DeepCTR) package and refer to the implementation of [DSIN](https://github.com/shenweichen/DSIN).

## Prepare data

1. Download Alimama Data: [Ad Display/Click Data on Taobao.com](https://tianchi.aliyun.com/dataset/dataDetail?dataId=56)
2. Extract the files into the `data/raw_data` directory
3. Follow the code of the data preprocessing in [DSIN](https://github.com/shenweichen/DSIN) to preprocess data

## Training and Evaluation
Run `python train_[model].py [model_type]` 

- Sum Pooling: `python train_base.py Base`
- MAF-S: `python train_base.py Base_All_Fields_Add`
- MAF-C: `python train_base.py Base_All_Fields_Concat`
- DIN: `python train_din.py DIN`
- DIN+: `python train_din.py All_Fields`
- DIEN: `python train_dien.py`
- DSIN: `python train_dsin.py`
- DotProduct: `python train_autoattention.py DotProduct`
- AutoAttention: `python train_autoattention.py AutoAttention_Prun 0.6`


