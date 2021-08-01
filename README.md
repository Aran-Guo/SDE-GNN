# SDE-GNN
**Sequential Dependency Enhanced Graph Neural
Network for Session-based Recommendation**

Wei Guo, Shoujin Wang, Wenpeng Lu∗
and Qian Zhang &nbsp;

The paper has been accepted by DSAA. 

**Prerequisites**

python 3

pytorch 1.1.0

**Usage**

You need to run the file `datasets/preprocess.py` first to preprocess the data. 

For example: `cd datasets; python preprocess.py --dataset=sample`

Then you can run the file `pytorch_code/main.py` to train the model.

For example: `cd pytorch_code; python main.py --dataset=sample`

**Dataset**

1. "Tmall: It contains anonymized user’s shopping logs on Tmall online shopping platform."
https://tianchi.aliyun.com/dataset/dataDetail?dataId=42

2. "Nowplaying: It describes the music listening behavior of users."
https://dbis.uibk.ac.at/node/263#nowplaying
