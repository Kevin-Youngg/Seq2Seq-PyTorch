# Seq2Seq-PyTorch
 Implementation of Seq2Seq(attetion, beamSearch...) with PyTorch

# 项目结构
.
├── README.md
├── checkpoints            #保存已训练的模型的参数
├── data
│   ├── __init__.py
│   ├── Dataset.py         #dataloader
│   ├── data_utils.py      #数据预处理相关操作
│   ├── test.txt           #测试数据
│   ├── test_ids.txt    
│   ├── train.txt          #训练数据
│   ├── train_ids.txt 
│   ├── valid.txt          #验证数据
│   ├── valid_ids.txt
│   └── vocab              #词典
├── main.py       
├── models
│   ├── __init__.py
│   ├── Attention.py
│   ├── Decoder.py
│   ├── Encoder.py
│   └──  Seq2Seq.py
├── requirements.txt
├── results
└── utils
    ├── __init__.py
    ├── Recorder.py       #记录训练过程
    ├── beamSearch.py     #集束搜索
    └── greadySearch.py   #贪婪搜索
