# m3dv
上海交大2019-2020机器学习课程，医学图像分类
# 环境
python==3.6.7  
pytorch==1.3.1  
cuda9.1
# 硬件
Tesla V100 32GB
# 使用
　　运行test.py -m train使用训练模式，运行test.py -m test使用测试模式，如无-m参数默认使用测试模式。  
　　运行前请确保当前目录下存在所有文件，其中train_val文件夹中存放393+72个数据集，test文件夹中存放117个测试集。train.csv为训练集目录，val.csv为验证集目录，sampleSubmission.csv为测试集目录，submission.csv为输出的提交文件，best_in_mynet.pkl为保存的网络参数。  
　　可在test.py的前几行更改训练集及测试集路径。
