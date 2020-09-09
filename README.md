# DeepLabv3-_Improved
改进的DeepLabv3+（基于PixelShuffle）

1）搭建环境（推荐）：
Python=3.6.7
keras ==2.2.4
Cython == 0.28.2
pydensecrf == 1.0rc3
tensorflow-gpu=1.13.1

（2）数据准备（制作自己的数据集）
get_data_new.ipynb
或get_data.ipynb生成自己的pkl


（3）模型训练（train.ipynb）
训练完成后保存模型为.h5文件


tensorboard：
在生成的mytensorboard文件中

（4）图像分割预测（predict.ipynb）
使用训练好的模型.h5文件


预测图像路径设置：（例）
files = glob.glob("../1-50/*.jpg")


