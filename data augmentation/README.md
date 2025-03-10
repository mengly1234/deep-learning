# 数据增强

## 综述

数据增强：增加有限数据的数量和多样性，提高模型的泛化能力。

### 一、单数据变形

1、几何变换：

方法：通过旋转、镜像、平移、裁剪、缩放和扭曲生成新的样本。

缺点：对数据重复记忆、增加信息量有限

2、色域变换：

方法：颜色抖动、PCA抖动、高斯抖动

缺点：在很多分类任务中，颜色比较重要，经过某些颜色抖动之后，可能会丢失一些重要的颜色信息，改变图像原有的语义信息，通道的选择十分重要。

3、清晰度变换

方法：使用核滤波器对图像进行锐化和模糊处理

4、噪声注入

方法：人为的对图像加噪声干扰，为数据集引入冗余和干扰信息，模拟生成不同成像质量的图像

常见噪声种类：高斯噪声、瑞利噪声、伽马噪声、均匀噪声、椒盐噪声等

5、局部擦除

方法：随机擦除、Cutout正则化、Hide-and-Seek、GridMask、不规则区域的局部擦除

注意：要观察数据集的特点

### 二、多数据混合

将多幅图像从图像空间或者特征空间进行信息混合。

1、图像空间的数据混合

对多幅图像进行线性叠加、非线性混合

方法：基于线性混合图像：SamplePairing、mixup 和 between-class learning

2、特征空间数据混合

借助CNN提取的图像特征，在特征空间进行数据增广，针对图像数据，在特征空间进行数据混合的方法很少被采用

方法：SMOTH方法、在特征空间外插值

注：在数据空间进行图像变换优于特征空间变换

### 三、学习数据分布

1、生成对抗网络

2、图像风格迁移

### 四、学习增广策略

1、基于元学习

2、基于强化学习



**补充**：yolov5所用的增强方法：

fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)
hsv_h: 0.015  # 色相
hsv_s: 0.7  # 饱和度
hsv_v: 0.4  # 亮度
degrees: 0.0  # 旋转
translate: 0.1  # 平移
scale: 0.5  # 缩放
shear: 0.0  # 剪切变换强度（倾斜变形）
perspective: 0.0  # 透视变换强度（模拟3D视角变化）
flipud: 0.0  # 垂直翻转
fliplr: 0.5  # 水平翻转
mosaic: 1.0  # Mosaic增强（四张图拼接）
mixup: 0.0  # 剪切混合
copy_paste: 0.0  # 将一张图上的目标复制粘贴到其他图上

