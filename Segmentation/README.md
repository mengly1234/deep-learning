# 图像分割算法学习

## 图像分割大致介绍

​		图像分割是指将数字图像划分成不同的区域，这些区域在某些特征（颜色、纹理、亮度等）上具有一致性或者相似性。同时相邻区域之间的这些特征存在明显的差异。

​		图像分割还包括**语义分割**和**实例分割**两部分。简单来说，语义分割不区分同一类别的不同对象，而实例分割需要区分同一类别的不同实例，对每一个实例有一个独特的标签。从应用场景上来看，语义分割可以在自动驾驶的时候区别行人车辆等，实例分割更适用于对单个对象进行精确的操作和分析，比如说个体的行为分析和异常识别等。

下面都是针对语义分割任务的一些算法。

基于**传统学习**的图像分割方法有：

#### 1、阈值分割法：

​		如果图像比较简单，且目标与背景差距很大，比如说，检测的目标是红色的，背景是蓝色的，那么就可以根据rgb像素值手动计算阈值来划分是背景还是目标，或者计算r,g,b某一个像素值占取的比例来划分目标和背景，实现比较简单。但是阈值的寻找比较麻烦，需要计算背景和目标的rgb值，选择合适的阈值，且边缘划分不够精细。同样可以将rgb转换到hsv，使用hsv值来限定目标和背景。

opencv提供的一些阈值分割的方法：

（1）直方图法：

适用于直方图为双峰的图像，通过设定一个峰值阈值来划分前景和后景。

（2）最大类间方差法（OTSU）：

使用聚类的思想，把图像的灰度数分成两部分，使得两个部分之间的灰度值差异最大，每个部分之间的灰度差异值最小，通过方差的计算来寻找一个合适的灰度级别来划分。

（3）自适应阈值分割：

自适应阈值分割每一个像素的阈值都不用，某一个点的分割阈值是以该点为中心的一定范围内计算平均值，然后再减去一个delta。

此外还有最大熵分割和均值迭代，这两种方法都需要迭代计算，过程会比较慢。

#### 2、区域生长法：

​		区域生长是根据事先定义的准则将像素或者子区域聚合成更大区域的过程。其基本思想是从一组生长点开始（生长点可以是单个像素，也可以是某个小区域），将与该生长点性质相似的相邻像素或者区域与生长点合并，形成新的生长点，重复此过程直到不能生长为止。生长点和相似区域的相似性判断依据可以是灰度值、纹理、颜色等图像信息。

------

以下是基于**深度学习**的分割算法：

## U-Net

​		整体的网络结构如下：

![image-20250106152514190](.\imgs_seg\U-Net.png)

​		总体来说结构比较简单，可以分为两个阶段，第一个阶段是图中左半侧的编码阶段，叠加卷积层以及使用最大池化的下采样层，一共经过了四次下采样，特征图的宽高逐渐变小；第二阶段是右侧的解码阶段，叠加卷积以及使用反卷积的上采样层，特征图的宽高逐渐变大，和左侧相同，一共进行了四次上采样，四个阶段一一对应。由于最大池化层和反卷积层会造成信息丢失，因此每次下采样之前都会将特征图复制裁剪成对应上采样阶段的特征图的大小，与上采样阶段特征融合。



## FCN

论文地址：https://arxiv.org/pdf/1605.06211v1

​		通常分类任务的CNN是一系列的卷积层+若干个全连接层，将卷积产生的特征图映射成为一个固定长度的特征向量，然后根据这个固定长度的特征向量去对整张图像做一个分类。

​		而 FCN 是对图像进行像素级分类，也就是说，每个像素点都需要分类。FCN与通常的分类网络不同，移除了CNN最后的全连接层，将其替换成一个1*1的卷积，通过1x1的卷积，将通道数转换到类别数，然后通过转置卷积层将特征图变换为输入图像的宽高，使特征图恢复成了原图的大小。

​		此时，假如需要分割的类别数是C，就像相当于，每一个空间位置，是一个C维的向量来代表每一类别的得分，接下来的过程就和分类相同，拿网络预测的类别与标签对比，计算损失，优化网络。只是分割任务是要对每一个空间位置都进行分类。

![FCN](.\imgs_seg\FCN.png)

上采样使用的是双线性插值。



## Yolo

​	yolo系列从yolov5-7.0添加了分割任务。在检测任务的基础上，实现将检测框内识别出来的物体从背景中分割出来。

​	以yolov5为例，分割任务就是将检测任务的检测头换成了分割头，其他的结构未变。分割头的输入还是检测头三个阶段的特征图，他先使用conv+upsample+conv的结构对输入的最大分辨率的特征图上采样，然后下面紧跟着还是检测头，



## Segmentation-Based Deep-Learning Approach for Surface-Defect Detection

论文地址：https://arxiv.org/pdf/1903.08536v3

基于U-Net网络，是一个两阶段的表面缺陷检测方法。第一阶段是先训练一个分割网络，第二阶段是在分割网络上构建一个附加决策网络，以判断图像中是否存在异常。

![structure](.\imgs_seg\seg.png)

从上图中可以看出，第一阶段提出了一个分割网络，该网络对表面缺陷进行像素级定位，使用单个像素作为训练样本，从而增加了训练样本的数量，防止网络训练过程中出现过拟合现象。第二阶段使用第一阶段的输出特征进行二值分类，判断是否属于缺陷。

##### 为了检测出小目标缺陷，对分割网络进行如下修改：

1.使用额外的下采样层，且在较高层中使用较大的内核大小来显著增加感受野大小；

2.将每次下采样的层数更改为在架构的具有较少层的较低部分，这增加了具有较大感受野大小的特征容量；

3.最后，只用Maxpooling层而不是使用卷积层进行下采样，这将确保较小的特征在前向传播的过程中保存下来。

##### 网络训练细节：

​        利用交叉熵损失函数训练决策网络。学习与分割网络分开进行。首先，只独立训练分割网络，然后冻结分割网络的权重，只训练决策网络层。通过仅微调决策层，网络避免了分割网络中大量权重的过度拟合问题。这在学习决策层的阶段比在学习分割层的阶段更重要。在学习决策层时，GPU内存的限制将批大小限制为每批仅一个或两个样本，但在学习分割层时，图像的每个像素被视为一个单独的训练样本，因此将有效批大小增加了几倍。

