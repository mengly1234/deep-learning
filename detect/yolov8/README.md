# yolov8

yolov8主要借鉴了yolov5，yolov6，yoloX等模型的设计优点，偏重在工程实践上上。

## backbone

同样是借鉴了CSP模块的思想，将yolov5中的C3替换成了C2f模块，C3是将输入特征并行的进行两次卷积，每个卷积的输出是目标输出的一半，其中一部分的卷积再经过n次Bottleneck之后与另外一半的特征进行concat，然后共同再进行一次卷积，一共进行了三次卷积操作。C2f先通过一个卷积，然后将卷积之后的特征split，一部分经过n次Bottleneck之后再和另外一部分的卷积concat，最后再卷积一次，一共执行了两次卷积。

## neck

继续使用PAN的思想

## head

和yolov6相似，yolov8的head部分使用解耦头结构，将分类头和检测头分开，不再使用anchor-base，和v6一样使用anchor-free。

## loss

#### 正负样本分配策略：

Task-Aligned Assigner：

简单来说就是针对所有像素点预测的cls score和reg score，通过加权的方式得到最终的加权分数，通过加权分数进行排序后，选择topk个样本。

#### loss计算：

yolov8没有了之前的object分支，只有分类分支和检测分支。分类分支使用BCE Loss，检测头的回归损失函数使用的是CIou_Loss和Distribution Focal Loss。
