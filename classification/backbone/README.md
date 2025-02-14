# mobilenet系列

## mobilenetV1（2017）

论文地址：https://arxiv.org/pdf/1704.04861

主要是使用深度可分离卷积

深度可分离卷积包含两个过程：逐通道卷积（depthwise convolution）和逐点卷积（pointwise convolution）

DWConv:

![dwconv](.\imgs_backbone\DWConv.jpg)

PWConv:

![pwconv](.\imgs_backbone\PWConv.jpg)

1、逐通道卷积：

逐通道卷积是一个卷积核负责一个通道，比如说，现在有一个28X28X256 的 feature map，那么逐通道卷积就一共有256个卷积核，每一个卷积核负责一个通道，经过逐通道卷积之后得到的 feature map 和输入的 feature map 的通道数是一样的，没有改变。

2、逐点卷积：

前面的逐通道卷积并不改变特征的通道数，所以无法拓展 feature map，而且由于每个通道独立进行卷积运算，没有有效的利用不同通道在相同空间位置上的特征信息，因此需要逐点卷积来进行空间信息上的组合并且扩展特征的通道维度。

逐点卷积的每个卷积核的通道数和输入特征的通道数是一样的，但是卷积核的宽高都为1，也就是1X1的卷积，也就是将特征在通道方向上进行加权组合。例如，现在有一个28X28X256的特征，那么逐点卷积的一个卷积核的大小是1X1X256，卷积核的个数可有网络结构设计而定。



## mobilenetV2（2018）

论文地址：https://arxiv.org/pdf/1801.04381

在mobilenetV1的基础上，增加了倒置残差结构（Inverted Residuals）和线性瓶颈层（Linear Bottlenecks）。

1、倒置残差结构：

由于mobilenet使用的是深度可分离卷积，DWConv每个卷积核的通道数都是1，就是导致特征通道数太少，不利于提取丰富的特征信息，于是V2设计了倒置残差结构。与ResNet的残差结构不同，Inverted Residuals输入和输出的通道数很少，在中间对卷积通道数进行扩展。

![Inverted_Residuals](.\imgs_backbone\Inverted_Residuals.png)

在Inverted_Residuals中的扩张层和深度可分离卷积之后，使用的激活函数是ReLU6。Inverted Residuals 具体的实现如下：

![mobile](.\imgs_backbone\mobileV2.png)



2、线性瓶颈层

如上图所示，在倒置残差结构里面的最后一层没有使用ReLU激活函数，还是使用的是一个1x1的线性层，防止非线性操作破坏过多的信息，避免信息损失。

## mobilenetV3（2019）

mobilenetV3整体还是采用了v2的结构，主要改进点在于：

1、使用NAS执行模块级搜索，构建一个高速的框架。

2、加入了SEAttention。

3、将最后一步的平均池化层前移并移除最后一个卷积层，引入h-swish激活函数。

![v3](.\imgs_backbone\mobilenetv3.png)

## mobilenetV4（2024）



# GhostNet系列

## GhostNetV1

​		作者提出，在一个训练好的神经网络中，会存在很多的冗余特征，而这些冗余的特征可以通过一些廉价的操作从别的特征图变换中得到。因此作者的想法是，通过卷积得到一部分的特征，然后通过一些廉价的操作从这些特征图中得到冗余的特征。ghost卷积的结构如下：

![image-20250214165043231](.\imgs_backbone\ghost.png)

卷积过程：

（1）先通过卷积生成一部分的特征图。

（2）对每个通道通过廉价的线性变换（其实就是对每个通道进行一次卷积，等价于深度卷积）。

（3）将第一步生成的特征图与第二步生成的特征图拼接到一起。

ghost bottleneck:

![image-20250214165338225](.\imgs_backbone\ghost_bottleneck.png)

（1）stride = 1的情况：

      当stride=1的时候，不进行下采样，特征图的宽高保持不变。这一部分思想类似mobilenetV2，先通过一个Ghost module用作扩展层，增加特征的通道数，扩展出冗余的特征，这是为了防止低维度的特征在ReLU激活的时候造成信息丢失。经过ReLU激活之后使用一个Ghost module降低特征维度，减少计算量。

（2）stride = 2的情况：

      当stride=2的时候，对特征进行下采样。两个Ghost module作用和stride=1的时候是相同的，除此之外在两个Ghost module之间加入一个stride=2的DWConv，实现下采样的功能。

## GhostNetV2

