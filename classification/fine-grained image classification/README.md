# 细粒度图像分类

细粒度图像分类是对一个大类下的子类分类。

## HERBS

论文：Fine-grained Visual Classification with High-temperature Refinement and Background Suppression

提出的方法：高温细化和背景抑制

**整体结构**如下：

![HERBS](.\img\HERBS.png)

整体的网络结构跟聚合网络(PA-Net)以及 yolo 相似。

**背景抑制**

​        背景抑制模块利用分置信度得分将特征图划分为前景和背景，抑制低置信度区域的特征值，同时增强判别特征。主要的实现过程是：

1、生成分类图

​		具体来说就是主干网络生成的特征图 h（维度为C x H x W，在上图中表示的是第三列深橘色卷积之后的特征）经过一个分类器（线性变换），转换成一个分类图Y（维度为 cls_num x H x W）, 然后使用softmax计算概率，再沿通道维求平均值，生成了一个空间位置上的置信度（大小为 H x W），然后在W的维度求最大值所在的坐标，根据坐标选取置信度最高的特征进行合并分类预测。剩下的被丢弃的特征使用双曲正切函数处理，以抑制背景信息，增强背景信息与前景特征之间的对比度。

2、损失函数

​		对于被选择出来的前景特征，跟真实标签使用交叉熵来计算合并损失(loss_m)；对于被丢弃的背景特征，跟-1使用交叉熵来计算丢弃损失（loss_d）。

​		为了防止所有阶段的特征图经过筛选之后，仅在相同的位置有特征响应，而丢弃的特征被 tanh 函数映射到一个不受概率限制的范围，不对正向的分类有贡献价值，HERBS还将主干网络每层生成的特征图也通过分类器与真实标签使用交叉熵计算损失，并求平均，得到平均层（loss_l）。

​		最终的损失是三个损失的加权和：
$$
loss_{bs} = λ_m * loss_m + λ_d * loss_d + λ_l * loss_l
$$
**高温细化**

​		上面的结构图中灰色的矩形框表示的是分类器，可以看到自顶向下，每一个stage都有一个分类器（Ki_1），自底向上，每个stage也都有一个分类器（Ki_2）。高温细化模块目标是通过使 Ki_1 学习 Ki_2 的输出分布来提高模型的性能，使用了模型蒸馏的思想，希望通过细化目标函数，在早期学习更多样化更强的特征表示，同时允许后面的层更专注于更精细的细节。

Ki_1分类器的输出特征为Pi_1，Ki_2分类器的输出特征为Pi_2，如下：
$$
P_{i1}=LogSoftmax(Y_{i1}/T_e)
$$

$$
P_{i2}=LogSoftmax(Y_{i2}/T_e)
$$

细化损失为：
$$
loss_r=P_{i2}log(P_{i2}/P_{i1})
$$
HERBS的总损失可以表示为：
$$
loss = loss_{bs}+λ_rloss_r
$$
**总结**

​		整体来说，高温细化除了主干网络之外，还需要自顶而下和自底而上两个过程，一定程度上增加耗时。背景抑制部分，使用分类置信度来挑选特征，是一个不错的方法，不过中间涉及到一些排序，topk的算法，就目前瑞芯微的部署上会有一些问题。