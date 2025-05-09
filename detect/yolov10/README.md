# yolov10

主要解决了NMS实时推理的耗时问题。

one-to-many label assign : 意味着一个真实目标会被分配给多个候选区域作为正样本进行学习。

one-to-one label assign : 意味着一个真实目标只会被分配给一个候选区域作为正样本学习。

比如说，基于iou的时候，一张图像有一个物体的真实框，有10个anchor box，其中有6个与真实框的iou都大于0.5，那么在one-to-many label assign下，这6个anchor boxes都会被分配给这个物体作为正样本，而one-to-one label assign只会选择iou最大的那个anchor作为正样本。

one-to-many label assign需要NMS进行后处理，影响部署的速度，one-to-one label assign会带来额外的推理开销且效果不好。



## dual label assignmets策略

yolov10在训练的时候包含两个结构相同，但是参数不同的head，一个是one-to-many，一个是one-to-one，训练的时候两个head同时，而推理的时候只需要使用one-to-one的head，推理的时候不需要nms。