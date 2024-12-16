# yolov11

## backbone

改进了backbone，设计了C3k2机制，在网络的浅层，c3k参数设置为False，等于v8中的C2f，当c3k参数设置为True时，相当于在v8的bottleneck里面又套了一个循环，又有若干个bottleneck，整体模型的复杂度提高了。

## neck

就是在PAN的部分设计了一个PSABlock的模块，这个模块里面添加了多头注意力。

## head

相较于v8来说，v11在分类分支部分，将卷积换成了深度可分离卷积，来提高检测头的速度。