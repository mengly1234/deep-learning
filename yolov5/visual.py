# import numpy as np
# import torch
# from utils.augmentations import letterbox
# import cv2
# model = torch.load('best.pt')
# print(model)
#
# def img_process(img_path, img_size, stride, auto):
#     img = cv2.imread(img_path)
#     im = letterbox(img, img_size, stride=stride, auto=auto)[0]  # padded resize
#     im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
#     im = np.ascontiguousarray(im)  # contiguous
#     im = torch.from_numpy(im).to(model.device)
#     im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
#     im /= 255  # 0 - 255 to 0.0 - 1.0
#     if len(im.shape) == 3:
#         im = im[None]  # expand for batch dim
#
# def reference():
