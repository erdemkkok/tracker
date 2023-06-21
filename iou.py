# # import the necessary packages
# from collections import namedtuple
# import numpy as np
# import cv2
# Detection = namedtuple("Detection", ["image_path", "gt", "pred"])
# def bb_intersection_over_union(boxA, boxB):
# 	xA = max(boxA[0], boxB[0])
# 	yA = max(boxA[1], boxB[1])
# 	xB = min(boxA[2], boxB[2])
# 	yB = min(boxA[3], boxB[3])
# 	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
# 	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
# 	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
# 	iou = interArea / float(boxAArea + boxBArea - interArea)
# 	return iou
# examples = [
# 	Detection("image_0002.jpg", [39, 63, 203, 112], [54, 66, 198, 114]),
# 	Detection("image_0016.jpg", [49, 75, 203, 125], [42, 78, 186, 126]),
# 	Detection("image_0075.jpg", [31, 69, 201, 125], [18, 63, 235, 135]),
# 	Detection("image_0090.jpg", [50, 72, 197, 121], [54, 72, 198, 120]),
# 	Detection("image_0120.jpg", [35, 51, 196, 110], [36, 60, 180, 108])]
# for detection in examples:
# 	image = cv2.imread(detection.image_path)

# 	cv2.rectangle(image, tuple(detection.gt[:2]), 
# 		tuple(detection.gt[2:]), (0, 255, 0), 2)
# 	cv2.rectangle(image, tuple(detection.pred[:2]), 
# 		tuple(detection.pred[2:]), (0, 0, 255), 2)
# 	iou = bb_intersection_over_union(detection.gt, detection.pred)
# 	cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30),
# 		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
# 	print("{}: {:.4f}".format(detection.image_path, iou))
# 	cv2.imshow("Image", image)
# 	cv2.waitKey(0)

#----------
import time
import numpy as np
def get_iou(ground_truth, pred):
    ix1 = np.maximum(ground_truth[0], pred[0])
    iy1 = np.maximum(ground_truth[1], pred[1])
    ix2 = np.minimum(ground_truth[2], pred[2])
    iy2 = np.minimum(ground_truth[3], pred[3])
     
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
     
    area_of_intersection = i_height * i_width
     
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1     
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1
     
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
     
    iou = area_of_intersection / area_of_union
     
    return iou
start=time.time()
ground_truth_bbox = np.array([1202, 123, 1650, 868], dtype=np.float32)
 
prediction_bbox = np.array([1162.0001, 92.0021, 1619.9832, 694.0033], dtype=np.float32)

iou = get_iou(ground_truth_bbox, prediction_bbox)
print('IOU: ', iou)
end=time.time()
print("Normal->",end-start)

#------

import torch
from torchvision import ops

def get_iou_torch(ground_truth, pred):
    ix1 = torch.max(ground_truth[0][0], pred[0][0])
    iy1 = torch.max(ground_truth[0][1], pred[0][1])
    ix2 = torch.min(ground_truth[0][2], pred[0][2])
    iy2 = torch.min(ground_truth[0][3], pred[0][3])
    
    i_height = torch.max(iy2 - iy1 + 1, torch.tensor(0.))
    i_width = torch.max(ix2 - ix1 + 1, torch.tensor(0.))
    
    area_of_intersection = i_height * i_width
    
    gt_height = ground_truth[0][3] - ground_truth[0][1] + 1
    gt_width = ground_truth[0][2] - ground_truth[0][0] + 1
    
    pd_height = pred[0][3] - pred[0][1] + 1
    pd_width = pred[0][2] - pred[0][0] + 1
    
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
    
    iou = area_of_intersection / area_of_union
    
    return iou
start=time.time()
ground_truth_bbox = torch.tensor([[1202, 123, 1650, 868]], dtype=torch.float)
 
prediction_bbox = torch.tensor([[1162.0001, 92.0021, 1619.9832, 694.0033]], dtype=torch.float)

iou_val = get_iou_torch(ground_truth_bbox, prediction_bbox)
print('IOU : ', iou_val.numpy())
end=time.time()
print("TORCH->",end-start)