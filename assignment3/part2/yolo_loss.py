from shutil import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
    
def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.mse = nn.MSELoss(reduction= "sum")

    def xywh2xyxy(self, boxes):
        """
        Parameters:
        boxes: (N,4) representing by x,y,w,h

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """
        ### CODE ###
        # Your code here
        x1_y1 = torch.sub(torch.div(boxes[:,:2], self.S), torch.mul(boxes[:,2:4], 0.5))
        x2_y2 = torch.add(torch.div(boxes[:,:2], self.S), torch.mul(boxes[:,2:4], 0.5))

        return torch.cat((x1_y1, x2_y2), dim=1)

    def find_best_iou_boxes(self, pred_box_list, box_target):
        """
        Parameters:
        box_pred_list : [(tensor) size (-1, 5) ...]
        box_target : (tensor)  size (-1, 4)

        Returns:
        best_ious: (tensor) size (-1, 1)
        best_boxes : (tensor) size (-1, 5), containing the boxes which give the best iou among the two (self.B) predictions

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) For finding iou's use the compute_iou function
        3) use xywh2xyxy to convert bbox format if necessary,
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        """
        ## CODE ###
        # Your code here
        copy_one = pred_box_list[0].clone().detach().requires_grad_(True)
        copy_two = pred_box_list[1].clone().detach().requires_grad_(True)
        copy_target = box_target.clone().detach().requires_grad_(True)
        box_1 = self.xywh2xyxy(torch.flatten(copy_one[:,0:4], end_dim=-2))
        box_2 = self.xywh2xyxy(torch.flatten(copy_two[:,0:4], end_dim=-2))
        target_box = self.xywh2xyxy(torch.flatten(copy_target, end_dim=-2))

        iou_1 = torch.diagonal(compute_iou(box_1, target_box), 0)
        iou_2 = torch.diagonal(compute_iou(box_2, target_box), 0)
        
        best_boxes = torch.zeros_like(pred_box_list[0])
        max_iou = torch.zeros_like(iou_1)

        for i in range(len(iou_1)):
        
            iou1 = iou_1[i]
            iou2 = iou_2[i]
            
            if iou1 >= iou2:
                max_iou[i] = iou1
                best_boxes[i] = pred_box_list[0][i, :]
            
            else:
                max_iou[i] = iou2
                best_boxes[i] = pred_box_list[1][i, :]

        return torch.reshape(max_iou, (-1,1)).detach(), torch.reshape(best_boxes, (-1,5)).requires_grad_(True)



    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar
        """
        ### CODE ###
        # Your code 
        assert classes_pred.shape == classes_target.shape, "get_class_prediction_loss input dimension incorrect"
        return self.mse(classes_pred[has_object_map], classes_target[has_object_map])

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """
        Parameters:
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (N, S, S)

        Returns:
        loss : scalar

        Hints:
        1) Only compute loss for cell which doesn't contain object
        2) compute loss for all predictions in the pred_boxes_list list
        3) You can assume the ground truth confidence of non-object cells is 0
        """
        ### CODE ###
        # Your code here
        # loss = 0
        # for box in pred_boxes_list:
            # loss += torch.sum(torch.square(box[~has_object_map][4]))

        # return loss

        return torch.sum(torch.square(pred_boxes_list[0][:,:,:,4] * ~has_object_map)) + torch.sum(torch.square(pred_boxes_list[1][:,:,:,4] * ~has_object_map))

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar

        Hints:
        The box_target_conf should be treated as ground truth, i.e., no gradient

        """
        ### CODE
        # your code here
        assert box_pred_conf.shape == box_target_conf.shape, "get_contain_conf_loss input dimension incorrect"
        return self.mse(box_pred_conf, box_target_conf.detach())

    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 4)
        box_target_response : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar

        """
        ### CODE
        # your code here
        assert box_pred_response.shape == box_target_response.shape, "get_regression_loss input dimension incorrect"
        x = self.mse(box_pred_response[:,0], box_target_response[:,0])
        y = self.mse(box_pred_response[:,1], box_target_response[:,1])
        first_term = (x + y)
        w = self.mse(torch.sqrt(box_pred_response[:,2]), torch.sqrt(box_target_response[:,2]))
        h = self.mse(torch.sqrt(box_pred_response[:,3]), torch.sqrt(box_target_response[:,3]))
        second_term = (w + h)

        return (first_term + second_term)

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) N:batch_size
                      where B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """
        N = pred_tensor.size(0)
        total_loss = 0.0
        # split the pred tensor from an entity to separate tensors:
        # -- pred_boxes_list: a list containing all bbox prediction (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        # -- pred_cls (containing all classification prediction)
        pred_boxes_list = []
        tensor1, tensor2, tensor3 = torch.split(pred_tensor, [5, 5, 20], dim=3)
        pred_boxes_list = [tensor1, tensor2]
        pred_cls = tensor3
        # compute classification loss
        class_loss = self.get_class_prediction_loss(pred_cls, target_cls, has_object_map)
        # compute no-object
        noobj_loss = self.get_no_object_loss(pred_boxes_list, has_object_map)
        # Re-shape boxes in pred_boxes_list and target_boxes to meet the following desires
        # 1) only keep having-object cells
        # 2) vectorize all dimensions except for the last one for faster computation
        for index in range(len(pred_boxes_list)):
            pred_boxes_list[index] = torch.reshape(torch.flatten(pred_boxes_list[index][has_object_map], end_dim=-2), (-1,5))
        target_boxes = torch.reshape(torch.flatten(target_boxes[has_object_map], end_dim=-2), (-1,4))
        # find the best boxes among the 2 (or self.B) predicted boxes and the corresponding iou
        best_ious, best_boxes = self.find_best_iou_boxes(pred_boxes_list, target_boxes)
        if best_boxes.requires_grad == False: best_boxes.requires_grad_(True)        
        #TODO: fix this mother fucker
        assert best_ious.requires_grad == False, "best_ious not detached"
        # compute regression loss between the found best bbox and GT bbox for all the cell containing objects
        reg_loss = self.get_regression_loss(best_boxes[:,0:4], target_boxes)
        # compute contain_object_loss
        obj_loss = self.get_contain_conf_loss(best_boxes[:,4:5], best_ious)
        # compute final loss
        total_loss = class_loss + self.l_noobj * noobj_loss + self.l_coord *  reg_loss + obj_loss
        # print(class_loss, noobj_loss, reg_loss, obj_loss)
        # construct return loss_dict
        loss_dict = dict(
            total_loss= total_loss / N,
            reg_loss= reg_loss / N,
            containing_obj_loss=obj_loss / N,
            no_obj_loss=noobj_loss / N,
            cls_loss=class_loss / N,
        )
        return loss_dict


