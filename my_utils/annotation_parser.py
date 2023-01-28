"""
@author: wwjiang
@time: 2022/6/1: 17:35
"""

import os
import sys
import json 
import copy 
from uuid import uuid4 
from typing import Iterable, List, Optional, Dict, Union
import numpy as np
import cv2 
import scipy.spatial.distance as dist
import xml.etree.ElementTree as ET
curdir = os.path.dirname(__file__)
rtpath = os.path.join(curdir, "..")
sys.path.append(rtpath)
from my_utils.io_operation import json2dict


class OrderPoint:
    def __init__(self):
        "for vertical text "
        pass 
 
    @staticmethod
    def order_points_origin(pts):
        # sort the points based on their x-coordinates
        xSorted = pts[np.argsort(pts[:, 0]), :]
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]
        return np.array([tl, tr, br, bl], dtype="float32")
 
    @staticmethod
    def order_points_top_prior(pts):
        # sort the points based on their x-coordinates
        ySorted = pts[np.argsort(pts[:, 1]), :]
        topMost = ySorted[:2, :]
        bottomMost = ySorted[2:, :]
 
        topMost = topMost[np.argsort(topMost[:, 0] + topMost[:, 1]), :]
        topMost = topMost[np.argsort(topMost[:, 0] + topMost[:, 1]), :]
        (tl, tr) = topMost
 
        D = dist.cdist(tl[np.newaxis], bottomMost, "euclidean")[0]
        (br, bl) = bottomMost[np.argsort(D)[::-1], :]
        return np.array([tl, tr, br, bl], dtype="float32")
 
 
def four_point_transform(image, pts: np.ndarray):
    """
    obtain a consistent order of the points and unpack them
    """
    rect = OrderPoint.order_points_top_prior(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


class BaseOperation:
    def __init__(self):
        pass 

    def crop_by_polygon(self, img, points_ls: List[np.ndarray], save_rtpath=None, img_prefix=None):
        """
        Args:
            img(np.ndarray): origin image 
            points_ls(List[np.ndarray]): points matrix N * 2
        """
        crop_img_ls = []
        for points in points_ls:
            mini_rect = self.get_mini_rect(points=points) 
            crop_img = four_point_transform(image=img.copy(), pts=mini_rect)
            crop_img_ls.append(crop_img)
        if save_rtpath and img_prefix:
            for idx, cur_crop_img in enumerate(crop_img_ls):
                cv2.imwrite(os.path.join(save_rtpath, f"{img_prefix}_{idx:04d}.jpg"), cur_crop_img)
        return crop_img_ls

    @staticmethod
    def get_mini_rect(points):
        rect = cv2.minAreaRect(points)
        post_points = cv2.boxPoints(rect) 
        return post_points

    @staticmethod
    def rect2polygon(bbox_ls):
        bbox_ls_new = []
        for i in bbox_ls:
            [xmin, ymin], [xmax, ymax] = i 
            bbox_ls_new.append(np.array(
                [[xmin, ymin],
                 [xmax, ymin],
                 [xmax, ymax],
                 [xmin, ymax]]))
        return bbox_ls_new


class LabelStudioXmlParser(BaseOperation):
    def __init__(self):
        super(LabelStudioXmlParser, self).__init__()

    def parse_rectangle_single(self, filename, return_polygon=True): 
        tree = ET.parse(filename) # 解析读取xml函数
        label_ls = []
        bbox_ls = []
        
        for obj in tree.findall('object'):
            cur_label = obj.find('name').text
            cur_bnd = obj.find('bndbox')
            cur_bbox = np.array([int(cur_bnd.find('xmin').text),
                                 int(cur_bnd.find('ymin').text),
                                 int(cur_bnd.find('xmax').text),
                                 int(cur_bnd.find('ymax').text)])
            label_ls.append(cur_label)
            bbox_ls.append(cur_bbox.tolist())
        if return_polygon:
            bbox_ls = self.rect2polygon(bbox_ls)
        return label_ls, bbox_ls


class LabelmeParser(BaseOperation):
    
    def __init__(self):
        super(LabelmeParser, self).__init__()
    
    def parse_rect(self, json_file: str, return_polygon=True) -> tuple:
        """parse labelme
        return:
            label(List[str])
            bbox_ls(List[List]): 
        """
        labelme_info = json2dict(json_file)
        shape_info = labelme_info["shapes"]
        label_ls = []
        bbox_ls = []
        for i in shape_info:
            label = i["label"] 
            bbox = i["points"] 
            bbox_ls.append(np.array(bbox).astype(np.uint32).tolist())
            label_ls.append(label)     
        if return_polygon:
            bbox_ls = self.rect2polygon(bbox_ls) 
        return label_ls, bbox_ls 
    

class GenLabelmeJson:

    def __init__(self):
        pass
    
    @staticmethod
    def generate_info(json_info: Dict, img_name: str, img_base64: str, img_h: int, img_w: int):
        json_info["version"] = "4.5.10"
        json_info["flags"] = {}
        json_info["imagePath"] = img_name
        json_info["imageData"] = img_base64
        json_info["imageHeight"] = img_h
        json_info["imageWidth"] = img_w
        return json_info

    @staticmethod
    def get_str_label(label: Union[int, float, str]) -> str:
        if isinstance(label, (int, float)):
            label = repr(label) 
        elif isinstance(label, str):
            return label
        else:
            raise TypeError(f"label must belong to (int, float, str), but received {type(label)}")
        return label

    def __call__(self, img_name, img_base64, loc_ls: Iterable[List[List]], img_h: int, img_w: int, cls_ls: Optional[List[str]]=None) -> Dict:
        json_info = dict()
        json_info = self.generate_info(json_info, img_name, img_base64, img_h, img_w)
        shape_info = []
        if cls_ls is not None:
            assert len(cls_ls) == len(loc_ls)
        for idx, loc in enumerate(loc_ls):
            cur_info = dict()
            if len(loc) == 2:
                cur_info["shape_type"] = "rectangle"
            elif len(loc) == 4:
                cur_info["shape_type"] = "polygon"
            else: 
                raise NotImplementedError(f"loc_ls only support two type: the first one(rectangle) [[xmin, ymin], [xmax, ymax]]"
                                          f"the second(polygon) [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]")
            cur_info["points"] = loc
            cur_info["label"] = 'ok' if cls_ls is None else self.get_str_label(cls_ls[idx])
            cur_info["group_id"] = None
            cur_info["flags"] = {}
            shape_info.append(cur_info)
        json_info["shapes"] = shape_info
        return json_info 
        
    def generate_rect_label_json(self, img_name, img_base64, loc_ls: Iterable[List[List]], img_h: int, img_w: int, cls_ls: Optional[List]=None) -> Dict:
        json_info = dict()
        json_info = self.generate_info(json_info, img_name, img_base64, img_h, img_w)
        shape_info = []
        for idx, loc in enumerate(loc_ls):
            xmin, ymin, xmax, ymax = loc
            cur_info = {}
            if cls_ls is None:
                cur_info["label"] = "ok"
            else:
                cur_info["label"] = repr(int(cls_ls[idx]))
            cur_info["points"] = [[xmin, ymin], [xmax, ymax]]
            cur_info["group_id"] = None
            cur_info["shape_type"] = "rectangle"
            cur_info["flags"] = {}
            shape_info.append(cur_info)
        json_info["shapes"] = shape_info
        return json_info


class GenLabelStudioOCRJson:
    def __init__(self, file_url=''):
        """https://labelstud.io/guide/predictions.html#Example-JSON-3"""
        self.file_url=file_url

    @staticmethod
    def build_instance_label(img_h, img_w, xmin, ymin, xmax, ymax, instance_rot, label, idx, score=0.9999, model_version='', instance_label='Text'):
        width = xmax - xmin 
        height = ymax - ymin 
        instance_result_info = list()
        base_info = dict(
            original_width=img_w, 
            original_height=img_h, 
            image_rotation=0, 
            value=dict(x=(xmin / img_w) * 100, y=(ymin / img_h) * 100, width=(width / img_w) * 100, height=(height / img_h) * 100, rotation=instance_rot))
        bbox_baseinfo = base_info
        bbox_info = dict(**bbox_baseinfo, id=idx, from_name='bbox', to_name="image", type="rectangle")
        
        label_baseinfo = copy.deepcopy(base_info)
        label_baseinfo["value"]["labels"] = [instance_label]
        instance_label_info = dict(**label_baseinfo, id=idx, from_name='label', to_name="image", type="labels")
        
        text_baseinfo = copy.deepcopy(base_info)
        text_baseinfo["value"]["text"] = [label]
        instance_text_info = dict(**text_baseinfo, id=idx, from_name='transcription', to_name="image", type="textarea")
        instance_result_info.append(bbox_info)
        instance_result_info.append(instance_label_info)
        instance_result_info.append(instance_text_info) 
        return instance_result_info

    @staticmethod
    def check_ocr_inputs(item_ls, target_len, default_value):
        if item_ls is None:
            item_ls = [default_value] * target_len
        else:
            assert len(item_ls) == target_len
        return item_ls

    def __call__(self, img_path, loc_ls, img_h, img_w, cls_ls=None, img_rot_ls=None, instance_label_ls=None, score_ls=None, model_version='best-version') -> Dict:
        """
        Args:
            img_path: img http file path
            loc_ls: [[xmin, ymin], [xmax, ymax]] or [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            img_h: image height
            img_w: image width 
            cls_ls: list of predict text 
            img_rot_ls: list of text instance rotation angle
            instance_label_ls: list of text instance type, default type = Text
            score_ls: list of text instance prediction score (not using now)
            model_version: model info
        Return:
            json_info
        """
        
        json_info = dict()
        json_info["data"] = dict(ocr=self.file_url + img_path)
        json_info["predictions"] = list() 
        tgt_len = len(loc_ls)
        cls_ls = self.check_ocr_inputs(cls_ls, tgt_len, 'null')
        img_rot_ls = self.check_ocr_inputs(img_rot_ls, tgt_len, 0)
        score_ls = self.check_ocr_inputs(score_ls, tgt_len, 0.0000) 
        instance_label_ls = self.check_ocr_inputs(instance_label_ls, tgt_len, 'Text') 
        
        instance_info = dict() 
        instance_info["model_version"] = model_version
        instance_info["result"] = list() 

        for idx, (loc, label, instance_rot, instance_label, score) in enumerate(zip(loc_ls, cls_ls, img_rot_ls, instance_label_ls, score_ls)):
            if len(loc) == 2:
                xmin, ymin = loc[0][:2] 
                xmax, ymax = loc[1][:2] 
                assert xmax >= xmin, ymax >= ymin 
            elif len(loc) == 4:
                xmin, ymin = loc[0][:2]
                xmax, ymax = loc[2][:2] 
                assert xmax >= xmin, ymax >= ymin 
            else:
                raise ValueError 
            idx = str(uuid4())[:10]
            cur_result_info = self.build_instance_label(img_h, img_w, xmin, ymin, xmax, ymax, instance_rot, label, idx, score=score, model_version=model_version, instance_label=instance_label)
            instance_info["result"].extend(cur_result_info)
        json_info["predictions"].append(instance_info)
        return json_info
