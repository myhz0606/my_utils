"""
@author: wwjiang
@time: 2022/6/1: 17:35
"""

import os 
import sys
import cv2
import numpy as np 
from numpy import ndarray 
from PIL import Image, ImageFont, ImageDraw
from typing import List, Dict, Sequence, Tuple, Optional, Union

COLOR_BAR = ['#0000FF', '#04B431', '#FF7F50', '#FF3838', '#FF6347', '#CB38FF', '#0018EC', '#C76114', '#FF9D97', '#03A89E', '#FF701F', '#FFB21D', '#00C78C','#B0171F', '#CFD231', '#48F90A', '#92CC17', '#3DDB86', '#1A9334', '#00D4BB',
             '#2C99A8', '#00C2FF', '#344593', '#6473FF', '#F0FFFF', '#FF37C7', '#8438FF', '#520085', '#FF95C8']

curdir = os.path.dirname(__file__)
sys.path.append(curdir) 


class DrawImage:
    
    def __init__(self):
        pass 

    def rectangles(self, img: ndarray, boxes: Union[List, Tuple, ndarray], thickness: int=2) -> Optional[ndarray]:
        """
        Args:
            boxes: [[x1, y1, x2, y2], ...] or [[x1, y1, x2, y2, x3, y3, x4, y4], ...]
        """
        def check_boxes(boxes: Union[List, Tuple, ndarray]) -> Union[List, Tuple, None]:
            if isinstance(boxes, ndarray):
                boxes = boxes.tolist() 
            if not isinstance(boxes[0], (List, Tuple)):
                boxes = [boxes]
            if len(boxes[0]) == 8:  
                boxes_new = [] 
                for box in boxes:
                    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
                    boxes_new.append([x1, y1, x3, y3]) 
                return boxes_new 
            assert len(boxes[0]) == 4
            return boxes 
        img = img.copy()
        img = self.check_img(img) 
        boxes = check_boxes(boxes)

        for idx, box in enumerate(boxes):
            img = cv2.rectangle(img, box[:2], box[2:], color=self.hex2rgb(COLOR_BAR[idx % len(COLOR_BAR)]), thickness=thickness)
        return img 

    def circles(self, img: ndarray, pts: Union[List, Tuple, ndarray], radius: int=6, thickness: int=2, default_color=COLOR_BAR[0], circulate_color=False) -> Optional[ndarray]:
        """
        draw some circles in the images.
        Args:
            pts[List, Tuple, ndarray]: center of circle. [(x1, y1), (x2, y2)...]
        """
        if isinstance(default_color, str):
            default_color = self.hex2rgb(default_color)
        img = img.copy()
        img = self.check_img(img)  
        if isinstance(pts, ndarray):
            pts = pts.tolist() 
        if not isinstance(pts[0], (list, tuple)):  # multiple circles
            pts = [pts]
        for idx, pt in enumerate(pts):
            if circulate_color:
                cur_color = self.hex2rgb(COLOR_BAR[idx % len(COLOR_BAR)])
            else:
                cur_color = default_color
            img = cv2.circle(img=img, center=pt, radius=radius, thickness=thickness, color=cur_color)
        return img  

    def line(self, img, points: Union[List[List], np.ndarray], thickness=2, default_color=(0, 0, 255), circulate_color=False, with_circle=True, solid_circle=True):
        """
        points:
            if np.ndarray, dim: N * 4 (start_x, start_y, end_x, end_y)
            if List[List[float or int]], each item [tart_x, start_y, end_x, end_y]
        """

        def check_points(points):
            if isinstance(points, list):
                points = np.array(points).reshape(-1, 4)
            elif isinstance(points, np.ndarray):
                assert points.shape[1] == 4 and points.ndim == 2, f"but received: {points.shape}"
            else:
                raise TypeError(f"points only support list or ndarray, but received {type(points)}")
            return points.tolist() 
        img = self.check_img(img) 
        img = img.copy()
        points_ls = check_points(points=points)
        for idx, point in enumerate(points_ls):
            assert point.__len__() == 4, f"point only have for coordinate, but received {point}"
            x1, y1, x2, y2 = point[:4]
            if circulate_color:
                cur_color = self.hex2rgb(COLOR_BAR[idx % len(COLOR_BAR)])
            else:
                cur_color = default_color
            img = cv2.line(img, (x1, y1), (x2, y2), color=cur_color, thickness=thickness)

            if with_circle:
                circle_color = (np.array(cur_color) * 0.8 + np.array((0, 0, 0)) * 0.2).astype(np.uint8).tolist()
                img = self.circles(img, [x1, y1], radius=int(thickness*2), thickness=-1 if solid_circle else thickness, default_color=circle_color)
                img = self.circles(img, [x2, y2], radius=int(thickness*2), thickness=-1 if solid_circle else thickness, default_color=circle_color)
        return img 

    def rectangle_with_annotation(self, 
            img: ndarray, 
            boxes: Union[List, Tuple, ndarray], 
            labels: Union[List, Tuple, ndarray], 
            thickness: int=2, 
            font_thickness: int=1,
            ) -> Optional[ndarray]:
        """draw rectangle with annotations"""

        def check_boxes(boxes: Union[List, Tuple, ndarray]) -> Union[List, Tuple, None]:
            if isinstance(boxes, ndarray):
                boxes = boxes.tolist() 
            if not isinstance(boxes[0], (List, Tuple)):
                boxes = [boxes]
            if len(boxes[0]) == 8:  
                boxes_new = [] 
                for box in boxes:
                    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
                    boxes_new.append([x1, y1, x3, y3]) 
                return boxes_new 
            assert len(boxes[0]) == 4 
            return boxes 
        
        def check_labels(labels: Union[List, Tuple, ndarray]) -> Union[List, Tuple, None]:
            assert isinstance(labels, (List, Tuple, ndarray)) 
            if isinstance(labels, ndarray):
                labels = labels.tolist() 
            return labels 
        img = img.copy()
        img = self.check_img(img) 
        boxes = check_boxes(boxes) 
        labels = check_labels(labels) 

        for idx, (box, label) in enumerate(zip(boxes, labels)):
            cur_color = self.hex2rgb(COLOR_BAR[idx % len(COLOR_BAR)])
            img = cv2.rectangle(img, box[:2], box[2:], color=cur_color, thickness=thickness)

            f_w, f_h = cv2.getTextSize(label, 0, fontScale=font_thickness / 3, thickness=font_thickness)[0]
            if box[0] - f_h < 0:  # overflow upper limit 
                pt1 = (box[0], box[3]) 
                pt2 = (box[0] + f_w, box[3] + f_h + 3)
                font_pt = (pt1[0], pt1[1] + 5)
            else:
                pt1 = (box[0], box[1] - f_h - 3) 
                pt2 = (box[0] + f_w, box[1])
                font_pt = (pt1[0], pt1[1] - 1)

            img = cv2.rectangle(img, pt1, pt2, color=cur_color, thickness=-1) 
            img = cv2.putText(img, label, font_pt, fontFace=0, fontScale=font_thickness / 3, thickness=font_thickness, color=(255, 255, 255) if sum(cur_color) / 3 < 100 else (0, 0, 0))
        return img 

    def rectangle_with_annotation_new(self, 
        img: ndarray, 
        boxes: List[List], 
        labels: List[str],
        font_path: Optional[str]=None,
        font_size: int=16,        
        thickness: int=2, 
        ) -> Optional[ndarray]:
        """draw rectangle with annotations
        Args:
            img(ndarray)
            boxes(List[List]): [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...]
            labels(List[str]): [str1, str2,...]
        """
        assert isinstance(boxes, List)
        assert isinstance(boxes[0], List)
        assert isinstance(labels, Sequence)
        assert len(boxes) == len(labels)
        img = img.copy()
        img = self.check_img(img) 
        anno_info = []
        
        for idx, (box, label) in enumerate(zip(boxes, labels)):
            cur_color = self.hex2rgb(COLOR_BAR[idx % len(COLOR_BAR)])
            img = cv2.rectangle(img, box[:2], box[2:], color=cur_color, thickness=thickness)
            anno_info.append([box[0], box[1], label, cur_color, self.rect_vertical_classifier(box)])   

        if font_path is None:
            cur_font = ImageFont.truetype(os.path.join(curdir, 'Deng.ttf'), font_size)
        else:
            cur_font = ImageFont.truetype(font_path, font_size)
        
        img_pil = Image.fromarray(img)
        for idx, (lb_x, lb_y, label, color, is_vertical) in enumerate(anno_info):
            self.__draw_text(img_pil, int(is_vertical), label, [lb_x, lb_y], cur_font, color=color)
        return np.array(img_pil)  

    def poly_with_annotation(self, 
            img: ndarray, 
            boxes: Union[List, Tuple, ndarray], 
            labels: Union[List, Tuple, ndarray], 
            thickness: int=2, 
            font_thickness: int=1,
            ) -> Optional[ndarray]:
        """draw rectangle with annotations
        boxes:
            List: [_DIM(4, 2), _DIM(4, 2), ...] ; _DIM(4, 2) info: [(x1, y1), (x2, y2), ...]
        """

        def check_boxes(boxes: Union[List, Tuple, ndarray]) -> Union[List, Tuple, None]:
            if isinstance(boxes, ndarray):
                boxes = boxes.tolist() 
            if not isinstance(boxes[0], (List, Tuple)):
                boxes = [boxes]
            # assert len(boxes[0]) == 8
            return boxes 
        
        def check_labels(labels: Union[List, Tuple, ndarray]) -> Union[List, Tuple, None]:
            assert isinstance(labels, (List, Tuple, ndarray)) 
            if isinstance(labels, ndarray):
                labels = labels.tolist() 
            if not isinstance(labels[0], (List, Tuple)):
                return labels 
        img = img.copy()
        img = self.check_img(img) 
        boxes = check_boxes(boxes) 
        labels = check_labels(labels) 

        for idx, (box, label) in enumerate(zip(boxes, labels)):
            cur_color = self.hex2rgb(COLOR_BAR[idx % len(COLOR_BAR)])
           
            cur_box = (np.array(box).astype(np.uint32)).reshape(-1, 1, 2)
            print(f"boxes info {box}, {cur_box.shape}")
            img = cv2.polylines(img, [], True, color=cur_color, thickness=thickness)

            f_w, f_h = cv2.getTextSize(label, 0, fontScale=font_thickness / 3, thickness=font_thickness)[0]
            if box[0][0] - f_h < 0:  # overflow upper limit 
                pt1 = (box[0][0], box[0][1]) 
                pt2 = (box[0][0] + f_w, box[0][1] + f_h + 3)
                font_pt = (pt1[0], pt1[1] + 5)
            else:
                pt1 = (box[0][0], box[0][1] - f_h - 3) 
                pt2 = (box[0][0] + f_w, box[0][1])
                font_pt = (pt1[0], pt1[1] - 1)

            img = cv2.rectangle(img, pt1, pt2, color=cur_color, thickness=-1) 
            img = cv2.putText(img, label, font_pt, fontFace=0, fontScale=font_thickness / 3, thickness=font_thickness, color=(255, 255, 255) if sum(cur_color) / 3 < 100 else (0, 0, 0))
        return img 

    @classmethod
    def poly_with_annotation_new(cls, 
        img: ndarray, 
        boxes: Union[List, Tuple, ndarray], 
        labels: Union[List, Tuple, ndarray], 
        font_path: Optional[str]=None,
        font_size: int=16,
        thickness: int=2,               
        font_thickness: int=1,
        ) -> Optional[ndarray]:
        """draw rectangle with annotations
        boxes:
            when List: [_DIM(4, 2), _DIM(4, 2), ...] ; _DIM(4, 2) info: [(x1, y1), (x2, y2), ...]
            when ndarray: _DIM(N, 4, 2)
        """
        
        def check_boxes(boxes: Union[List, Tuple, ndarray]) -> Union[List, Tuple, None]:
            if isinstance(boxes, ndarray):
                boxes = boxes.tolist() 
            assert isinstance(boxes, (List, Tuple))
            return boxes 

        def check_labels(labels: Union[List, Tuple, ndarray]) -> Union[List, Tuple, None]:
            assert isinstance(labels, (List, Tuple, ndarray)) 
            if isinstance(labels, ndarray):
                labels = labels.tolist() 
            return labels 
        img = img.copy()
        img = cls.check_img(img) 
        boxes = check_boxes(boxes) 
        labels = check_labels(labels) 
        anno_info = []

        for idx, (box, label) in enumerate(zip(boxes, labels)):
            cur_color = cls.hex2rgb(COLOR_BAR[idx % len(COLOR_BAR)])
            cur_box = np.array(box).reshape(-1, 1, 2).astype(np.int32)
            img = cv2.polylines(img, [cur_box], True, color=cur_color, thickness=thickness)
            anno_info.append([cur_box[0][0][0], cur_box[0][0][1], label, cur_color, cls.poly_vertical_classifier(cur_box)])   
            
        if font_path is None:
            cur_font = ImageFont.truetype(os.path.join(curdir, 'Deng.ttf'), font_size)
        else:
            cur_font = ImageFont.truetype(font_path, font_size)
        
        img_pil = Image.fromarray(img)
        for lb_x, lb_y, label, color, is_vertical in anno_info:
            cls.__draw_text(img_pil, direction=int(is_vertical), text=label, start_pos=[lb_x, lb_y], font=cur_font, color=color)
        return np.array(img_pil) 
    
    @classmethod
    def draw_vertical_text(cls, img_pil: Image.Image, text: str, start_pos: Sequence, font: ImageFont, draw_outline=True, color=(0, 0, 255)):
        cls.__draw_text(img_pil, direction=1, text=text, start_pos=start_pos, font=font, draw_outline=draw_outline, color=color)

    @classmethod
    def draw_horizonal_text(cls, img_pil: Image.Image, text: str, start_pos: Sequence, font: ImageFont, draw_outline=True, color=(0, 0, 255)):
        cls.__draw_text(img_pil, direction=0, text=text, start_pos=start_pos, font=font, draw_outline=draw_outline, color=color)
        
    @staticmethod
    def __draw_text(img_pil: Image.Image, direction: int, text: str, start_pos: Sequence, font: ImageFont, draw_outline=True, color=(0, 0, 255)):
        """ draw text in canvas
        direction(int):
            0: draw horizontal text
            1: draw vertical text
        """
        draw = ImageDraw.Draw(img_pil) 
        img_size = img_pil.size
        char_wh_info = [font.getsize(i)[:2] for i in text]
        offset = max(char_wh_info, key=lambda x: x[1 - direction])[1 - direction]
        start_pos[1 - direction] = max(start_pos[1 - direction] - offset, 0)
        
        if direction:
            rect = [
                max(start_pos[0] - 1, 0), 
                start_pos[1], 
                min(offset + start_pos[0], img_size[0]), 
                min(start_pos[1] + sum([i[1] for i in char_wh_info]), img_size[1])
                ]
        else:
            rect = [
                start_pos[0], 
                max(start_pos[1] - 1, 0), 
                min(start_pos[0] + sum([i[0] for i in char_wh_info]), img_size[0]), 
                min(offset + start_pos[1], img_size[1])
                ]
        rect_color = (np.array([255, 255, 255]) * 0.4 + np.array(color) * 0.6).astype(np.uint8)
        if draw_outline:
            draw.rectangle(rect, fill=tuple(rect_color.tolist()))
        for idx, cur_char in enumerate(text):
            draw.text(start_pos, cur_char, fill=tuple([255 - i for i in color]), font=font) 
            start_pos[direction] += char_wh_info[idx][direction]  
            if start_pos[direction] > img_size[direction]:
                start_pos[direction] = img_size[direction] 
                break 
        rect.extend([start_pos[0], start_pos[1]])

    @classmethod
    def rect_vertical_classifier(cls, bbox: List, tolerate=2.5) -> bool:
        x1, y1, x2, y2 = bbox[:4]
        h = abs(y2 - y1)
        w = abs(x2 - x1)
        return cls.__vertical_classifier(h / w, tolerate=tolerate)
    
    @classmethod
    def poly_vertical_classifier(cls, bbox: np.ndarray, tolerate=2.5) -> bool:
        """bbox shape like (N, 1, 2)"""
        bbox = bbox.squeeze(axis=1) 
        print(bbox)
        xmin, xmax = min(bbox[:, 0]), max(bbox[:, 0])
        ymin, ymax = min(bbox[:, 1]), max(bbox[:, 1])
        h, w = abs(ymax - ymin), abs(xmax - xmin) 
        return cls.__vertical_classifier(h / w, tolerate=tolerate) 
    
    @staticmethod
    def __vertical_classifier(hw_ratio, tolerate=2.5):
        if hw_ratio > tolerate:
            return True 
        else:
            return False
    
    @staticmethod
    def check_img(img: Union[ndarray, Image.Image]) -> Optional[ndarray]: 
        if isinstance(img, Image.Image):
            img = np.array(img) 
        assert isinstance(img, ndarray)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=2) 
        assert img.ndim == 3
        return img 
    
    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def hex2bgr(h):  
        h = h[1:]  # remove #
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (4, 2, 0))


if __name__ == "__main__":
    import numpy as np 
    canvas = np.ones((512, 512, 3), dtype=np.uint8) * 255 
    bbox = [
        np.array([[20, 20], [30, 20], [30, 40], [20, 40]]), 
        np.array([[50, 50], [200, 50], [200, 80], [50, 80]])]
    # bbox = [[20, 20, 30, 40], [50, 50, 200, 80]]
    label = ["vertical text", "horizon"]
    draw_obj = DrawImage()
    res = draw_obj.poly_with_annotation_new(canvas, bbox, label, thickness=2)
    # res = draw_obj.rectangle_with_annotation_new(canvas, bbox, label)
    Image.fromarray(res).show()