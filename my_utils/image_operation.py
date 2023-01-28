"""
@author: wwjiang
@time: 2021/8/17: 14:35
"""

import os 
import math 
import traceback
import requests
from io import BytesIO
from typing import List, Dict, Tuple, Union, Optional
import numpy as np
import cv2
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger()

__all__ = [
    "resize_img", 
    "add_blank_edge_to_image", 
    "add_edge_to_image", 
    "remove_image_blank_edge", 
    "read_img", 
    "show_img", 
    "show_img_ls", 
    "ResizeImg", 
    "ImageDeocde", 
    ]


def resize_img(img: np.ndarray,
               resize_shape: Tuple,
               backend: str = "pil",
               padding: bool = False,
               padding_color: Union[str, Tuple] = "blank") -> np.ndarray:

    def padding_square_img(img, padding_color: Union[str, Tuple] = "blank"):
        ori_h, ori_w = img.shape[:2]
        aim_size = max(ori_h, ori_w)
        top = (aim_size - ori_h) // 2
        bottom = aim_size - ori_h - top
        left = (aim_size - ori_w) // 2
        right = aim_size - ori_w - left
        padding_img = add_edge_to_image(img, [top, bottom, left, right], color=padding_color)
        return padding_img

    if padding:
        img = padding_square_img(img, padding_color=padding_color)

    if backend == "pil":
        return np.array(Image.fromarray(img).resize(resize_shape))
    elif backend == "cv2":
        return cv2.resize(img, resize_shape)
    else:
        raise NotImplementedError(f"resize backend only support [pil, cv2]")


def add_blank_edge_to_image(img: np.ndarray, scope: List = [0, 0, 0, 0]) -> np.ndarray:
    """
    Args:
        img: np.ndarray, input image
        scope: List, 上下左右的偏移量
    Return:
        new_image: np.ndarray, 加了白边的图片输出
    """
    ori_h, ori_w = img.shape[:2]
    top, bottom, left, right = scope[:4]
    new_w, new_h = ori_w + left + right, ori_h + top + bottom
    new_image = np.ones((new_h, new_w), dtype=img.dtype) * 255
    assert img.ndim <= 3
    if img.ndim == 3:
        new_image = np.stack([new_image]*img.shape[2], axis=2)

    new_image[top: top+ori_h, left: left+ori_w, ...] = img[:]
    return new_image


def add_edge_to_image(img: np.ndarray,
                      scope: List = [0, 0, 0, 0],
                      color: Union[str, Tuple] = "blank") -> np.ndarray:
    """
    Args:
        img: np.ndarray, input image
        scope: List, 上下左右的偏移量
        color: string, edge color
    Return:
        new_image: np.ndarray, 加了白边的图片输出
    """
    assert img.ndim == 3 and img.shape[-1] == 3, f"received: img dim {img.ndim}, img shape {img.shape}"

    color_map = {
        "blank": (255, 255, 255),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 2),
        "black": (0, 0, 0)
    }
    if isinstance(color, str):
        assert color in color_map, f"received: color {color}"
        color_pixel = color_map[color]
    elif isinstance(color, Tuple):
        color_pixel = color
    else:
        raise TypeError(f"color must be Tuple or str!")
    ori_h, ori_w = img.shape[:2]
    top, bottom, left, right = scope[:4]
    new_w, new_h = ori_w + left + right, ori_h + top + bottom
    new_image = np.ones((new_h, new_w, 3), dtype=img.dtype)
    new_image[:] = color_pixel
    new_image[top: top+ori_h, left: left+ori_w, ...] = img[:]

    return new_image


def remove_image_blank_edge(img: np.ndarray, return_coord=False) \
        -> np.ndarray or Tuple[np.ndarray, List[int, int, int, int]]:
    """
    Args:
        img:
        return_coord: bool: whether return clip coordinate
    Returns:
        if return_coord:
            img_remove_blank: image without edge
        else:
            img_remove_blank: image without edge
            clip_loc: [xmin, ymin, xmax, ymax]

    """
    def get_blank_idx(pixel_sum: List, thresh: int=1) -> Tuple[int, int]:
        index_info = [idx for idx, i in enumerate(pixel_sum) if i >= thresh]
        if index_info.__len__() < 2:
            start = 0
            end = len(pixel_sum)
        else:
            start = index_info[0]
            end = index_info[-1]
        return start, end

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_canny = cv2.Canny(img_gray, 50, 150)  # 0为背景, 255为edge
    # return img_canny
    img_canny[img_canny[:] < 128] = 0
    img_canny[img_canny[:] >= 128] = 1

    row_sum = list(np.sum(img_canny, axis=0))  # 行和
    col_sum = list(np.sum(img_canny, axis=1))  # 列和

    row_start, row_end = get_blank_idx(col_sum)
    col_start, col_end = get_blank_idx(row_sum)

    img_remove_blank = img[row_start: row_end, col_start: col_end, ...]
    if not return_coord:
        return img_remove_blank
    else:
        return img_remove_blank, [col_start, row_start, col_end, row_end]


def read_img(img_path: str, debug=False, logger=logger) -> np.ndarray:
    """given an image path, return image ndarray with RGB channel, vecolity slow !!! """

    def rgba2rgb(img: np.ndarray) -> np.ndarray:
        assert img.shape[-1] == 4
        h, w = img.shape[:2]
        unique_a = np.unique(img[..., 3])
        if unique_a.shape[0] == 1:
            return img[..., :3]
        r, g, b, a = [img[..., i] for i in range(4)]
        a = a / 255.0
        new_rgb = np.zeros((h, w, 3), dtype=img.dtype)
        new_rgb[..., 0] = r * a + (1 - a) * 255
        new_rgb[..., 1] = g * a + (1 - a) * 255
        new_rgb[..., 2] = b * a + (1 - a) * 255
        return np.asarray(new_rgb, dtype=img.dtype)

    def gray2rgb(img: np.ndarray) -> np.ndarray:
        assert img.ndim == 2
        return np.stack([img] * 3, axis=2)
    
    img_pil = Image.open(img_path) 
    img_mode = img_pil.mode 
    if img_mode not in ["RGBA", "RGB"]:  # 有些图mode为‘1’， ‘p’不转化直接变为ndarray会出问题
        img_pil = img_pil.convert("RGB")

    img = np.array(img_pil)
    if debug:
        logger.info(f"image ndim is {img.ndim}")
    if img.ndim == 3:
        if img.shape[-1] == 4:
            if debug:
                logger.info(f"img.shape[-1]==4, go into rgba2rgb()")    
            img = rgba2rgb(img)
        elif img.shape[-1] == 3:
            if debug:
                logger.info(f"img.shape[-1]==3, go pass")    
            pass
        else:
            if debug:
                logger.info(f"img.shape[-1] < 3")    
            img = np.stack([img[..., 0]] * 3, axis=2)
    elif img.ndim == 2:
        logger.info(f"img.ndim==2, go into gray2rgb()")    
        img = gray2rgb(img)
    else:
        raise IOError(f"receive image shape is {img.shape} img path: {img_path}")
    return img  # bgr2rgb


def show_img(img: np.array, backend="matplotlib"):

    if backend == "matplotlib":
        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.show()

    elif backend == "opencv":
        import cv2
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif backend == "PIL":
        Image.fromarray(img).show()


def show_img_ls(img_ls, img_shape, max_width=1024, border=False) -> np.ndarray:

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def trans_img(img_ls, img_shape, border=False):
        img_ls_new = []
        for img in img_ls:
            img_new = cv2.resize(img, img_shape[:2][::-1])
            if img_new.ndim == 2:
                img_new = np.stack([img_new] * 3, axis=2)
            if border:
                img_new[:1, ...] = 128
                img_new[-2:, ...] = 128
                img_new[:, :1, ...] = 128
                img_new[:, -2:, ...] = 128
            img_ls_new.append(img_new)
        return img_ls_new

    h, w = img_shape[:2]
    img_num = len(img_ls)
    img_per_row = max_width // w
    pad_img_ls = [np.zeros((h, w, 3), dtype=np.uint8)] * int(np.ceil(img_num / img_per_row) * img_per_row - img_num)
    img_ls = img_ls + pad_img_ls
    img_ls_new = trans_img(img_ls, img_shape, border=border)

    img_chunk = list(chunks(img_ls_new, img_per_row))
    img_show = np.concatenate([np.concatenate(i, axis=1) for i in img_chunk], axis=0)
    return img_show


class ResizeImg:
    def __init__(self, **kwargs):
        """
        Example:
        >>>img = (np.random.rand(512, 768, 3) * 255).astype(np.uint8)
        >>>resize_params = [
            dict(params=dict(limit_long=1024, fixed_factor=16), output_shape=(512, 768, 3)),
            dict(params=dict(fixed_long=1024, fixed_factor=16), output_shape=(688, 1024, 3)),
            dict(params=dict(limit_long=1024, fixed_factor=16, min_size=600), output_shape=(608, 768, 3)),
            dict(params=dict(limit_short=600, fixed_factor=16), output_shape=(512, 768, 3)),
            dict(params=dict(fixed_short=600, fixed_factor=16), output_shape=(608, 912, 3)),
            dict(params=dict(limit_short=600, fixed_factor=16, max_size=700), output_shape=(608, 704, 3)),
            dict(params=dict(fixed_shape=(320, 632), fixed_factor=16), output_shape=(320, 640, 3)),
            dict(params=dict(fixed_shape=(320, 632), fixed_factor=16, min_size=350), output_shape=(352, 640, 3)),
        ]

        kwargs:
            ###########5种resize模式###########
            fixed_shape: 固定尺寸resize
            fixed_short: 最短边固定
            fixed_long: 最长边固定
            limit_short: 最短边不超过
            limit_long: 最长边不超过
            ###########可选参数###########
            fixed_factor: resize尺寸需要包含的因数（确保尺寸能被该数整除，向上取整）
            min_size: 图片最小尺寸不低于
            max_size： 图片最长尺寸不高于
            ###########默认参数###########
        """

        super(ResizeImg, self).__init__()
        self.min_size = kwargs.get("min_size", None)
        self.max_size = kwargs.get("max_size", None)
        self.fixed_factor = kwargs.get("fixed_factor", 1)
        self.fixed_len = 0
        self.reshape_size = None

        if "fixed_shape" in kwargs:
            self.reshape_size = kwargs['fixed_shape']
            self.resize_func = self.resize_image

        elif "limit_short" in kwargs:
            self.resize_func = self.resize_image_by_limit_short
            self.fixed_len = kwargs.get("limit_short")

        elif "limit_long" in kwargs:
            self.resize_func = self.resize_image_by_limit_long
            self.fixed_len = kwargs.get("limit_long")
        
        elif "fixed_short" in kwargs:
            self.resize_func = self.resize_image_by_fixed_short
            self.fixed_len = kwargs.get("fixed_short")

        elif "fixed_long" in kwargs:
            self.resize_func = self.resize_image_by_fixed_long
            self.fixed_len = kwargs.get("fixed_long")

        else:
            self.resize_func = self.resize_image_by_adaptive

    def __call__(self, img) -> np.ndarray:
        ori_h, ori_w = img.shape[:2]
        resize_params = dict()
        resize_params["img"] = img
        resize_params["fixed_len"] = self.fixed_len
        resize_params["reshape_size"] = self.reshape_size
        resize_params["fixed_factor"] = self.fixed_factor
        resize_params["min_size"] = self.min_size
        resize_params["max_size"] = self.max_size
        resize_img, [ratio_h, ratio_w] = self.resize_func(**resize_params)
        return resize_img

    def resize_image_by_fixed_short(self, img: np.ndarray, fixed_len: int, **kwargs):
        """最短边固定为fixed_len。若fix_factor不为1，则将最短边固定为 round(fixed_len / fixed_factor) * fixed_len
        """
        h, w = img.shape[:2]
        if h < w:
            # h 为短边
            resize_h = fixed_len
            resize_w = fixed_len * float(w) / h
        else:
            resize_w = fixed_len
            resize_h = fixed_len * float(h) / w
        reshape_size = (resize_h, resize_w)
        del kwargs["reshape_size"]
        return self.resize_image(img, reshape_size=reshape_size, **kwargs)

    def resize_image_by_fixed_long(self, img: np.ndarray, fixed_len: int, **kwargs):
        """最长边固定为fixed_len。若fix_factor不为1，则将最短边固定为 round(fixed_len / fixed_factor) * fixed_len
        """
        h, w = img.shape[:2]
        if h > w:
            # h 为长边
            resize_h = fixed_len
            resize_w = fixed_len * float(w) / h
        else:
            resize_w = fixed_len
            resize_h = fixed_len * float(h) / w
        reshape_size = (resize_h, resize_w)
        del kwargs["reshape_size"]
        return self.resize_image(img, reshape_size=reshape_size, **kwargs)
    
    def resize_image_by_limit_short(self, img: np.ndarray, fixed_len: int, **kwargs):
        """最短边固定为fixed_len。若fix_factor不为1，则将最短边固定为 round(fixed_len / fixed_factor) * fixed_len
        """
        h, w = img.shape[:2]
        if h < w:
            # h 为短边
            resize_h = fixed_len if h > fixed_len else h
            resize_w = resize_h * float(w) / h
        else:
            resize_w = fixed_len if w > fixed_len else w
            resize_h = resize_w * float(h) / w
        reshape_size = (resize_h, resize_w)
        del kwargs["reshape_size"]
        return self.resize_image(img, reshape_size=reshape_size, **kwargs)

    def resize_image_by_limit_long(self, img: np.ndarray, fixed_len: int, **kwargs):
        """最长边固定为fixed_len。若fix_factor不为1，则将最短边固定为 round(fixed_len / fixed_factor) * fixed_len
        """
        h, w = img.shape[:2]
        if h > w:
            # h 为长边
            resize_h = fixed_len if h > fixed_len else h
            resize_w = resize_h * float(w) / h
        else:
            resize_w = fixed_len if w > fixed_len else w
            resize_h = resize_w * float(h) / w
        reshape_size = (resize_h, resize_w)
        del kwargs["reshape_size"]
        return self.resize_image(img, reshape_size=reshape_size, **kwargs)

    def resize_image_by_adaptive(self, img: np.ndarray, fixed_len: int, **kwargs):
        """根据当前尺寸基于固定因数resize
        """
        h, w = img.shape[:2]
        reshape_size = (h, w)
        del kwargs["reshape_size"]
        return self.resize_image(img, reshape_size=reshape_size, **kwargs)
    
    @staticmethod
    def resize_image(img: np.ndarray, reshape_size: Tuple, fixed_factor: int = 1, **kwargs) -> Tuple:
        img_dtype = img.dtype
        ori_h, ori_w = img.shape[:2]
        resize_h, resize_w, *_ = reshape_size
        resize_h = int(math.ceil(max(resize_h / fixed_factor, 1)) * fixed_factor)
        resize_w = int(math.ceil(max(resize_w / fixed_factor, 1)) * fixed_factor)

        min_size = kwargs["min_size"]
        max_size = kwargs["max_size"]
        if isinstance(min_size, (int, float)) and min_size > 0:
            min_size = int(math.ceil(max(min_size / fixed_factor, 1)) * fixed_factor)
            resize_h = max(min_size, resize_h) 
            resize_w = max(min_size, resize_w)
        if isinstance(max_size, (int, float)) and max_size > 0:
            max_size = int(math.ceil(max(max_size / fixed_factor, 1)) * fixed_factor)
            resize_h = min(max_size, resize_h)
            resize_w = min(max_size, resize_w)
        assert isinstance(resize_h, int) and isinstance(resize_w, int)
        # print(resize_w, resize_h)
        resize_img = cv2.resize(img, (resize_w, resize_h))
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        return resize_img.astype(img_dtype), [ratio_h, ratio_w]


class ImageDeocde:
    
    def __init__(self, return_mode="bgr"):
        """
        Args:
            return mode: 
                RGB, return image format is `RGB`
                BGR, return image format is `BGR` 
        parser method:
            open_url_to_ndarray
            s3_path_to_ndarray
            local_path_to_ndarray
            bytes_to_ndarray

        """
        assert return_mode.lower() in ("bgr", "rgb")
        self.return_mode=return_mode

    def open_url_to_ndarray(self, img_url, logger=logger) -> np.ndarray:
        get_request = requests.get(img_url)
        get_request.raise_for_status()
        bytes_data = get_request.content
        img_bgr = self.imageIOcv2(bytes_data, logger=logger)
        return img_bgr if self.return_mode.lower() == "bgr" else img_bgr[..., ::-1]

    def s3_path_to_ndarray(self, img_url, logger=logger, endpoint_url=None) -> Optional[np.ndarray]:
        """
        loading s3 image path to ndarray
        Args:
            img_url (Union[str, dict]): when str: {bucket}[SEP]{storage_path}; when dict, key1: bucket, key2: storage_path
        Return:
            ndarray
        """
        import boto3 
        import botocore
        Session = boto3.Session(
        aws_access_key_id=os.getenv('AWS_KEY_ID'), 
        aws_secret_access_key=os.getenv('AWS_SECRET_KEY'), 
        region_name=os.getenv('AWS_REGION')
        )
        S3Client = Session.client("s3", endpoint_url=endpoint_url)
        
        try:
            bucket, storage_path = img_url.split("[SEP]")
            assert len(bucket) > 0 and len(storage_path) > 0 
        except:
            bucket = img_url.get("bucket", None)
            storage_path = img_url.get("storage_path", None)
        # logger.info(f"Decode image from s3 path:  BUCKET: {bucket} STORAGE_PATH: {storage_path}")
        try:
            f = BytesIO()
            S3Client.download_fileobj(bucket, storage_path, f)
            f.seek(0)
            image_byte_data = f.getvalue()
        except botocore.exceptions.ClientError as error:
            logger.error(f"Decode image error found {error}")
            logger.error(f"{traceback.print_exc()}")
            return None 
        img = self.bytes_to_ndarray(image_byte_data, logger=logger)
        return img

    def local_path_to_ndarray(self, img_url, logger=logger):
        img_bgr = cv2.imread(img_url)
        return img_bgr if self.return_mode.lower() == "bgr" else img_bgr[..., ::-1]

    def bytes_to_ndarray(self, bytes_data, logger=logger):
        if self.isgif(bytes_data):
            img_bgr = self.imageIOpil(bytes_data, logger=logger) 
        else:
            img_bgr = self.imageIOcv2(bytes_data, logger=logger)
        return img_bgr if self.return_mode.lower() == "bgr" else img_bgr[..., ::-1]
    
    def identity(self, img_bgr: np.ndarray, logger=None) -> np.ndarray:
        return img_bgr

    @staticmethod
    def isgif(h) -> bool:
        """GIF ('87 and '89 variants)"""
        if h[:6] in (b"GIF87a", b"GIF89a"):
            return True
        else:
            return False

    @staticmethod
    def imageIOcv2(bytes_data, logger) -> Optional[np.ndarray]:
        """return BGR image"""
        try:
            img_bgr = cv2.imdecode(
                np.asarray(bytearray(bytes_data), dtype=np.uint8), cv2.IMREAD_COLOR
            )
        except Exception as e:
            logger.warning("Error in decoding byte data into image: {}".format(e))
            return None

        if img_bgr is None:
            logger.warning("Error in decoding byte data into image")
            return None
        else:
            return img_bgr

    @staticmethod
    def imageIOpil(bytes_data, logger):
        """
        parse image from bytes to Image with bgr channel
        """
        stream = BytesIO(bytes_data)
        try:
            image_pil = Image.open(stream).convert("RGB")
        except Exception as e:
            logger.warning("PIL parse byte data to RGB image error: {}".format(e))
            try:
                image_pil_L = Image.open(stream).convert("L")
                image_pil = image_pil_L.convert("RGB")
            except Exception as e:
                logger.warning("PIL parse byte data to gray image error: {}".format(e))
                return None
        finally:
            stream.close()
        image_rgb = np.array(image_pil)
        assert image_rgb.ndim == 3
        return image_rgb[..., ::-1]  # rgb 2 bgr


if __name__ == "__main__":
    print("hello word!")