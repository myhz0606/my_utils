"""
@author: wwjiang
@time: 2021/12/06: 13:39
"""


import os
import zipfile
from io import BytesIO
import json
import time 
import lmdb
import glob 
import yaml
import csv 
import random
import base64 
import shutil 
import logging 
from logging.handlers import RotatingFileHandler
import numpy as np 
from PIL import Image
import cv2 
from tqdm import tqdm 
from typing import List, Tuple, Dict, Union, Sequence


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger()


__all__ = [
    "get_prefix",
    "get_suffix", 
    "get_basename", 
    "PlainLogger",
    "checkdir", 
    "array_to_bytes", 
    "bytes_to_array", 
    "read_base64_file", 
    "base64_file_to_array", 
    "ndarray2base64str",
    "base64str2ndarray",
    "get_all_file_path", 
    "get_all_dir_path", 
    "dict2json", 
    "json2dict", 
    "image_sampler", 
    "file_cp_tree", 
    "file_mv_tree", 
    "decode_text_by_line",
    "dict2str_hierarchy",
    "parse_csv",
    "save_csv",
    "load_yaml_config",
    "ImageLMDBBuilder", 
    "ImageLMDBReader", 
    "imgfile2lmdb",
    "imgarr2lmdb", 
    "cal_time", 
    "cal_time_ms", 
    "get_logger", 
    "extract_zipfile_img", 
    "extract_zipfile_img_batch", 
    "get_key", 

]


get_prefix = lambda x: '.'.join(os.path.basename(x).split('.')[:-1])
get_suffix = lambda x: os.path.basename(x).split('.')[-1] 
get_basename = lambda x: os.path.basename(x)


class PlainLogger:
    def info(self, s):
        pass 
    def debug(self, s):
        pass 
    def warn(self, s):
        pass 
    def error(self, s):
        pass  


def checkdir(path: str) -> None:
    """check whether path exist, if not, create it! """
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass 


def array_to_bytes(x: np.ndarray) -> bytes:
    """convert ndarray to byte"""
    np_bytes = BytesIO()
    np.save(np_bytes, x, allow_pickle=True)
    return np_bytes.getvalue()
 
 
def bytes_to_array(b: bytes) -> np.ndarray:
    """inverse array_to_bytes"""
    np_bytes = BytesIO(b)
    return np.load(np_bytes, allow_pickle=True)


def read_base64_file(file_path: str) -> str:
    """read file base64 string"""
    with open(file_path, "rb") as f:
        file_base64 = base64.b64encode(f.read())
    return file_base64.decode('utf-8')


def base64_file_to_array(img_base64: str) -> np.ndarray:
    """inverse read base64 file"""
    img_data = base64.b64decode(img_base64)
    img_bytes = BytesIO(img_data)
    img = Image.open(img_bytes).convert("RGB")
    return np.array(img)


def ndarray2base64str(ndarray_vector: np.ndarray, dtype='>f4') -> str:
    """convert np.ndarray to base64 str"""
    base64_vector = base64.b64encode(ndarray_vector.astype(np.dtype(dtype))).decode("utf-8")
    return base64_vector


def base64str2ndarray(base64_vector, dtype='>f4') -> np.ndarray:
    """convert base64 str to np.ndarray"""
    ndarray_vector = np.frombuffer(base64.decodebytes(base64_vector.encode()), dtype=np.dtype(dtype))
    return ndarray_vector


def get_all_file_path(rtpath, 
                        file_suffix_ls: List = ["jpg", "jpeg", "png", "gif", "jfif", "PNG", "JPEG", "JPG"],
                        max_recurrent_deep=10000):
    """
    递归rtpath目录及其子目录，保存目录下所有后缀在file_siffix_ls的文件
    Args:
        rtpath: str, 递归的目标目录
        file_suffix_ls: List, 需要保存的文件类型,
            when "*" in file_suffix_ls or not isinstance(file_suffix_ls, list) return all file path
        max_recurrent_deep: 最大递归深度

    Returns: List, 所有目标文件的绝对路径
    """
    total_file_path = [] 
    for rtdir, dirname, files in os.walk(rtpath):
        cur_level = rtdir.split(rtpath)[-1].split(os.sep).__len__()
        if cur_level > max_recurrent_deep:
            continue 
        if "*" in file_suffix_ls or file_suffix_ls == "*":
            total_file_path.extend([os.path.join(rtdir, i) for i in files])
        else:
            total_file_path.extend([os.path.join(rtdir, i) for i in files if get_suffix(i).lower() in file_suffix_ls])
    return total_file_path


def get_all_file_path_old(rtpath,
                      file_suffix_ls: List = ["jpg", "jpeg", "png", "gif", "jfif", "PNG", "JPEG", "JPG"],
                      max_recurrent_deep=10000):
    """
    递归rtpath目录及其子目录，保存目录下所有后缀在file_siffix_ls的文件
    Args:
        rtpath: str, 递归的目标目录
        file_suffix_ls: List, 需要保存的文件类型,
            when "*" in file_suffix_ls or not isinstance(file_suffix_ls, list) return all file path
        max_recurrent_deep: 最大递归深度

    Returns: List, 所有目标文件的绝对路径
    """

    recurrent_deep_info = {}
    tgt_path = [rtpath]
    total_file_path = []
    recurrent_deep_info[rtpath] = 1
    while tgt_path:
        cur_path = tgt_path.pop(0)
        cur_dir_deep = recurrent_deep_info[cur_path]
        if cur_dir_deep > max_recurrent_deep:
            continue
        cur_item_ls = os.listdir(cur_path)
        for cur_item in cur_item_ls:
            cur_item_path = os.path.join(cur_path, cur_item)
            if os.path.isfile(cur_item_path):
                cur_file_suffix = cur_item.split('.')[-1]
                if isinstance(file_suffix_ls, list) and '*' not in file_suffix_ls:
                    if cur_file_suffix.lower() in file_suffix_ls:
                        total_file_path.append(cur_item_path)
                else:
                    total_file_path.append(cur_item_path)
                continue
            elif os.path.isdir(cur_item_path):
                recurrent_deep_info[cur_item_path] = recurrent_deep_info[cur_path] + 1
                tgt_path.append(cur_item_path)
                continue
            else:
                continue
    return total_file_path


def get_all_dir_path(rtpath: str, max_recurrent_deep=1000):
    """
        递归rtpath目录及其子目录，保存目录下递归深度小于max_recurrent_deep的所有路径
        Args:
            rtpath: str, 递归的目标目录
            file_suffix_ls: List, 需要保存的文件类型
            max_recurrent_deep: 最大递归深度

        Returns: List, 所有目标文件的绝对路径
        """

    recurrent_deep_info = {}
    tgt_path = [rtpath]
    total_dir_path = [rtpath]
    recurrent_deep_info[rtpath] = 1
    while tgt_path:
        cur_path = tgt_path.pop(0)
        cur_dir_deep = recurrent_deep_info[cur_path]
        if cur_dir_deep > max_recurrent_deep:
            continue
        cur_item_ls = os.listdir(cur_path)
        for cur_item in cur_item_ls:
            cur_item_path = os.path.join(cur_path, cur_item)
            if os.path.isfile(cur_item_path):
                continue
            elif os.path.isdir(cur_item_path):
                total_dir_path.append(cur_item_path)
                recurrent_deep_info[cur_item_path] = recurrent_deep_info[cur_path] + 1
                tgt_path.append(cur_item_path)
                continue
            else:
                continue
    return total_dir_path


def dict2json(input_dict: Dict, json_path: str):
    json_str = json.dumps(input_dict, indent=4)
    with open(json_path, 'w', encoding="utf-8") as f:
        f.write(json_str)


def json2dict(json_path: str) -> Dict:
    with open(json_path, 'r', encoding="utf-8") as f:
        json_info = json.load(f)
    return json_info


def image_sampler(rtpath,
                  save_rtpath,
                  sample_num=10):

    """
    it is applicable for directory with second-level directories, sampling N items from sub dir
    Args:
        rtpath: str, src path
        save_rtpath:str, save rtpath
        sample_num: int, sample number

    Returns:

    """

    sub_dir_ls = [i for i in os.listdir(rtpath) if os.path.isdir(os.path.join(rtpath, i))]
    if not os.path.exists(save_rtpath):
        os.mkdir(save_rtpath)
    print(f"start sample, \n src rtpath {rtpath} \n dst rtpath {save_rtpath}")
    for i in sub_dir_ls:
        cur_src_dir = os.path.join(rtpath, i)
        cur_dst_dir = os.path.join(save_rtpath, i)
        if not os.path.exists(cur_dst_dir):
            os.mkdir(cur_dst_dir)

        cur_file_ls = os.listdir(cur_src_dir)
        random.shuffle(cur_file_ls)
        sample_file_ls = cur_file_ls[: sample_num]

        for file in sample_file_ls:
            cur_src_path = os.path.join(cur_src_dir, file)
            cur_dst_path = os.path.join(cur_dst_dir, file)
            shutil.copyfile(cur_src_path, cur_dst_path)

    print(f"finish all...")


def __file_op_tree(src_rtpath, dst_rtpath, op_mode="move", file_suffix_ls=['*'], max_recurrent_deep=1000):
    """move src path list to dst rtpath"""
    if op_mode in ["move", "mv"]:
        op_func = shutil.move
    elif op_mode in ["cp", "copy", "copyfile"]:
        op_func = shutil.copyfile
    else:
        raise NotImplementedError(f"op mode not accomplish") 
    src_path_ls = get_all_file_path(src_rtpath, file_suffix_ls=file_suffix_ls, max_recurrent_deep=max_recurrent_deep)
    print(f"total path number is {len(src_path_ls)}") 
    for src in src_path_ls:
        file_name = os.path.split(src)[-1]
        dst = os.path.join(dst_rtpath, file_name)
        op_func(src, dst)
    print(f"finish all")


def file_cp_tree(src_rtpath, dst_rtpath, file_suffix_ls=['*'], max_recurrent_deep=1000):
    __file_op_tree(src_rtpath, dst_rtpath, op_mode='cp', file_suffix_ls=file_suffix_ls, max_recurrent_deep=max_recurrent_deep)


def file_mv_tree(src_rtpath, dst_rtpath, file_suffix_ls=['*'], max_recurrent_deep=1000):
    __file_op_tree(src_rtpath, dst_rtpath, op_mode='mv', file_suffix_ls=file_suffix_ls, max_recurrent_deep=max_recurrent_deep)


def decode_text_by_line(txt_path):
    """read text file then return by line"""
    line_info = [] 
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.readlines()
        for line in content:
            line_clean = line.strip(' ').strip("\t").strip("\n")
            if len(line_clean) > 0:
                line_info.append(line_clean)
            else:
                continue 
    return line_info


def dict2str_hierarchy(d: Dict, logger=None, sep='\n') -> str:
    """
    print dict info with format
    Args:
        d(Dict): dict
        depth(int): depth of dict key
        sep: connect char
    Returns:
        dict_info_str(str)
    Examples:
    >>> d = dict(a=1, b=dict(c=1, d='apple'))
    >>> dict2str_hierarchy(d)
    -a
      |--(int)1
    -b
      -c
        |--(int)1
      -d
        |--(str)apple"
    >>> dict2str_hierarchy(d, sep=' ')
    -a  |--(int)1 -b    -c...
    """
    dict_info = []
    depth = 0
    tap_str: str = "  "

    def _dict2str_bk(d, depth):
        key_indent = ''.join([tap_str for _ in range(depth)]) if depth > 0 else ''
        for k, v in d.items():
            candidate = [(v, depth)]
            dict_info.append(f"{key_indent}-{k}")
            while candidate:
                cur_item, depth = candidate.pop()
                if not isinstance(cur_item, Dict):
                    cur_indent = ''.join([tap_str for _ in range(depth + 1)])
                    cur_item_type = repr(type(cur_item)).split("'")[-2]
                    dict_info.append(f"{cur_indent}|--({cur_item_type}){cur_item}")
                else:
                    _dict2str_bk(cur_item, depth+1)
    _dict2str_bk(d, depth)
    if logger is not None:
        for i in dict_info:
            logger.info(i)
    out_str = sep.join(dict_info)
    del dict_info
    return out_str


def parse_csv(csv_path: str, wanted_fields_id_ls: List=["all"], include_head: bool=True):
    """parse csv file
    example:
    >>> parse_csv("xxx.csv", [0, 1], include_head=True)
    ============================================
    xxx.csv format
    number alphabet other
    1      a        apple
    2      b        banana
    3      c        caffee
    ============================================
    >>> [["1", "2" , "3"], ["a", "b", "c"]]
    Args:
        csv_path(str): csv file path
        include_head(bool): current csv file whether having head
        wanted_fields_id_ls(List): which field we wanted. if ["all"] return all fields
    Return:
        csv_head_ls(Optional[List]): if include_head is True, return title ls, else return None
        container(List): each item is a List, information of wanted_fields_id_ls
    """
  
    with open(csv_path, 'r') as f:
        csv_info = csv.reader(f)
        if include_head:
            csv_head_ls = next(csv_info)
        for idx, line in tqdm(enumerate(csv_info)):
            if idx == 0:  # init container
                item_num = len(line) 
                if wanted_fields_id_ls == ["all"]:
                    container = [[line[i]] for i in range(item_num)] 
                    wanted_fields_id_ls = [i for i in range(item_num)]  
                else:
                    if max(wanted_fields_id_ls) > item_num:
                        raise OverflowError(f"csv only have {item_num} fields but received {max(wanted_fields_id_ls)}") 
                    container = [[line[wanted_fields_id_ls[i]]] for i in range(len(wanted_fields_id_ls))] 
            else:
                for idx2, j in enumerate(wanted_fields_id_ls):
                    container[idx2].append(line[j]) 
    if include_head:
        return csv_head_ls, container
    return None, container


def save_csv(csv_path: str, save_info: List[List]) -> None:
    """
    save information to csv file
    Example:
    >>> save("xxx.csv", [["number", "alphabet", "other"], ["1", "a", "apple"], ["2", "b", "banana"], ["3", "c", "caffee"]])
    >>> File: xxx.csv
    ============================================
    xxx.csv format
    number alphabet other
    1      a        apple\
    2      b        banana
    3      c        caffee
    ============================================
    Args:
        csv_path(str): save csv file path 
        save_info(List[List]): saving information. each item is the line element of csv file.
    Return:
        csv file.
    """
    with open(csv_path, 'w', encoding="utf-8") as f:
        csv_writer = csv.writer(f) 
        csv_writer.writerows(save_info)


def load_yaml_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    global_config = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    return global_config


class ImageLMDBBuilder:

    def __init__(self, lmdb_path, map_size=1024 * 1024 * 1024 * 1024, db_discription=""):
        """
        map_size: default 1T
        """
        self.env, self.txn = self.init_database(lmdb_path, map_size=map_size, db_discription=db_discription)
        self.error_item = []
        self.n_sample = 0
        self._id = 0

    def init_database(self, lmdb_path, map_size, db_discription):
        env = lmdb.open(lmdb_path, map_size=map_size)
        txn = env.begin(write=True)
        txn.put(key='discription'.encode(), value=db_discription.encode())
        return env, txn

    def insert_data(self, img_name, img: Union[np.ndarray, bytes]):
        assert isinstance(img_name, str)
        if isinstance(img, np.ndarray):
            img = array_to_bytes(img) 
        assert isinstance(img, bytes) 
        self.txn.put(key=f"image_{self._id:08d}".encode(), value=img)
        self.txn.put(key=f"name_{self._id:08d}".encode(), value=img_name.encode())
        self._id += 1
        self.n_sample += 1
    
    def commit_lmdb(self):
        self.txn.commit()
    
    def close_lmdb(self):
        self.env.close()

    def encode_image_to_db(self, img_rtpath, process_func=lambda x: x, save_file_bytes: bool=True):
        """
        img_rtpath(str): img root path
        process_func(function): transform image ndarray 
        save_file_bytes(bool): 
        """
        img_path_ls = get_all_file_path(img_rtpath) 
        for img_path in tqdm(img_path_ls):
            if save_file_bytes:
                with open(img_path, 'rb') as f:
                    img = f.read()
            else:
                try:
                    img = cv2.imread(img_path)[..., :3][..., ::-1]  # read rgb image 
                    img = process_func(img) 
                    assert img is not None 
                except Exception as e:
                    print(e)
                    self.error_item.append(img_path) 
                    continue
            self.insert_data(os.path.basename(img_path), img)
        self.txn.put(key="image_numbers".encode(), value=repr(self.n_sample).encode())
        self.commit_lmdb()
        self.close_lmdb()


class ImageLMDBReader:

    def __init__(self, lmdb_path, logger=PlainLogger()):
        """
        Args:
            lmdb_path: lmdb file path
            img_info(str): choose from (arr, file_byte):
                if arr: decode_func = self.decode_arr
                else: decode_func = self.decode_file_bytes 
        """

        self.env = lmdb.open(lmdb_path, readonly=True)
        self.txn = self.env.begin()
        self.n_sample = int(self.txn.get("image_numbers".encode()))
        self.logger = logger 

    def __len__(self):
        return self.n_sample 

    def __getitem__(self, idx):
        """
        Args: 
            idx: index of db
        Return:
            img(bytes)
            img_name (str)
        """
        img, img_name = self.get_lmdb_item(idx) 
        return img, img_name
    
    def get_lmdb_item(self, idx) -> Tuple[np.ndarray,  str]: 
        img_id = f"image_{idx:08d}".encode()
        name_id = f"name_{idx:08d}".encode()
        img = self.txn.get(img_id)
        img_name = self.txn.get(name_id).decode("utf-8")
        return img, img_name 

    def decode_arr(self, lmdb_img) -> np.ndarray:
        """decode ndarray bytes to ndarray"""
        img = bytes_to_array(lmdb_img)
        return img  

    def decode_file_bytes_to_arr(self, lmdb_img) -> np.ndarray:
        """deocode file bytes to ndarray"""
        try:
            img_rgb = cv2.imdecode(np.asarray(
                    bytearray(lmdb_img), dtype=np.uint8), cv2.IMREAD_COLOR)[..., :3][..., ::-1]
            assert img_rgb is not None 
        except Exception as e:
            self.logger.info(f"Opencv Decode Error, msg: {e}")
            try:
                stream = BytesIO(lmdb_img)
                img_rgb = Image.open(stream).convert("RGB")
            except:
                img_L = Image.open(stream).convert("L")
                img_rgb = img_L.convert("RGB")
                if img_rgb.ndim == 2:
                    img_rgb = np.stack([img_rgb] * 3, axis=2)
                assert img_rgb.ndim == 3 and img_rgb.shape[-1] == 3
        assert isinstance(img_rgb, np.ndarray) and  img_rgb.ndim == 3 and img_rgb.shape[-1] == 3
        return img_rgb

    def decode_file_bytes_to_base64(self, lmdb_img) -> str:
        """deocode file bytes to base64 str"""
        assert isinstance(lmdb_img, bytes)
        file_base64 = base64.b64encode(lmdb_img)
        return file_base64.decode("utf-8")


def imgfile2lmdb(img_rtpath, lmdb_path, db_discription="save image file bytes to db"):
    """save img file byte to lmdb"""
    db_builder = ImageLMDBBuilder(lmdb_path, db_discription=db_discription) 
    db_builder.encode_image_to_db(img_rtpath, save_file_bytes=True)
    print(f"finish all~") 


def imgarr2lmdb(img_rtpath, lmdb_path, process_func, db_discription="save image arr bytes to db"):
    """save img file byte to lmdb"""
    db_builder = ImageLMDBBuilder(lmdb_path, db_discription=db_discription) 
    db_builder.encode_image_to_db(img_rtpath, save_file_bytes=False, process_func=process_func)
    print(f"finish all~")  


def cal_time(t_start: float) -> float:
    """calculate time consume. From t_start to current time.
    """
    return time.perf_counter() - t_start


def cal_time_ms(t_start: float) -> float:
    """calculate time consume. From t_start to current time. millisecond"""
    return cal_time(t_start) * 1000


def get_logger(log_file, backupcount=3, maximum_log_file_size=50) -> logging.Logger:
    """generate logger object"""
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    maxbytes = maximum_log_file_size * 1024 * 1024  #
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh = RotatingFileHandler(filename=log_file, maxBytes=maxbytes, backupCount=backupcount, encoding="utf-8")
    fh.setLevel(logging.INFO)
    #  set format
    formatter = logging.Formatter(
        "%(asctime)s-%(filename)s[line:%(lineno)d]-%(process)d-%(thread)d-%(levelname)s-%(message)s")
    # formatter2 = logging.Formatter("%(asctime)s-%(filename)s[line:%(lineno)d]-%(levelname)s-%(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def extract_zipfile_img(zip_file, save_rtpath) -> None:
    """extract zipfile image to destination dir"""
    checkdir(save_rtpath)
    unzip_save_dir = os.path.join(save_rtpath, f"{get_prefix(zip_file)}")

    def keep_img_file(file_ls, img_suffix_ls=("png", "PNG", "jpg", "jpeg", "JPEG", "JPG", "bmp", "jfif")):
        return [i for i in file_ls if get_suffix(i) in img_suffix_ls]

    with zipfile.ZipFile(zip_file, mode="r") as zf:
        img_file_ls = keep_img_file(zf.namelist())
        for img_file in img_file_ls:
            zf.extract(img_file, path=unzip_save_dir)

    img_path_ls = get_all_file_path(unzip_save_dir)
    img_prefix = f"{get_prefix(zip_file)}_"
    for img_path in img_path_ls:
        dst_img = os.path.join(save_rtpath, f"{img_prefix}{get_basename(img_path)}")
        shutil.move(img_path, dst_img)

    shutil.rmtree(unzip_save_dir)


def extract_zipfile_img_batch(zip_file_rtpath, save_rtpath) -> None:
    """extract zipfile image"""
    zip_file_ls = get_all_file_path(zip_file_rtpath, file_suffix_ls=["zip"])
    for zip_file in tqdm(zip_file_ls):
        extract_zipfile_img(zip_file, save_rtpath)


def get_key(_d: Dict, key_ls: Sequence) -> List:
    """get dict information by key"""
    res = []
    for k in key_ls:
        res.append(_d[k])
    assert len(res) == len(key_ls)


if __name__ == "__main__":
    import time
    rtpath = r"/Users/jiangweiwei/Data/s0_image_retrieval/s1_invalid_patent"
    t1 = time.perf_counter() 
    img_path_1 = get_all_file_path_old(rtpath, file_suffix_ls=["jpg"], max_recurrent_deep=300)
    t2 = time.perf_counter()
    img_path_2 = get_all_file_path(rtpath, file_suffix_ls=["jpg"], max_recurrent_deep=300)
    t3 = time.perf_counter()
    print(f"{(t3-t2)*1000:.2f}ms  {(t2-t1)*1000:.2f}ms")
    print(set(img_path_1) == set(img_path_2))



