import glob
import lxml.etree
import numpy as np
import os


def get_image_paths(image_dir, pattern='*.jpg', return_path=False, return_np=True):
    image_paths = glob.glob(os.path.join(image_dir, pattern))
    image_names = [os.path.basename(path) for path in image_paths]

    ret = image_paths if return_path else image_names
    if return_np:
        ret = np.array(ret)

    return ret


def read_xml_info(xml_file):
    with open(xml_file, 'rb') as f:
        doc = lxml.etree.parse(f)

        # 图片名
        filename = doc.xpath('/annotation/filename')[0].text
        # 图片绝对路径
        path = doc.xpath('/annotation/path')[0].text

        boxes = []
        objects = doc.xpath('/annotation/object')
        for obj in objects:
            class_name = obj.xpath('name')[0].text
            xmin = int(float(obj.xpath('bndbox/xmin')[0].text))
            xmax = int(float(obj.xpath('bndbox/xmax')[0].text))
            ymin = int(float(obj.xpath('bndbox/ymin')[0].text))
            ymax = int(float(obj.xpath('bndbox/ymax')[0].text))
            boxes.append([class_name, xmin, xmax, ymin, ymax])

    return path, filename, boxes


