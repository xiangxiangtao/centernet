# coding:utf-8
# 运行前请先做以下工作：
# pip install lxml
# 将所有的图片及xml文件存放到xml_dir指定的文件夹下，并将此文件夹放置到当前目录下
#

import os
import glob
import json
import shutil
import numpy as np
import xml.etree.ElementTree as ET

START_BOUNDING_BOX_ID = 1
save_path = "."


def get(root, name):
    return root.findall(name)


def get_and_check(root, name, length):
    vars = get(root, name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def convert(xml_list, json_file,img_ext):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    categories = pre_define_categories.copy()
    bnd_id = START_BOUNDING_BOX_ID
    all_categories = {}
    for index, line in enumerate(xml_list):
        # print("Processing %s"%(line))
        xml_f = line
        tree = ET.parse(xml_f)
        root = tree.getroot()

        filename = os.path.basename(xml_f)[:-4] + ".{}".format(img_ext)
        image_id = 20210000001 + index
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height, 'width': width, 'id': image_id}
        json_dict['images'].append(image)
        #  Currently we do not support segmentation
        segmented = get_and_check(root, 'segmented', 1).text
        assert segmented == '0'
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category in all_categories:
                all_categories[category] += 1
            else:
                all_categories[category] = 1
            if category not in categories:
                if only_care_pre_define_categories:
                    continue
                new_id = len(categories) + 1
                print(
                    "[warning] category '{}' not in 'pre_define_categories'({}), create new id: {} automatically".format(
                        category, pre_define_categories, new_id))
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(float(get_and_check(bndbox, 'xmin', 1).text))
            ymin = int(float(get_and_check(bndbox, 'ymin', 1).text))
            xmax = int(float(get_and_check(bndbox, 'xmax', 1).text))
            ymax = int(float(get_and_check(bndbox, 'ymax', 1).text))
            assert (xmax > xmin), "xmax <= xmin, {}".format(line)
            assert (ymax > ymin), "ymax <= ymin, {}".format(line)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                image_id, 'bbox': [xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'steel', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)

    # print(json_dict)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    print("------------create {} done--------------".format(json_file))
    print("find {} categories: {} -->>> your pre_define_categories {}: {}".format(len(all_categories),
                                                                                  all_categories.keys(),
                                                                                  len(pre_define_categories),
                                                                                  pre_define_categories.keys()))
    print("category: id --> {}".format(categories))
    print(categories.keys())
    print(categories.values())


if __name__ == '__main__':
    composite='composite_gas_gmy_500_400'
    composite1='composite_gas_1_gmy_500_400'
    gas_composite_18_1 = "composite_18.1_gmy"

    real_annotated='real_annotated'
    real_annotated_1='real_annotated_1'
    real_annotated_1 = 'real_annotated_1'
    real_annotated_gmy='real_annotated_gmy'
    real_7_gmy='real_7_gmy'

    dataset_name=real_7_gmy############################################################################################

    dataset_list0=[composite,composite1,gas_composite_18_1]
    dataset_list1=[real_annotated,real_annotated_1,real_annotated_gmy,real_7_gmy]

    if dataset_name in dataset_list0:
        dataset_mode="composite"
        myclass='gas'
        img_ext='jpg'
    elif dataset_name in dataset_list1:
        dataset_mode="real"
        myclass = 'smoke'
        img_ext='png'
    else:
        print("error!")

    # 定义你自己的类别
    # classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    classes = [myclass]
    pre_define_categories = {}
    for i, cls in enumerate(classes):
        pre_define_categories[cls] = i + 1
    # 这里也可以自定义类别id，把上面的注释掉换成下面这行即可
    # pre_define_categories = {'a1': 1, 'a3': 2, 'a6': 3, 'a9': 4, "a10": 5}
    print(pre_define_categories)
    only_care_pre_define_categories = True  # or False

    # 保存的json文件
    save_folder=r"/home/ecust/txx/project/gmy_2080_copy/CenterNet-master/CenterNet-master/data"
    save_json_folder = os.path.join(save_folder,'{}'.format(dataset_name),'annotations')
    if not os.path.exists(save_json_folder):
        os.makedirs(save_json_folder)
    save_json_train = os.path.join(save_json_folder, 'train_{}.json'.format(myclass))
    save_json_val = os.path.join(save_json_folder, 'val_{}.json'.format(myclass))
    if dataset_name=="composite":
        save_json_test = os.path.join(save_json_folder, 'test_{}.json'.format(myclass))######################

    # 初始文件所在的路径
    # train_xml_dir = r"E:\jupyter_code\NEU-DET\train\label"
    # val_xml_dir = r"E:\jupyter_code\NEU-DET\valid\label"

    train_xml_dir = r"/home/ecust/txx/dataset/gas/IR/{}/{}/train/label".format(dataset_mode,dataset_name)
    val_xml_dir = r"/home/ecust/txx/dataset/gas/IR/{}/{}/val/label".format(dataset_mode,dataset_name)
    if dataset_name == "composite":
        test_xml_dir= r"/home/ecust/txx/dataset/gas/IR/{}/{}/test/label".format(dataset_mode,dataset_name)###################

    train_xml_list = glob.glob(train_xml_dir + "/*.xml")
    val_xml_list = glob.glob(val_xml_dir + "/*.xml")
    if dataset_name == "composite":
        test_xml_list = glob.glob(test_xml_dir + "/*.xml")###################
        print(len(train_xml_list), len(val_xml_list), len(test_xml_list))#####################
    # print(len(train_xml_list), len(val_xml_list))

    # # 将xml文件转为coco文件，在指定目录下生成三个json文件（train/test/food）
    convert(train_xml_list, save_json_train,img_ext)
    convert(val_xml_list, save_json_val,img_ext)
    if dataset_name == "composite":
        convert(test_xml_list, save_json_test,img_ext)######################



    print("-" * 50)
    print("train number:", len(train_xml_list))
    print("val number:", len(val_xml_list))
    if dataset_name == "composite":
        print("test number:", len(test_xml_list))##################


