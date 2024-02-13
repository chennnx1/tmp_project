# Code written by Chen Xie, 2024.
import argparse
import cv2
import numpy as np
import os
import xmltodict
import xml.etree.ElementTree as ET
import json
import pandas as pd
from tqdm import tqdm

coco_cls = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush']


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def main(args):
    label_file = os.path.join(args.label_root, args.set, args.video + '_annt.xml')
    tree = ET.parse(label_file)
    xml_data = tree.getroot()
    xmlstr = ET.tostring(xml_data, encoding='utf-8', method='xml')
    label_file = dict(xmltodict.parse(xmlstr))

    imgs = sorted(os.listdir(os.path.join(args.dataroot, args.set, args.video)))

    coco = {'images': [], 'annotations': [], 'categories': []}

    # categories
    categories = []
    cls_id = 0
    for cls in coco_cls:
        tmp_dict = {
            'id': cls_id,
            'name': cls
        }
        categories.append(tmp_dict)
    coco['categories'] = categories

    img_id, annos_id = 0, 0
    for frame, img in enumerate(tqdm(imgs)):
        # images
        img_path = os.path.join(args.dataroot, args.set, args.video, img)  # absolute img path
        img_file = cv2.imread(img_path)
        img_height, img_width = img_file.shape[0], img_file.shape[1]

        image = {
            'id': img_id,
            'file_name': os.path.join(args.set, args.video, img),
            'height': img_height,
            'width': img_width
        }
        coco['images'].append(image)

        if args.labels:
            ped_infos = [t for t in label_file['annotations']['track'] if t['@label'] == 'pedestrian']

            # extract all ped bboxes in one frame
            annotated_frames = pd.read_csv(os.path.join(args.label_root, args.set, args.set + '_annotated_frames.csv'),
                                           sep='\t', header=None)
            pd.set_option('display.max_columns', None)
            tmp = annotated_frames.iloc[int(args.video.split('_')[-1]) - 1]
            annotated_frames = list(np.array(tmp)[0].split(','))

            if str(frame) in annotated_frames:
                for ped_info in ped_infos:
                    for info in ped_info['box']:
                        if info['@frame'] == str(frame):
                            x1 = float(info['@xtl'])
                            y1 = float(info['@ytl'])
                            x2 = float(info['@xbr'])
                            y2 = float(info['@ybr'])

                            h = y2 - y1
                            w = x2 - x1

                            annotation = {
                                'id': annos_id,
                                'image_id': img_id,
                                'category_id': 0,  # person
                                'bbox': [x1, y1, w, h],
                                'area': h * w,
                                'iscrowd': 0
                            }

                            coco['annotations'].append(annotation)
                            annos_id += 1
        img_id += 1

    # Write COCO annotation to file
    output_path = os.path.join(args.output_dir, args.set)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_file = os.path.join(output_path, args.video + '.json')

    with open(os.path.join(output_file), 'w') as f:
        json.dump(coco, f, cls=NpEncoder)
    print("Annotations in COCO format saved to", output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert PIE annotations to COCO json format',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', type=str,
                        default='/datafast/104-1/Datasets/xiechen/ped_intention/PIE/PIE_imgs',
                        help="Path where images are saved.")

    parser.add_argument('--label_root', type=str,
                        default='/home/xiechen/codes/ped_intention_img/pie_devkit/annotations',
                        help="Path where bboxes labels are saved.")

    parser.add_argument('--labels', type=str,
                        default=True,
                        help="Whether need labels")

    parser.add_argument('--set', type=str,
                        default='set01',
                        help="set name in PIE")

    parser.add_argument('--video', type=str,
                        default='video_0001',
                        help="video name in PIE")

    parser.add_argument('--output_dir', type=str,
                        default='/datafast/104-1/Datasets/xiechen/ped_intention/PIE/PIE_imgs/coco_annos',
                        help='Output directory.')

    args = parser.parse_args()
    main(args)
