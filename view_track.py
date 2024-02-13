# Code written by Chen Xie, 2024.
import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.colors as mcolors


def pie_viewer(args):
    # read labels
    frames, obj_id_dict, bbox_dict = [], {}, {}
    with open(args.label_path) as f:
        for line in f.readlines():
            line = line.split()
            frame_id = int(line[0])
            obj_id = int(line[1])
            bbox = line[6:10]

            if frame_id not in obj_id_dict.keys():
                frames.append(frame_id)
                obj_id_dict[frame_id], bbox_dict[frame_id] = [], []
                obj_id_dict[frame_id].append(obj_id)
                bbox_dict[frame_id].append(bbox)
            else:
                obj_id_dict[frame_id].append(obj_id)
                bbox_dict[frame_id].append(bbox)

    # color map for obj ids
    np.random.seed(0)
    discrete_colors = list(mcolors.CSS4_COLORS.values())
    np.random.shuffle(discrete_colors)

    max_ids = np.array(obj_id_dict[frames[-1]]).max()
    while len(discrete_colors) < max_ids:
        discrete_colors += discrete_colors
    colors = discrete_colors[:max_ids]
    colors = [mcolors.hex2color(c) for c in colors]

    # draw bboxes
    img_files = []
    for frame in tqdm(range(0, 1800)):
        img_path = os.path.join(args.img_root, str(frame).zfill(5) + '.png')
        img = cv2.imread(img_path)

        if frame in frames:
            obj_ids = obj_id_dict[frame]
            bboxes = bbox_dict[frame]
            for id, box in zip(obj_ids, bboxes):
                x1, y1, x2, y2 = int(float(box[0])), int(float(box[1])), int(float(box[2])), int(float(box[3]))
                color = (colors[id][0] * 255, colors[id][1] * 255, colors[id][2] * 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, "ID:" + str(id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        img_files.append(img)
        if args.show:
            cv2.imshow('im', img)
            cv2.waitKey(10)

    # save to videos
    if args.save_video:
        os.makedirs('videos/', exist_ok=True)
        save_path = 'videos/'+args.video_name
        fps = 30
        resolution = (img.shape[1], img.shape[0])
        video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, resolution)

        print("Video save start:")
        for im in tqdm(img_files):
            video_writer.write(im)

        video_writer.release()
        print(f'Video saved as {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='View tracking result of PIE data')
    parser.add_argument('--img_root',
                        default='/Users/xc/Downloads/dataset/PIE/set01_img_1800frames/',
                        type=str, help='img root of the PIE data')
    parser.add_argument('--label_path',
                        default='/Users/xc/Documents/codes/cvc_ped_intention/pie_utils/oc_sort_pie/tracking_results/'
                                'yolov8-x/set01_video0001.txt',  # todo
                        type=str, help='tracking results file')
    parser.add_argument('--show', default=False, help='whether to show the tracking results')
    parser.add_argument('--save_video', default=True, help='whether to save the tracking video')
    parser.add_argument('--video_name', default='yolov8-x.mp4', help='video name')  # todo
    args = parser.parse_args()

    pie_viewer(args)
