# Copyright (c) OpenMMLab. All rights reserved.
'''
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

data[iframe][0=bbox][0|1=iclass][ianimal*5] = numpy.array
data[iframe][1=segm][0|1=iclass][ianimal]   = dict
'''

import argparse
import mmcv
from mmdet.apis import inference_detector, init_detector
from mmdet.core import encode_mask_results
import glob
import os

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection video demo')
    parser.add_argument('videos', help='Video folder')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--out', type=str, help='Output folder')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.out or args.show, \
        ('Please specify at least one operation (save/show the '
         'video) with the argument "--out" or "--show"')

    model = init_detector(args.config, args.checkpoint, device=args.device)

    videos = glob.glob(args.videos+'/*.mp4') + glob.glob(args.videos+'/*.avi')
    os.makedirs(args.out, exist_ok=True)
    out_pkls = [os.path.join(args.out, os.path.split(f)[-1][:-4]+'.pkl') for f in videos]
    

    for j, (video, out_pkl) in enumerate(zip(videos, out_pkls)):
        print('[{} in {}]: {}'.format(j+1, len(videos), os.path.split(video)[-1]))
        video_reader = mmcv.VideoReader(video)
        outputs = []
        for i, frame in enumerate(mmcv.track_iter_progress(video_reader)):
            result = inference_detector(model, frame)
            if len(result)==2:
                result = [result]
            result = [(bbox_results, encode_mask_results(mask_results))
                        for bbox_results, mask_results in result]
            resultf = filt_by_thr(result)
            outputs.extend(resultf)
            
        mmcv.dump(outputs, out_pkl)


def filt_by_thr(result, thr=0.5):
    result_out = []
    for a_frame_result in result:
        bbox_results, mask_results = a_frame_result
        a_frame_out = [[],[]]
        for a_class_bbox, a_class_mask in zip(bbox_results, mask_results):
            p_vals = a_class_bbox[:,-1]
            valid  = p_vals > thr
            a_class_bbox = a_class_bbox[valid]
            a_class_mask = [mask for mask,v in zip(a_class_mask,valid) if v]
            a_frame_out[0].append(a_class_bbox)
            a_frame_out[1].append(a_class_mask)

        result_out.append(a_frame_out)
    return result_out


if __name__ == '__main__':
    main()
