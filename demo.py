from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2

from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    detector.pause = False
    while True:
        _, img = cam.read()
        cv2.imshow('input', img)
        ret = detector.run(img)
        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)
        if cv2.waitKey(1) == 27:
            return  # esc to quit
  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]

    for (image_name) in image_names:
      ret = detector.run(image_name)

      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)

      formated_ret = format_results(ret)
      print("formated_ret: ", formated_ret)

'''
  detector.run returns:
   {'results': results, 'tot': tot_time, 'load': load_time,
    'pre': pre_time, 'net': net_time, 'dec': dec_time,
    'post': post_time, 'merge': merge_time}
'''


def format_results(ret):
    det_candidates = ret["results"][3].tolist()
    print(len(det_candidates))
    det_candidates.extend(ret["results"][8].tolist())
    print(len(det_candidates))
    print("detection results: ", det_candidates)  # 1:car, 4:truck

    thresh = 0.5
    formated_ret = []
    for det_candidate in det_candidates:
        x1, y1, x2, y2, conf = det_candidate

        if conf < thresh: continue
        else:
            formated_candidate = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
            formated_ret.append(formated_candidate)
    return formated_ret


if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
