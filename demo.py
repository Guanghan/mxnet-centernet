from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
from opts import opts
from detectors.center_detector import CenterDetector

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  detector = CenterDetector(opt)

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
      '''
      detector.run returns:
           {'results': results, 'tot': tot_time, 'load': load_time,
            'pre': pre_time, 'net': net_time, 'dec': dec_time,
            'post': post_time, 'merge': merge_time}
      '''
      ret = detector.run(image_name)

      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)

      formated_ret = format_results(ret)

      img_name = image_name.split('/')[-1]
      output_path = "/export/guanghan/CenterNet-Gluon/output/" + img_name
      visualize_results(formated_ret, image_name, output_path)


def format_results(ret):
    det_candidates = ret["results"][1].tolist()  # choose class 0
    for i in range(2, 81):
        det_candidates.extend(ret["results"][i].tolist())  # add other classes

    thresh = 0.3
    formated_ret = []
    for det_candidate in det_candidates:
        x1, y1, x2, y2, conf = det_candidate

        if conf < thresh: continue
        else:
            formated_candidate = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
            formated_ret.append(formated_candidate)
    return formated_ret


def visualize_results(formated_ret, img_path, save_path):
    # Visualize Centers
    img = cv2.imread(img_path)
    for bbox in formated_ret:
        x1, y1, w, h = bbox
        center_x = int(x1 + w * 0.5)
        center_y = int(y1 + h * 0.5)
        cv2.circle(img, (center_x, center_y), 4, (255, 0, 0), 2)
        cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
    cv2.imwrite(save_path, img)


if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
