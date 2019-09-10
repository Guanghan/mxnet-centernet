import cv2
import numpy as np
import time

import sys
sys.path.insert(0, "/Users/guanghan.ning/Desktop/dev/CenterNet-Gluon/")

from external.nms import soft_nms
from models.tensor_utils import flip_tensor
from utils.image import get_affine_transform

from detectors.base_detector import BaseDetector

from models.decoder import decode_centernet
from utils.post_process import post_process_centernet
from mxnet import nd

class CenterDetector(BaseDetector):
    def __init__(self, opt):
        super(CenterDetector, self).__init__(opt)

    def process(self, images, return_time=False):
        output = self.model(images)[-1]
        heatmaps = output["hm"].sigmoid()
        wh = output["wh"]
        reg = output["reg"] if self.opt.reg_offset else None

        if self.opt.flip_test:
            heatmaps = (heatmaps[0:1] + flip_tensor(heatmaps[1:2])) / 2
            wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
            reg = reg[0:1] if reg is not None else None

        nd.waitall()
        forward_time = time.time()
        dets = decode_centernet(heatmaps, wh, reg, K=self.opt.K)
        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.asnumpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = post_process_centernet(dets.copy(),
                                 [meta['c']],
                                 [meta['s']],
                                 meta['out_height'],
                                 meta['out_width'],
                                 self.opt.num_classes
                                )
        for i in range(1, self.num_classes + 1):
            dets[0][i] = np.array(dets[0][i], dtype=np.float32).reshape(-1, 5)
            dets[0][i][:, :4] /= 4
        return dets[0]


    def merge_outputs(self, detections):
        results = {}
        for i in range(1, self.num_classes + 1):
            results[i] = np.concatenate([detection[i] for detection in detections], axis=0).astype(np.float32)
            if len(self.scales) > 1 or self.opt.nms:
                soft_nms(results[i], Nt=0.5, method=2)

        scores = np.hstack([results[i][:,4] for i in range(1, self.num_classes + 1)])

        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for i in range(1, self.num_classes + 1):
                keep_inds = (results[i][:, 4] >= thresh)
                results[i] = results[i][keep_inds]
        return results


    ''' Add debugger '''
    '''
    def debug(self, debugger, images, dets, output, scale=1):
        detection = dets.detach().cpu().numpy().copy()
        detection[:, :, :4] *= self.opt.down_ratio
        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
        for k in range(len(dets[i])):
            if detection[i, k, 4] > self.opt.center_thresh:
                debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                     detection[i, k, 4],
                                     img_id='out_pred_{:.1f}'.format(scale))

    def show_results(self, debugger, image, results):
        debugger.add_img(image, img_id='ctdet')
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[4] > self.opt.vis_thresh:
                    debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
        debugger.show_all_imgs(pause=self.pause)
    '''
