import cv2
import numpy as np
import time

from external.nms import soft_nms
from models.tensor_utils import flip_tensor
from utils.image import get_affine_transform

from base_detector import BaseDetector

# need to implement the decoder first
from models.decoder import decode_centernet
# need to implement the post processing utilities
from utils.post_process import post_process_centernet


class CenterDetector(BaseDetector):
    def __init__(self, opt):
        super(CenterDetector, self).__init__(opt)

    def process(self, images, return_time=False):
        output = self.model(images)[-1]
        heatmaps = output["hm"].sigmoid_()
        wh = output["wh"]
        reg = output["reg"] if self.opt.reg_offset else None

        if self.opt.flip_test:
            heatmaps = (heatmaps[0:1] + flip_tensor(heatmaps[1:2])) / 2
            wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
            reg = reg[0:1] if reg is not None else None

        dets = decode_centernet(heatmaps, wh, reg, K=self.opt.K)

        return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.reshape(-1, 1, dets.shape[2])
        dets = post_process_centernet(dets.copy(),
                                 [meta['c']],
                                 [meta['s']],
                                 meta['out_height'],
                                 meta['out_width'],
                                 self.opt.num_classes
                                )
        for i in range(1, self.num_classes + 1):
            dets[0][i] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][i][:, :4] /= 4
        return dets[0]


    def merge_outputs(self, detections):
        results = {}
        for i in range(1, self.num_classes + 1):
            results[i] = np.concatenate([detection[i] for detection in detections], axis=0).astype(np.float32)
            if len(self.scales) > 1 or self.opt.nms:
                soft_nms(results[i], Nt=0.5, method=2)

        scores = np.hstack([results[i][0:4] for i in range(1, self.num_classes + 1)])

        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for i in range(1, self.num_classes + 1):
                keep_inds = (results[i][:, 4] >= thresh)
                results[i] = results[i][keep_inds]
        return results
