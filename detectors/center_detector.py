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
        hm = output["hm"].sigmoid_()
        wh = output["wh"]
        reg = output["reg"] if self.opt.reg_offset else None


        return
