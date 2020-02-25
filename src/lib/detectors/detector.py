from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from external.nms import soft_nms
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import ctdet_decode
from models.utils import flip_tensor
from utils.post_process import ctdet_post_process

import cv2
import numpy as np
import time
import torch

from models.model import create_model, load_model
from utils.image import get_affine_transform
from utils.debugger import Debugger
from detectors.metric import mAP


class CtdetDetector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')

        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.max_per_image = 32
        self.num_classes = opt.num_classes
        self.scales = opt.test_scales
        self.opt = opt
        self.pause = True

        self.img_id = 0
        self.iou_level = opt.iou

    def process(self, images, return_time=False):
        with torch.no_grad():
            output = self.model(images)[-1]
            hm = output['hm'].sigmoid_()
            chm = output['chm'].sigmoid_()
            wh = output['wh']
            reg = output['reg'] if self.opt.reg_offset else None
            if self.opt.flip_test:
                hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
                chm = (chm[0:1] + flip_tensor(chm[1:2])) / 2
                wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                reg = reg[0:1] if reg is not None else None
            torch.cuda.synchronize()
            forward_time = time.time()
            dets, centers = ctdet_decode(hm, chm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)

        if return_time:
            return output, dets, centers, forward_time
        else:
            return output, dets, centers

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            if len(self.scales) > 1 or self.opt.nms:
                soft_nms(results[j], Nt=0.5, method=2)
        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

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

    def show_results(self, debugger, original_image, results_b, results_c):
        debugger.add_img(original_image, img_id='ctdet')
        for j in range(1, self.num_classes + 1):
            for bbox in results_b[j]:
                if bbox[4] > self.opt.vis_thresh:
                    bbox[:4], _ = self.dataset.pano.getOriginalCoord(original_image, bbox[:4], bbox[:2])
                    debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
            for center in results_c[j]:
                if center[4] > self.opt.vis_thresh:
                    _, center[:2] = self.dataset.pano.getOriginalCoord(original_image, center[:4], center[:2])
                    debugger.add_center_point(center[:2], img_id='ctdet')
        debugger.show_all_imgs(pause=self.pause)

    def save_results(self, debugger, original_image, results_b, results_c):
        debugger.add_img(original_image, img_id=self.img_id)
        for j in range(1, self.num_classes + 1):
            for bbox in results_b[j]:
                if bbox[4] > self.opt.vis_thresh:
                    bbox[:4], _ = self.dataset.pano.getOriginalCoord(original_image, bbox[:4], bbox[:2])
                    debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id=self.img_id)
            for center in results_c[j]:
                if center[4] > self.opt.vis_thresh:
                    _, center[:2] = self.dataset.pano.getOriginalCoord(original_image, center[:4], center[:2])
                    debugger.add_center_point(center[:2], img_id=self.img_id)
        debugger.save_img(imgId=self.img_id, path='../exp/results/')
        self.img_id += 1

    def get_mAP(self, ap, preds, gts):
        return ap.getAP(gts, preds)

    def pre_process(self, image, scale, meta=None):
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        if self.opt.fix_res:
            inp_height, inp_width = self.opt.input_h, self.opt.input_w
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
        else:
            inp_height = (new_height | self.opt.pad) + 1
            inp_width = (new_width | self.opt.pad) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        if self.opt.flip_test:
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}
        return images, meta

    def run(self, dataset, img_id, meta=None):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug == 3),
                            theme=self.opt.debugger_theme)
        ap = mAP(self.iou_level, self.num_classes)
        start_time = time.time()

        self.dataset = dataset
        image, anns = self.dataset.pano.loadData(img_id)
        original_image = cv2.imread(img_id)

        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        detections = []
        centers = []
        for scale in self.scales:
            scale_start_time = time.time()

            images, meta = self.pre_process(image, scale, meta)

            images = images.to(self.opt.device)
            torch.cuda.synchronize()
            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time

            output, dets, cents, forward_time = self.process(images, return_time=True)

            torch.cuda.synchronize()
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time

            dets = self.post_process(dets, meta, scale)
            cents = self.post_process(cents, meta, scale)
            torch.cuda.synchronize()
            post_process_time = time.time()
            post_time += post_process_time - decode_time

            detections.append(dets)
            centers.append(cents)

        results_b = self.merge_outputs(detections)
        results_c = self.merge_outputs(centers)
        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time

        ap_value = ap.getAP(anns, results_b)

        if self.opt.debug >= 2:
            self.show_results(debugger, original_image, results_b, results_c)

        if self.opt.debug >= 1:
            self.save_results(debugger, original_image, results_b, results_c)

        return {'results': results_b, 'tot': tot_time, 'load': load_time,
                'pre': pre_time, 'net': net_time, 'dec': dec_time,
                'post': post_time, 'merge': merge_time, 'mAP': ap_value}