#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import pickle
import json

sys.path.append(os.environ['CAFFE2_HOME'])
from caffe2.python import workspace

sys.path.append(os.environ['DETECTRON_HOME'])
from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils
import numpy as np
import pdb
import urllib3
from PIL import Image
import io

c2_utils.import_detectron_ops()

from flask import Flask, jsonify, request
from flask_restful import Resource, Api

http = urllib3.PoolManager()

# WHICH GPU DO YOU WANNA USE BRAH?!
gpu_id = 7

def init_caffe2():
    """Initialize caffe2 ONCE so we don't have to do it over and over.
    """
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    logger = logging.getLogger(__name__)

    weights = "/app/model_final.pkl"

    arg_cfg = os.path.join(os.environ['DETECTRON_HOME'],'configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml')
    output_ext = 'pdf' 

    merge_cfg_from_file(arg_cfg)
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg(cache_urls=False)

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'

    model = infer_engine.initialize_model_from_cfg(weights, gpu_id=gpu_id)

    return model

model = init_caffe2()

app = Flask(__name__)
api = Api(app)


class Detectron(Resource):
    """Get an input image and run FasterRCNN on it to get bboxes.
    """
    def put(self, image_id):
        img_url= request.form['data']
        r = http.request("GET", img_url)
        if r.status == 200:
            im = np.array(Image.open(io.BytesIO(r.data)).convert("RGB"))
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        else:
            return jsonify([])

        with c2_utils.NamedCudaScope(gpu_id):
            cls_boxes, _, _ = infer_engine.im_detect_all(
                model, im, None
            )

	cls_boxes_str = pickle.dumps(cls_boxes)
        res = {'cls_boxes':cls_boxes_str}

        return jsonify(res)

    def get(self, image_id):
        return jsonify(image_id)



api.add_resource(Detectron, '/<string:image_id>')

if __name__ == "__main__":
    app.run(use_reloader=False, debug=True, host='0.0.0.0')
