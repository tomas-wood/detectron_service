import os
import cv2
import skimage
import numpy as np

from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, core, dyndep

from detectron.core.config import merge_cfg_from_file, cfg
import detectron.utils.model_convert_utils as mutils
import detectron.utils.blobl as blob_utils


def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]


def rescale(img, input_height, input_width):
    print("Original image shape:" + str(img.shape) + " and remember it should be in H, W, C!")
    print("Model's input shape is %dx%d") % (input_height, input_width)
    aspect = img.shape[1]/float(img.shape[0])
    print("Orginal aspect ratio: " + str(aspect))
    if(aspect>1):
        # landscape orientation - wide image
        res = int(aspect * input_height)
        imgScaled = skimage.transform.resize(img, (input_width, res))
    if(aspect<1):
        # portrait orientation - tall image
        res = int(input_width/aspect)
        imgScaled = skimage.transform.resize(img, (res, input_height))
    if(aspect == 1):
        imgScaled = skimage.transform.resize(img, (input_width, input_height))

    return imgScaled


def _prepare_blobs(
    im,
    pixel_means,
    target_size,
    max_size,
    ):
    """ Reference: blob.prep_im_for_blob()"""

    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape

    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    # Reuse code in blob_utils and fit FPN
    blob = blob_utils.im_list_to_blob([im])

    blobs = {}
    blobs['data'] = blob
    blobs['im_info'] = np.array(
        [[blob.shape[2], blob.shape[3], im_scale]],
        dtype=np.float32
    )
    return blobs


## JUST PLAYING AROUND. KEEPING TRACK OF WHAT NEEDS TO BE DONE.

def load_model():
    """ Loads the model defined in INIT_NET and PREDICT_NET into
    the caffe2 workspace. """

    device_opts = caffe2_pb2.DeviceOption()
    device_opts.device_type = caffe2_pb.CPU
    INIT_NET = "out/model_init.pb"
    PREDICT_NET = "out/model.pb"
    init_def = caffe2_pb2.NetDef()
    with open(INIT_NET) as f:
        init_def.ParseFromString(f.read())
    init_def.device_option.CopyFrom(device_opts)
    workspace.RunNetOnce(init_net)
    net_def = caffe2_pb2.NetDef()
    with open(PREDICT_NET) as f:
        net_def.ParseFromString(f.read())

    net_def.device_option.CopyFrom(device_opts)
    workspace.CreateNet(net_def.SerializeToString(), overwrite=True)


def get_image(img_url):
    img = skimage.img_as_float(skimage.io.imread(img_url)).astype(np.float32)
