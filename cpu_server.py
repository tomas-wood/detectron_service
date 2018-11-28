import numpy as np
import skimage
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, core, dyndep
from detectron.core.config import merge_cfg_from_file, cfg
import detectron.utils.model_convert_utils as mutils
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


## JUST PLAYING AROUND. KEEPING TRACK OF WHAT NEEDS TO BE DONE.

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

