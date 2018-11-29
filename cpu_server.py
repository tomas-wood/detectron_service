import os
import cv2
import skimage
import numpy as np

from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, core, dyndep

from detectron.core.config import merge_cfg_from_file, cfg
import detectron.utils.model_convert_utils as mutils
import detectron.utils.blob as blob_utils


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


def _sort_results(boxes, segms, keypoints, classes):
    indices = np.argsort(boxes[:, -1])[::-1]
    if boxes is not None:
        boxes = boxes[indices, :]
    if segms is not None:
        segms = [segms[x] for x in indices]
    if keypoints is not None:
        keypoints = [keypoints[x] for x in indices]
    if classes is not None:
        if isinstance(classes, list):
            classes = [classes[x] for x in indices]
        else:
            classes = classes[indices]

    return boxes, segms, keypoints, classes


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

def _get_result_blobs(check_blobs):
    ret = {}
    for x in check_blobs:
        sn = core.ScopedName(x)
        if workspace.HasBlob(sn):
            ret[x] = workspace.FetchBlob(sn)
        else:
            ret[x] = None

    return ret

## JUST PLAYING AROUND. KEEPING TRACK OF WHAT NEEDS TO BE DONE.

def load_model():
    """ Loads the model defined in INIT_NET and PREDICT_NET into
    the caffe2 workspace. """

    device_opts = caffe2_pb2.DeviceOption()
    device_opts.device_type = caffe2_pb2.CPU
    INIT_NET = "out/model_init.pb"
    PREDICT_NET = "out/model.pb"
    init_def = caffe2_pb2.NetDef()
    with open(INIT_NET, 'rb') as f:
        init_def.ParseFromString(f.read())
        init_def.device_option.CopyFrom(device_opts)
        workspace.RunNetOnce(init_def.SerializeToString())

    net_def = caffe2_pb2.NetDef()
    with open(PREDICT_NET, 'rb') as f:
        net_def.ParseFromString(f.read())
        net_def.device_option.CopyFrom(device_opts)
        workspace.CreateNet(net_def.SerializeToString(), overwrite=True)

    return init_def, net_def

def get_image(img_url):
    return skimage.img_as_float(skimage.io.imread(img_url)).astype(np.float32)

def run_on_image(img_url, net):
    img = get_image(img_url)
    input_blobs = _prepare_blobs(
                                 img,
                                 cfg.PIXEL_MEANS,
                                 cfg.TEST.SCALE,
                                 cfg.TEST.MAX_SIZE
                                )

    for k, v in input_blobs.items():
        workspace.FeedBlob(
                           core.ScopedName(k),
                           v,
                           mutils.get_device_option_cpu()
                          )
    workspace.RunNetOnce(net)
    blob_names = workspace.Blobs()
    for x in blob_names:
        print(x)
    #goods = [x for x in blob_names if x.split('_')[-1] == 'nms']
    #print(goods)
    scores = workspace.FetchBlob('score_nms')
    classids = workspace.FetchBlob('class_nms')
    boxes = workspace.FetchBlob('bbox_nms')
    cls_prob = workspace.FetchBlob('cls_prob')
    bbox_pred = workspace.FetchBlob('bbox_pred')
 
    print(scores)
    print(classids)
    print(boxes)

    print("cls_prob: shape {}".format(cls_prob.shape))
    print(cls_prob)
    print("bbox_pred: shape {}".format(bbox_pred.shape))
    print(bbox_pred)
    #except Exception as e:
    #    print('Model failed to run')
    #    R = 0
    #    scores = np.zeros((R,), dtype=np.float32)
    #    boxes = np.zeros((R, 4), dtype=np.float32)
    #    classids = np.zeros((R,), dtype=np.float32)

    boxes = np.column_stack((boxes, scores))
    #print(boxes)
    boxes, _, _, classids = _sort_results(
        boxes, None, None, classids)

    check_blobs = [
        "result_boxes", "result_classids",  # result
        ]
    workspace.FeedBlob('result_boxes', boxes)
    workspace.FeedBlob('result_classids', classids)

    ret = _get_result_blobs(check_blobs)

    return ret

def load_environment():

    core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    DETECTRON_HOME = os.environ.get('DETECTRON_HOME')
    if DETECTRON_HOME is None:
        DETECTRON_HOME = '/home/thomas/code/detectron_service/detectron'
    cfg_fname = os.path.join(
                DETECTRON_HOME,
                'configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml'
                          )
    merge_cfg_from_file(cfg_fname)
    detectron_ops_lib = "/home/thomas/caffe2/lib/libcaffe2_detectron_ops_gpu.so"
    dyndep.InitOpsLibrary(detectron_ops_lib)
    _, net = load_model()
    return net

def main():
    #img_url = "https://cdn.japantimes.2xx.jp/wp-content/uploads/2018/03/p9-masangkay-cathat-a-20180319-870x580.jpg"
    img_url = "https://d3d00swyhr67nd.cloudfront.net/w1200h1200/STF/STF_STKMG_030.jpg"
    net = load_environment()
    ret = run_on_image(img_url, net)
    print(ret)

if __name__ == "__main__":
    main()
