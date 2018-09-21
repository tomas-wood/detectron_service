import urllib3
import pickle
import json

http = urllib3.PoolManager()
detectron_srvc_url = "http://0.0.0.0:5000/detectron"

def get_cls_boxes(img_url, detectron_srvc_url=detectron_srvc_url):
    r = http.request('PUT', detectron_srvc_url, fields={"data":img_url})
    return pickle.loads(json.loads(r.data)["cls_boxes"])
