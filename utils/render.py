# coding: utf-8

__author__ = 'cleardusk'

import sys

sys.path.append(r"H:\arc3d\research\3DDFA_V2")

import cv2
import numpy as np
import datetime

from Sim3DR import RenderPipeline
from utils.functions import plot_image
#from .tddfa_util import _to_ctype
from tddfa_util import _to_ctype

cfg = {
    'intensity_ambient': 0.3,
    'color_ambient': (1, 1, 1),
    'intensity_directional': 0.6,
    'color_directional': (1, 1, 1),
    'intensity_specular': 0.1,
    'specular_exp': 5,
    'light_pos': (0, 0, 5),
    #'light_pos': (5, 0, 0),
    'view_pos': (0, 0, 5)
}

render_app = RenderPipeline(**cfg)

def render(img, ver_lst, tri, alpha=0.6, show_flag=False, wfp=None, with_bg_flag=True,cv_wait_val=0):
    if with_bg_flag:
        overlap = img.copy()
    else:
        overlap = np.zeros_like(img)

    for ver_ in ver_lst:
        ver = _to_ctype(ver_.T)  # transpose
        time_s = datetime.datetime.now()

        overlap = render_app(ver, tri, overlap)

        time_e = datetime.datetime.now()
        time_use = (time_e - time_s).total_seconds()
        print("use time= %.02f ms" % (time_use * 1000))

    if with_bg_flag:
        res = cv2.addWeighted(img, 1 - alpha, overlap, alpha, 0)
    else:
        res = overlap

    if wfp is not None:
        cv2.imwrite(wfp, res)
        print('Save visualization result to {wfp}')

    if show_flag:
        #plot_image(res)
        cv2.imshow("img",res)
        cv2.waitKeyEx(cv_wait_val)

    return res

def mesh_model(mesh_template,n_ver,n_tri,is_mean_shape=1,scale=1.0):
    vertex=np.zeros(shape=[n_ver,3])
    triangles=np.zeros(shape=[n_tri,3])
    for i in range(len(mesh_template)):
        if (i < n_ver):
            tmp_str=mesh_template[i]
            tmp_infos=tmp_str.strip().split()
            vertex[i,0]=float(tmp_infos[1])*scale
            vertex[i, 1] = float(tmp_infos[2])*scale
            vertex[i, 2] = float(tmp_infos[3])*scale
        else:
            if(is_mean_shape):
                tmp_str = mesh_template[i]
                tmp_infos = tmp_str.strip().split()
                triangles[i-n_ver, 0] = int(tmp_infos[1])
                triangles[i-n_ver, 1] = int(tmp_infos[2])
                triangles[i-n_ver, 2] = int(tmp_infos[3])
            else:
                break

    return vertex,triangles

# with_bg_flag = True
with_bg_flag = False
# show_flag=False
show_flag = True

def test_3ddfa_render():
    print("#-------------------- 3ddfa_v2 ---------------------#")
    img_path=r"H:\arc3d\research\3DDFA_V2\img.jpg"
    img=cv2.imread(img_path)
    ver_lst=np.load(r"H:\arc3d\research\3DDFA_V2\vert.npy")  # nx 3
    tri=np.load(r"H:\arc3d\research\3DDFA_V2\tri.npy")       # kx 3

    light_pos_range=np.arange(-5,5,1)
    while(1):
        for i in range(10):
            light_pos=[light_pos_range[i],0,5]
            render_app.update_light_pos(light_pos)
            render(img, [ver_lst.T], tri, show_flag=show_flag, with_bg_flag=with_bg_flag,cv_wait_val=0)

def test_3dmm_render():
    a=0
    # print("#-------------------- id100+exp50 ---------------------#")
    # mesh_template_txt = r"H:\arc3d\其他\facedata\mean_shape.obj"
    # mesh_template = list(open(mesh_template_txt, "r"))
    # n_ver=2645
    # n_tri=5104
    # vertex, triangles = mesh_model(mesh_template,n_ver,n_tri)
    # ver_lst=np.array(vertex,np.float32)
    # tri=np.int32(triangles)

    # print("#-------------------- bfm ---------------------#")
    # mesh_template_txt = r"H:\data_vee_train\dtc600\speech_fast_small\20220418_Kor_006\20220418_Kor_006_000\test_mesh_00.obj"
    # mesh_template = list(open(mesh_template_txt, "r"))
    # n_ver=35709
    # n_tri=70789
    # scale=100.0
    # vertex, triangles = mesh_model(mesh_template,n_ver,n_tri,scale=scale)
    # ver_lst=np.array(vertex,np.float32)
    # tri=np.int32(triangles)

    # cap=cv2.VideoCapture(r"H:\data_vee_train\dtc600\speech_fast_small\20220418_Chs_053.mp4")
    # ret,img=cap.read()
    # cap.release()

    imgh=640
    imgw=480
    #
    img=np.uint8(np.zeros(shape=[640,480,3]))
    #img = np.uint8(np.zeros(shape=[720,1280, 3]))

    # ver_lst[:, 0] = ver_lst[:, 0] + imgw / 2.0
    # ver_lst[:, 1] = ver_lst[:, 1] + imgh / 2.0

    #---------- x,y,z to [0,0,0] --------------#
    #mean=np.mean(ver_lst,axis=0)
    #ver_lst=ver_lst-mean

    render(img,[ver_lst.T],tri,show_flag=show_flag,with_bg_flag=with_bg_flag)

if __name__=="__main__":
    test_3ddfa_render()
