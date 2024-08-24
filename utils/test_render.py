import cv2
import os
import numpy as np
from utils_biwi import read_depth_rgb_cal,read_pose_txt


import sys

sys.path.append('..')

from Sim3DR import RenderPipeline

cfg = {
    'intensity_ambient': 0.3,
    'color_ambient': (1, 1, 1),
    'intensity_directional': 0.6,
    'color_directional': (1, 1, 1),
    'intensity_specular': 0.1,
    'specular_exp': 5,
    #'light_pos': (0, 0, 5),
    'light_pos': (0, 0, -5),
    'view_pos': (0, 0, 5)
}

render_app = RenderPipeline(**cfg)

def read_obj(obj_path):
    vertices,faces=[],[]
    infos=list(open(obj_path,"r"))

    for idx,info in enumerate(infos):
        info=info.strip()
        if("v " in info):
            _,x,y,z=info.split()
            x,y,z=float(x),float(y),float(z)
            vertices.append([x,y,z])

        if("f " in info):
            _,idx1,idx2,idx3=info.split()
            if("/" in idx1):
                idx1,idx2,idx3=idx1.split("/")[0],idx2.split("/")[0],idx3.split("/")[0]

            idx1,idx2,idx3=int(idx1),int(idx2),int(idx3)
            faces.append([idx1,idx2,idx3])

    vertices=np.array(vertices)
    faces=np.array(faces)

    return vertices,faces

def save_obj(vertices,faces,obj_path):
    fw=open(obj_path,"w")
    n_ver,n_face=vertices.shape[0],faces.shape[0]
    for i in range(n_ver):
        fw.write("v %f %f %f\n"%(vertices[i,0],vertices[i,1],vertices[i,2]))
    for i in range(n_face):
        fw.write("f %d %d %d\n"%(faces[i,0],faces[i,1],faces[i,2]))
    fw.close()

def test_read_obj():
    obj_path=r"D:\code\landmarks\test_62\data\mean.obj"
    obj_path=r"D:\download\datasets\biwi\faces_0\01.obj"

    vert,tri=read_obj(obj_path)

    print("vert.shape:",vert.shape)
    print("tri.shape:",tri.shape)

    obj_save_path=obj_path[0:-4]+"_res.obj"
    save_obj(vert,tri,obj_save_path)

    print("save: ",obj_save_path)

    vert, tri = read_obj(obj_save_path)

    vert_new,tri_new=[],[]
    n_ver,n_face=vert.shape[0],tri.shape[0]
    idx_new=1
    vert_dict={}
    tmp=[]
    for i in range(n_ver):
        if(vert[i,2]<16):
            vert_new.append(vert[i])

            vert_dict[i+1]=idx_new

            tmp.append(i+1)

            idx_new += 1

    # 原来面片映射到新面片，注意faces指标从1开始，不是0
    for i in range(n_face):
        idx1,idx2,idx3=tri[i]
        if(idx1 in tmp and idx2 in tmp and idx3 in tmp):
            tri_new.append([vert_dict[idx1],vert_dict[idx2],vert_dict[idx3]])
            #tri_new.append(tri[i])
        else:
            #print(tri[i])
            a=0

    vert_new,tri_new=np.array(vert_new),np.array(tri_new)

    obj_save_path=obj_path[0:-4]+"_res_front.obj"
    save_obj(vert_new,tri_new,obj_save_path)

def visualize_biwi(img_path=r"D:\download\datasets\biwi\faces_0\01\frame_00030_rgb.png"):

    img=cv2.imread(img_path)

    pose_txt=img_path[0:-8]+"_pose.txt"
    R_mat,T_mat=read_pose_txt(pose_txt)

    txt_path=r"D:\download\datasets\biwi\faces_0\01\rgb.cal"
    K_mat,dist,R_mat_rgb,T_mat_rgb=read_depth_rgb_cal(txt_path)

    obj_path = r"D:\download\datasets\biwi\faces_0\01_res.obj"
    #obj_path=r"D:\download\datasets\biwi\faces_0\01_res_front.obj"
    vert,tri=read_obj(obj_path)

    # 深度相机下的点云
    vert_depth=np.dot(R_mat,vert.T).T+T_mat

    vert_rgb=np.dot(R_mat_rgb,vert_depth.T).T+T_mat_rgb

    vert_render=vert_rgb.copy()

    pnt=np.dot(K_mat,vert_rgb.T).T
    pnt[:,0]=pnt[:,0]/pnt[:,2]
    pnt[:, 1] = pnt[:, 1] / pnt[:, 2]

    vert_render[:,0]=pnt[:,0]
    vert_render[:,1]=pnt[:,1]

    vert_render=np.array(vert_render,np.float32)
    vert_render[:,2]=-vert_render[:,2]

    tri=tri-1
    overlap = render_app(vert_render, tri, img.copy())

    n=pnt.shape[0]
    for i in range(n):
        x=int(pnt[i,0])
        y = int(pnt[i, 1])
        cv2.circle(img,(x,y),1,(0,0,255),-1,8)

    cv2.imshow("img",img)
    cv2.imshow("overlap", overlap)
    if(cv2.waitKey(0)==27):
        exit(-1)

if __name__=="__main__":
    #test_read_obj()
    for i in range(0,500,3):
        img_path=r"D:\download\datasets\biwi\faces_0\01\frame_%05d_rgb.png"%i
        if (not  os.path.exists(img_path)):
            continue

        visualize_biwi(img_path)
