# coding=utf-8
import numpy as np
import cv2
import math
import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET



def parse_xml(xml_path):
    boxes=[]
    target = ET.parse(xml_path).getroot()
    for obj in target.iter('object'):
        name = obj.find('name').text.lower().strip()
        # name = 'face'
        # name = 'face'
        bbox = obj.find('bndbox')
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text)
            # scale height or width
            # cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
            bndbox.append(cur_pt)
        boxes.append(bndbox)

    return boxes

def _rand_rotate(img,boxes,angle,scale=1.):
    w,h=img.shape[0],img.shape[1]
    rangle=np.deg2rad(angle)

    nw=(abs(np.sin(rangle)*h)+abs(np.cos(rangle)*w))*scale
    nh=(abs(np.cos(rangle)*h)+abs(np.sin(rangle)*w))*scale

    rot_mat=cv2.getRotationMatrix2D((nw*0.5,nh*0.5),angle,scale)
    rot_move=np.dot(rot_mat,np.array([(nw-w)*0.5,(nh-h)*0.5,0]))

    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]

    img=cv2.warpAffine(img,rot_mat,(int(math.ceil(nw)),int(math.ceil(nh))),flags=cv2.INTER_LANCZOS4)

    coord_bboxes = []
    for bbox in boxes:
        xmin, ymin, xmax, ymax = bbox
        point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
        point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
        point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
        point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
        # concat np.array
        concat = np.vstack((point1, point2, point3, point4))
        # change type
        concat = concat.astype(np.int32)
        rx, ry, rw, rh = cv2.boundingRect(concat)

        xmin = rx
        ymin = ry
        xmax = rx + rw
        ymax = ry + ry
        bbox = [xmin, ymin, xmax, ymax]
        coord_bboxes.append(bbox)

    coord_bboxes=np.asarray(coord_bboxes,dtype=np.float32)
    return img,coord_bboxes



def _rand_rotate_vis(img_file,xml_path,angle,scale=1.):
    img=cv2.imread(img_file)
    boxes=parse_xml(xml_path)


    for box in boxes:
        cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),color=(0,255,0),thickness=2)
    # cv2.imshow("hello.png",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("./hello.png",img)
    w,h=img.shape[1],img.shape[0]
    rangle=np.deg2rad(angle)

    nw=(abs(np.sin(rangle)*h)+abs(np.cos(rangle)*w))*scale
    nh=(abs(np.cos(rangle)*h)+abs(np.sin(rangle)*w))*scale


    rot_mat=cv2.getRotationMatrix2D((nw*0.5,nh*0.5),angle,scale)
    rot_move=np.dot(rot_mat,np.array([(nw-w)*0.5,(nh-h)*0.5,0]))

    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]



    img=cv2.warpAffine(img,rot_mat,(int(math.ceil(nw)),int(math.ceil(nh))),flags=cv2.INTER_LANCZOS4)

    coord_bboxes=[]
    for bbox in boxes:
        xmin, ymin, xmax, ymax = bbox
        point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
        point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
        point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
        point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
        # concat np.array
        concat = np.vstack((point1, point2, point3, point4))
        # change type
        concat = concat.astype(np.int32)
        rx, ry, rw, rh = cv2.boundingRect(concat)

        xmin = rx
        ymin = ry
        xmax = rx + rw
        ymax = ry + rh
        bbox = [xmin, ymin, xmax, ymax]
        coord_bboxes.append(bbox)

    coord_bboxes=np.asarray(coord_bboxes,dtype=np.float32)
    for box in coord_bboxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=2)
    cv2.imwrite("./hello_rotate.png",img)

    return img,coord_bboxes
if __name__=="__main__":
    img_file="/home/hp/Data/DOTA-experiment/data/VOCdevkit/VOC2007/JPEGImages/P0000_25.png"
    xml_path="/home/hp/Data/DOTA-experiment/data/VOCdevkit/VOC2007/Annotations/P0000_25.xml"

    img_,boxes=_rand_rotate_vis(img_file,xml_path,angle=350)
