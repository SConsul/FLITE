import argparse
import numpy as np
import os
import json
import cv2


def main():
    '''
    The data folder (containing the image and bbox folders) is assumed to be outside (in the same level) 
    as the repository folder, if not explicitly mention the folder paths
    
    example terminal command: python bbox_visualisation.py --mode train --user P100 --_class front\ door 

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, type=str, help="train, val or test")
    parser.add_argument("--data_folder_path", type=str, default="../../data/orbit_benchmark_224", help="root location of ORBIT image dataset")
    parser.add_argument("--bbox_folder_path", type=str, default= "../../data/orbit_clutter_bounding_boxes", help="root location of ORBIT bounding boxes")
    parser.add_argument("--user", required=True, type=str, help="ID of user, eg P100")
    parser.add_argument("--_class", default="exercise bench", type=str, help="class of object to be observed")
    args = parser.parse_args()

    data_folder_path = os.path.join(args.data_folder_path,args.mode+"/")
    bbox_folder_path = os.path.join(args.bbox_folder_path,args.mode+"_annotations/")

    img_folder = data_folder_path+args.user+'/' + args._class+"/clutter/"
    img_folder = os.path.join(img_folder,os.listdir(img_folder)[-1])


    img_list = os.listdir(img_folder)
    vid_clip = img_list[0][:-10]
    bbox_json = vid_clip+".json"
    with open(os.path.join(bbox_folder_path,bbox_json), 'r') as bbox_file:
        bbox_data=bbox_file.read()
        
    bbox = json.loads(bbox_data)

    for frame in img_list:
        img_path = os.path.join(img_folder,frame)

        cx,cy,w,h = bbox[frame]['object_bounding_box']

        x1 = int((cx-w/2)/1080*224)
        x2 = int((cx+w/2)/1080*224)
        y1 = int((cy-h/2)/1080*224)
        y2 = int((cy+h/2)/1080*224)
        pt1 = (x1,y1)
        pt2 = (x2,y2)

        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        im = cv2.rectangle(img, pt1, pt2, (0,255,0), thickness=2)
        cv2.imshow(args.user+" "+args._class,im)
        k = cv2.waitKey(300)
        if k == 27:         # If escape was pressed exit
            cv2.destroyAllWindows()
            break
if __name__ == "__main__":
    main()