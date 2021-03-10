import numpy as np
import os
import os.path as osp
import cv2

def vis_result(imn, imt, ant, pred, save_dir, n_class=11):  
    img = gray2rgbimage(imt)
    imn = imn[0]
    pred_img = draw_img(imt, pred, n_class=n_class)
    if(ant is None):
        cv2.imwrite(osp.join(save_dir, imn), np.hstack((img, pred_img)).astype('uint8'))
    else:
        ant_img = draw_img(imt, ant, n_class=n_class)
        cv2.imwrite(osp.join(save_dir, imn), np.hstack((img,ant_img,pred_img )).astype('uint8'))
        cv2.imwrite(osp.join(save_dir, 'label/' + imn), ant_img.astype('uint8'))
        cv2.imwrite(osp.join(save_dir, 'pred/' + imn), pred_img.astype('uint8'))

def draw_img(img, seg, title = None, n_class=11):
    mask = img
    label_set = [i+1 for i in range(n_class)]
    color_set = {
                    #0:(200,200,200),
                    1:(255, 0, 0), #BGR   #NFL
                    2:(0, 255, 0),   #GCL
                    3:(0, 0, 255),   #IPL
                    4:(0, 255, 255),  #INL
                    5: (255, 0, 255), #OPL
                    6: (255, 255, 0),  #ONL
                    7: (0, 0, 150),  #IS/OS
                    8: (0, 150, 0),  #RPE
                    9: (150, 0, 150), #choroid
                    10: (100, 50, 250),  # choroid
                    11: (50, 100, 250),  # choroid
                }

    mask = gray2rgbimage(mask)
    img = gray2rgbimage(img)
    if(title is not None):
        mask = cv2.putText(mask, title, (16, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # white title
    for draw_label in label_set:
        mask[:, :, 0][seg[0,:,:] == draw_label] =(color_set[draw_label][0])
        mask[:, :, 1][seg[0,:,:] == draw_label] = ( color_set[draw_label][1])
        mask[:, :, 2][seg[0,:,:] == draw_label] = (color_set[draw_label][2])
    img_mask = cv2.addWeighted(img,0.4,mask,0.6,0)
    return img_mask


def gray2rgbimage(image):
    a,b = image.shape
    new_img = np.ones((a,b,3))
    new_img[:,:,0] = image.reshape((a,b)).astype('uint8')
    new_img[:,:,1] = image.reshape((a,b)).astype('uint8')
    new_img[:,:,2] = image.reshape((a,b)).astype('uint8')
    return new_img








