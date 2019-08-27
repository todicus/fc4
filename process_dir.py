import json
import os
import sys
import os.path as osp
from glob import glob

import cv2
import numpy as np
import tensorflow as tf

from fcn import FCN
from utils import get_session
from summary_utils import get_visualization
from config import *


def test_external(self, imgpaths, scale=1.0, show=True, save_prefix="cc_outputs", write=False, write_compare=False):
    illums = []
    confidence_maps = []
    errors = []
    for i, imgp in enumerate(imgpaths):
        filename = osp.basename(imgp)+".jpg"  # force save as jpg
        print ""
        print i, filename
        
        try:
            img = cv2.imread(imgp)
        except Exception as e:
            print("ERROR can't read image {}\n{}".format(filename, e))
            errors.append(imgp)
            continue
        
        if img is None:
            print("ERROR read None for {}".format(filename))
            errors.append(imgp)
            continue
        
        # reverse gamma correction for sRGB
        img_linear = (img / 255.0) ** 2.2 * 65536
        
        if scale != 1.0:
            img_linear = cv2.resize(img_linear, (0, 0), fx=scale, fy=scale)
        
        shape = img_linear.shape[:2]    
        if shape not in self.test_nets:
            #print("shape {}".format(shape))
            
            aspect_ratio = 1.0 * shape[1] / shape[0]
            if aspect_ratio < 1:
                target_shape = (MERGED_IMAGE_SIZE, MERGED_IMAGE_SIZE * aspect_ratio)
            else:
                target_shape = (MERGED_IMAGE_SIZE / aspect_ratio, MERGED_IMAGE_SIZE)
            
            target_shape = tuple(map(int, target_shape))

            test_net = {}
            test_net['illums'] = tf.placeholder(
                tf.float32, shape=(None, 3), name='test_illums')
            test_net['images'] = tf.placeholder(
                tf.float32, shape=(None, shape[0], shape[1], 3), name='test_images')
            with tf.variable_scope("FCN", reuse=True):
                try:
                    test_net['pixels'] = FCN.build_branches(test_net['images'], 1.0)
                except ValueError as e:
                    print("ERROR {}".format(e))
                    errors.append(imgp)
                    continue

                test_net['est'] = tf.reduce_sum(test_net['pixels'], axis=(1, 2))
            test_net['merged'] = get_visualization(
                test_net['images'], test_net['pixels'], test_net['est'],
                test_net['illums'], target_shape)
            self.test_nets[shape] = test_net
            
        test_net = self.test_nets[shape]

        pixels, est, merged = self.sess.run(
            [test_net['pixels'], test_net['est'], test_net['merged']],
            feed_dict={
                test_net['images']: img_linear[None, :, :, :],
                test_net['illums']: [[1, 1, 1]]
            }
        )
        est = est[0]
        est /= np.linalg.norm(est)

        pixels = pixels[0]
        confidences = np.linalg.norm(pixels, axis=2)
        #confidence_maps.append(confidences)
        ind = int(confidences.flatten().shape[0] * 0.95)
        print("Confidence: {:0.1f} mean, {:0.1f} max, {:0.1f} 95%".format(
            confidences.mean(),
            confidences.max(),
            sorted(confidences.flatten())[ind]))
        merged = merged[0]
        #illums.append(est)

        if show:
            cv2.imshow('Ret', merged[:, :, ::-1])
            k = cv2.waitKey(0) % (2**20)

        try:
            os.makedirs(save_prefix)
        except:
            pass

        corrected = np.power(img_linear[:,:,::-1] / 65535 / est[None, None, :] * np.mean(est), 1/2.2)[:,:,::-1]
        if show:
            cv2.imshow("corrected", corrected)
            cv2.waitKey(0)

        corrected = corrected * 255.0
        output_img = corrected
        if write_compare:
            orig_img = img #np.power(img / 65535, 1/2.2) * 255
            output_img = np.concatenate((orig_img, corrected), axis=1)

        cv2.imwrite(osp.join(save_prefix, 'corrected_%s' % filename), output_img, )

    return errors

if __name__ == '__main__':
    target_dir = sys.argv[1]

    if len(sys.argv) > 2:
        start_ind = int(sys.argv[2])
    else:
        start_ind = 0

    image_dir = osp.join(target_dir, "*")
    print(image_dir)
    imgpaths = glob(image_dir)[start_ind:]
    print(len(imgpaths))


    save_dir = osp.join(osp.dirname(target_dir), osp.basename(target_dir)+"_colorcorrected")
    print(save_dir)

    ckpt = "pretrained/colorchecker_fold1and2.ckpt"
    with get_session() as sess:
        fcn = FCN(sess=sess, name=ckpt)
        fcn.load_absolute(ckpt)
        errors = test_external(fcn, imgpaths=imgpaths, save_prefix=save_dir, show=False, write_compare=False)

    with open(osp.join(target_dir, "errored_images.txt"), "w") as f:
        for p in errors:
            f.write(p)
    
