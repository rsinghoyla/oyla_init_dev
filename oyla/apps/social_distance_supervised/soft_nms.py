# -*- coding:utf-8 -*-
# Author:Richard Fang
"""
This is a Python version used to implement the Soft NMS algorithm.
Original Paper：Improving Object Detection With One Line of Code
"""
import numpy as np
#import tensorflow as tf
#from keras import backend as K
import time


def py_cpu_softnms(dets, sc, Nt=0.2, sigma=0.5, thresh=0.2, method='nms', depth = None):
    """
    py_cpu_softnms
    :param dets:   boexs 坐标矩阵 format [y1, x1, y2, x2]
    :param sc:     每个 boxes 对应的分数
    :param Nt:     iou 交叠门限
    :param sigma:  使用 gaussian 函数的方差
    :param thresh: 最后的分数门限
    :param method: 使用的方法
    :return:       留下的 boxes 的 index
    """

    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    # the order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = sc
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    depth_means = np.zeros((N,1))
    for i in range(N):
        depth_means[i] = np.average(depth[int(dets[i][1]):int(dets[i][3]),int(dets[i][0]):int(dets[i][2])])

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        tdepth_mean = depth_means[i].copy()
        pos = i + 1
        #d_i_mean=np.average(depth[int(dets[i][0]):int(dets[i][0]+dets[i][2]),int(dets[i][1]):int(dets[i][1]+dets[i][3])])
        #d_i_mean = np.average(depth[int(dets[i][1]):int(dets[i][3]),int(dets[i][0]):int(dets[i][2])])
        #print(i,d_i_mean)
        #
        
        if i != N-1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

            depth_means[i] = depth_means[maxpos +i +1]
            depth_means[maxpos + i + 1] = tdepth_mean
            tdepth_mean = depth_means[i]

        # IoU calculate
        xx1 = np.maximum(dets[i, 1], dets[pos:, 1])
        yy1 = np.maximum(dets[i, 0], dets[pos:, 0])
        xx2 = np.minimum(dets[i, 3], dets[pos:, 3])
        yy2 = np.minimum(dets[i, 2], dets[pos:, 2])

        diff = np.abs(depth_means[i]-depth_means[pos:])
        dmax = np.maximum(depth_means[i], depth_means[pos:])
        bev_iou = diff/(dmax+0.00001)
        try:
            bev_iou = np.squeeze(bev_iou,axis=1)
        except:
            pass
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)
        
        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 'soft_nms_d':  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = (weight[ovr > Nt] - ovr[ovr > Nt])*(weight[ovr>Nt]-bev_iou[ovr>Nt])
        elif method == 'soft_nms':  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = (weight[ovr > Nt] - ovr[ovr > Nt])
        elif method == 'soft_nms_g':  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        elif method=='nms':  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0
        else:
            print(method)

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    print(scores)
    inds = dets[:, 4][scores > thresh]
    keep = inds.astype(int)
    print(len(keep))
    return keep


def speed():
    boxes = 1000*np.random.rand((1000, 100, 4))
    boxscores = np.random.rand((1000, 100))

    start = time.time()
    for i in range(1000):
        py_cpu_softnms(boxes[i], boxscores[i], method=2)
    end = time.time()
    print("Average run time: %f ms" % (end-start))


def test():
    # boxes and scores
    boxes = np.array([[200, 200, 400, 400], [220, 220, 420, 420], [200, 240, 400, 440], [240, 200, 440, 400], [1, 1, 2, 2]], dtype=np.float32)
    boxscores = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)

    # tf.image.non_max_suppression 中 boxes 是 [y1,x1,y2,x2] 排序的。
    # with tf.Session() as sess:
    #     # index = sess.run(tf.image.non_max_suppression(boxes=boxes, scores=boxscores, iou_threshold=0.5, max_output_size=5))
    #     # print(index)
    #     index = py_cpu_softnms(boxes, boxscores, method=3)
    #     selected_boxes = sess.run(K.gather(boxes, index))
    #     print(selected_boxes)


if __name__ == '__main__':
    test()
    # speed()

