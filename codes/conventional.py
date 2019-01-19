# -*- coding: utf-8 -*-
"""
従来のDEM作成手法のための module
Written in python3, created by Imose Kazuki on 2018/11/13
"""
from matplotlib.font_manager import FontProperties
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import time
import glob


def get_psnr(im, recon):
    """ PSNRを得る """
    # return 10. * np.log(im.max() / np.sqrt(np.mean((im - recon) ** 2)))
    return 20. * np.log10(im.max() / np.sqrt(np.mean((im - recon) ** 2)))


def get_rmse(im, recon):
    """ RMSEを得る """
    # return 10. * np.log(im.max() / np.sqrt(np.mean((im - recon) ** 2)))
    return np.sqrt(np.mean((im - recon) ** 2))


def average_filter_previous(im, mask):
    """ average_filter """
    # 欠損箇所に周辺8セルの平均値を代入
    col = im.shape[0]
    row = im.shape[1]
    zerosv = np.zeros((1, row))
    zerosh = np.zeros((col + 2, 1))
    mask_temp = np.vstack((zerosv, mask, zerosv))
    mask_temp = np.hstack((zerosh, mask_temp, zerosh))
    im_temp = np.vstack((zerosv, im, zerosv))
    im_temp = np.hstack((zerosh, im_temp, zerosh))

    Z = np.empty((col, row))

    for i in range(1, col + 1):
        for j in range(1, row + 1):
            if mask_temp[i, j] == 1:
                Z[i - 1, j - 1] = im_temp[i, j]
            else:
                n = mask_temp[i - 1, j - 1] + mask_temp[i, j - 1] + mask_temp[i + 1, j - 1] + mask_temp[
                    i - 1, j] + mask_temp[i + 1, j] + mask_temp[i - 1, j + 1] + mask_temp[i, j + 1] + \
                    mask_temp[i + 1, j + 1]
                if n == 0:
                    Z[i - 1, j - 1] = 0
                else:
                    Z[i - 1, j - 1] = (im_temp[i - 1, j - 1] + im_temp[i, j - 1] + im_temp[i + 1, j - 1] + im_temp[
                        i - 1, j] + im_temp[i + 1, j] + im_temp[i - 1, j + 1] + im_temp[i, j + 1] + im_temp[
                                           i + 1, j + 1]) / n
    iter_num = 0
    while Z.min() == 0:
        iter_num += 1
        print(iter_num)
        im_tem = np.vstack((zerosv, Z, zerosv))
        im_tem = np.hstack((zerosh, im_tem, zerosh))
        for ii in range(0, col):
            for jj in range(0, row):
                if Z[ii, jj] == 0:
                    A = im_tem[ii:(ii + 2), jj:(jj + 2)]
                    if np.count_nonzero(A) != 0:
                        n = np.sum(A) / np.count_nonzero(A)
                        Z[ii, jj] = n
        if iter_num > 20:
            break
    return Z


def average_filter(im, mask):
    """ average_filter """
    """
    検索範囲内においてデータが取れていないセルがある場合は，検索範囲を広げる
    """
    # 欠損箇所に周辺8セルの平均値を代入
    distance = 1  # ターゲットセルから平均対象の距離
    row = im.shape[0]
    col = im.shape[1]
    # 結果を格納する配列を準備
    Z = np.zeros((row, col))
    num = row * col - np.count_nonzero(Z)  # Z 内のゼロの数
    num0_mask = row * col - np.count_nonzero(mask)  # mask 内のゼロの数
    num0_im = row * col - np.count_nonzero(im)  # im 内のゼロの数
    num_last = num0_im - num0_mask
    print("ゼロ要素の数：", num)  # 0 要素の数
    while num > num_last:
        print("ターゲットセルから平均対象の距離：", distance)
        zerosv = np.zeros((distance, col))
        zerosh = np.zeros((row + 2 * distance, distance))
        mask_temp = np.vstack((zerosv, mask, zerosv))
        mask_temp = np.hstack((zerosh, mask_temp, zerosh))
        im_temp = np.vstack((zerosv, im, zerosv))
        im_temp = np.hstack((zerosh, im_temp, zerosh))
        for i in range(0, row):
            for j in range(0, col):
                if mask_temp[(i + distance), (j + distance)] == 1:
                    Z[i, j] = im[i, j]
                else:
                    A = im_temp[i:(i + 2 * distance), j:(j + 2 * distance)]
                    if np.count_nonzero(A) != 0:
                        n = np.sum(A) / np.count_nonzero(A)
                        Z[i, j] = n
        num = row * col - np.count_nonzero(Z)
        print("ゼロ要素の数：", num)  # 0 要素の数
        distance += 1
    return Z
