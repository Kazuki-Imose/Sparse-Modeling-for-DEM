# -*- coding: utf-8 -*-
"""
Implementation of Conventional method(average filter) to DEM
Written in python3, created by Imose Kazuki on 2018/11/13
"""
import glob
import time

import matplotlib.pyplot as plt
import numpy as np

import conventional
import visualize_dem

if __name__ == "__main__":
    time1 = time.clock()
    # DEM ファイルの指定
    # files = glob.glob('E:/CS_DEM/files/targetdem_002.txt')
    files = glob.glob('E:/21_Sparse_Coding/files/targetdem_002.txt')
    # DEM の読み込み
    dem = visualize_dem.read_dem(files)
    im = dem
    im_row = im.shape[0]  # DEM の行数
    im_col = im.shape[1]  # DEM の列数
    print('Dimension of DEM : ', im.shape)  # 画像の次元を表示
    # 最小値をゼロにする
    h_min = np.min(im)
    print(h_min)
    im_zero = im - h_min
    new_h_max = np.max(im_zero)
    print(new_h_max)

    """
    マスク画像の読み込みおよび作成
    """
    sig = 0
    Y = im + np.random.randn(im_col, im_row) * sig  # 標準偏差 sig の正規分布ノイズを画像に追加
    # deficits = [5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]
    deficits = [11, 12, 13, 14, 15, 16, 17]
    mask = []  # mask の格納されるリスト
    Y_masked = []  # mask された DEM の格納されるリスト
    im_zero_masked = []  # mask された im_zero の格納されるリスト
    # mask の読み込み
    for deficit in deficits:
        mask_temp = np.fromfile('files/mask_' + str(deficit), dtype=np.uint8).reshape(im.shape)
        mask.append(mask_temp)
    # DEM のマスク処理
    for mask_num in range(len(deficits)):
        Y_masked_temp = Y * mask[mask_num]
        Y_masked.append(Y_masked_temp)
    # im_zero のマスク処理
    for im_zero_num in range(len(deficits)):
        im_zero_masked_temp = im_zero * mask[im_zero_num]
        im_zero_masked.append(im_zero_masked_temp)

    for i in range(len(deficits)):
        num0_mask = 512 * 512 - np.count_nonzero(mask[i])  # mask 内のゼロの数
        print(num0_mask / 512 / 512)




    # """
    # average_filterの適用
    # """
    # Y_recon_AF = []
    # for i in range(len(deficits)):
    #     Y_recon_AF_temp = conventional.average_filter(Y_masked[i], mask[i])
    #     Y_recon_AF.append(Y_recon_AF_temp)
    #     print(str(i + 1), '/', len(deficits), ' of average filter finished')
    #
    # """
    # イメージの保存
    # """
    # fig, ax = plt.subplots(3, 2, figsize=(8, 12))
    # ax = ax.flatten()
    # ax[0].imshow(im_zero_masked[0], cmap='gray', interpolation='Nearest')
    # ax[1].imshow(Y_recon_AF[0], cmap='gray', interpolation='Nearest')
    # ax[2].imshow(im_zero_masked[1], cmap='gray', interpolation='Nearest')
    # ax[3].imshow(Y_recon_AF[1], cmap='gray', interpolation='Nearest')
    # ax[4].imshow(im_zero_masked[2], cmap='gray', interpolation='Nearest')
    # ax[5].imshow(Y_recon_AF[2], cmap='gray', interpolation='Nearest')
    # ax[0].axis('off')
    # ax[1].axis('off')
    # ax[2].axis('off')
    # ax[3].axis('off')
    # ax[4].axis('off')
    # ax[5].axis('off')
    # ax[0].set_title('deficit 5%')
    # ax[1].set_title('reconstructed by AVERAGE FILTER\nRMSE = {:.6f}'.format(conventional.get_rmse(im, Y_recon_AF[0])))
    # ax[2].set_title('deficit 10%')
    # ax[3].set_title('reconstructed by AVERAGE FILTER\nRMSE = {:.6f}'.format(conventional.get_rmse(im, Y_recon_AF[1])))
    # ax[4].set_title('deficit 20%')
    # ax[5].set_title('reconstructed by AVERAGE FILTER\nRMSE = {:.6f}'.format(conventional.get_rmse(im, Y_recon_AF[2])))
    # # ax[1].set_title('interpolate with mean')
    # plt.tight_layout()
    # # plt.savefig('images_AF/DEM_002/DEM_002_AF_5-20_3.png', dpi=220)
    # plt.savefig('dem_002_av_nonr_01/dem_002_av_5-20.png', dpi=220)
    # plt.clf()
    #
    # fig2, ax2 = plt.subplots(3, 2, figsize=(8, 12))
    # ax2 = ax2.flatten()
    # ax2[0].imshow(im_zero_masked[3], cmap='gray', interpolation='Nearest')
    # ax2[1].imshow(Y_recon_AF[3], cmap='gray', interpolation='Nearest')
    # ax2[2].imshow(im_zero_masked[4], cmap='gray', interpolation='Nearest')
    # ax2[3].imshow(Y_recon_AF[4], cmap='gray', interpolation='Nearest')
    # ax2[4].imshow(im_zero_masked[5], cmap='gray', interpolation='Nearest')
    # ax2[5].imshow(Y_recon_AF[5], cmap='gray', interpolation='Nearest')
    # ax2[0].axis('off')
    # ax2[1].axis('off')
    # ax2[2].axis('off')
    # ax2[3].axis('off')
    # ax2[4].axis('off')
    # ax2[5].axis('off')
    # ax2[0].set_title('deficit 25%')
    # ax2[1].set_title('reconstructed by AVERAGE FILTER\nRMSE = {:.6f}'.format(conventional.get_rmse(im, Y_recon_AF[3])))
    # ax2[2].set_title('deficit 30%')
    # ax2[3].set_title('reconstructed by AVERAGE FILTER\nRMSE = {:.6f}'.format(conventional.get_rmse(im, Y_recon_AF[4])))
    # ax2[4].set_title('deficit 40%')
    # ax2[5].set_title('reconstructed by AVERAGE FILTER\nRMSE = {:.6f}'.format(conventional.get_rmse(im, Y_recon_AF[5])))
    # # ax[1].set_title('interpolate with mean')
    # plt.tight_layout()
    # # plt.savefig('images_AF/DEM_002/DEM_002_AF_25-40_3.png', dpi=220)
    # plt.savefig('dem_002_av_nonr_01/dem_002_av_25-40.png', dpi=220)
    # plt.clf()
    #
    # fig3, ax3 = plt.subplots(3, 2, figsize=(8, 12))
    # ax3 = ax3.flatten()
    # ax3[0].imshow(im_zero_masked[6], cmap='gray', interpolation='Nearest')
    # ax3[1].imshow(Y_recon_AF[6], cmap='gray', interpolation='Nearest')
    # # ax3[2].imshow(im_zero_masked[7], cmap='gray', interpolation='Nearest')
    # # ax3[3].imshow(Y_recon_AF[7], cmap='gray', interpolation='Nearest')
    # # ax3[4].imshow(im_zero_masked[8], cmap='gray', interpolation='Nearest')
    # # ax3[5].imshow(Y_recon_AF[8], cmap='gray', interpolation='Nearest')
    # ax3[0].axis('off')
    # ax3[1].axis('off')
    # ax3[2].axis('off')
    # ax3[3].axis('off')
    # ax3[4].axis('off')
    # ax3[5].axis('off')
    # ax3[0].set_title('deficit 50%')
    # ax3[1].set_title('reconstructed by AVERAGE FILTER\nRMSE = {:.6f}'.format(conventional.get_rmse(im, Y_recon_AF[6])))
    # # ax3[2].set_title('deficit 60%')
    # # ax3[3].set_title('reconstructed by AVERAGE FILTER\nRMSE = {:.6f}'.format(conventional.get_rmse(im, Y_recon_AF[7])))
    # # ax3[4].set_title('deficit 70%')
    # # ax3[5].set_title('reconstructed by AVERAGE FILTER\nRMSE = {:.6f}'.format(conventional.get_rmse(im, Y_recon_AF[8])))
    # # ax[1].set_title('interpolate with mean')
    # plt.tight_layout()
    # # plt.savefig('images_AF/DEM_002/DEM_002_AF_50-70_3.png', dpi=220)
    # plt.savefig('dem_002_av_nonr_01/dem_002_av_50-70.png', dpi=220)
    # plt.clf()
    #
    # # fig4, ax4 = plt.subplots(3, 2, figsize=(8, 12))
    # # ax4 = ax4.flatten()
    # # ax4[0].imshow(im_zero_masked[9], cmap='gray', interpolation='Nearest')
    # # ax4[1].imshow(Y_recon_AF[9], cmap='gray', interpolation='Nearest')
    # # ax4[2].imshow(im_zero_masked[10], cmap='gray', interpolation='Nearest')
    # # ax4[3].imshow(Y_recon_AF[10], cmap='gray', interpolation='Nearest')
    # # ax4[4].imshow(im_zero_masked[11], cmap='gray', interpolation='Nearest')
    # # ax4[5].imshow(Y_recon_AF[11], cmap='gray', interpolation='Nearest')
    # # ax4[0].axis('off')
    # # ax4[1].axis('off')
    # # ax4[2].axis('off')
    # # ax4[3].axis('off')
    # # ax4[4].axis('off')
    # # ax4[5].axis('off')
    # # ax4[0].set_title('deficit 75%')
    # # ax4[1].set_title('reconstructed by AVERAGE FILTER\nRMSE = {:.6f}'.format(conventional.get_rmse(im, Y_recon_AF[9])))
    # # ax4[2].set_title('deficit 80%')
    # # ax4[3].set_title('reconstructed by AVERAGE FILTER\nRMSE = {:.6f}'.format(conventional.get_rmse(im, Y_recon_AF[10])))
    # # ax4[4].set_title('deficit 90%')
    # # ax4[5].set_title('reconstructed by AVERAGE FILTER\nRMSE = {:.6f}'.format(conventional.get_rmse(im, Y_recon_AF[11])))
    # # # ax[1].set_title('interpolate with mean')
    # # plt.tight_layout()
    # # # plt.savefig('images_AF/DEM_002/DEM_002_AF_75-90_3.png', dpi=220)
    # # plt.savefig('dem_002_av_nonr_01/dem_002_av_75-90.png', dpi=220)
    # # plt.clf()
