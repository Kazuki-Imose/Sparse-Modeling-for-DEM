# -*- coding: utf-8 -*-
"""
Implementation of Reconstruction by using TV + DCT to DEM
Written in python3, created by Imose Kazuki on 2018/11/12
"""
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import compressed_sensing as cs
import dictionary_learning as dl
import visualize_dem

if __name__ == "__main__":
    time1 = time.clock()  # 時間の記録
    # 欠損あり DEM ファイルの指定
    dem = pd.read_csv("files/DEM_102.csv", header=None)
    im = np.array(dem)
    im_row = im.shape[0]  # DEM の行数
    im_col = im.shape[1]  # DEM の列数
    print('Dimension of DEM : ', im.shape)  # 画像の次元を表示
    # mask の作成
    # mask = np.where(im == 0, 0, 1)
    # csv ファイルとして保存
    # np.savetxt("dem_102_02/mask_DEM_102.csv", mask, delimiter=',')
    # mask の読み込み
    mask1 = np.array(pd.read_csv("mask_dem/mask_dem_102.csv", header=None))
    # TV 補間後の DEM を読み込む

    """
    TVによる大規模欠損補間
    """
    # # モルフォロジー演算（クロージング）
    # kernel_width = 3  # カーネルサイズを指定 要変更
    # kernel = np.ones((kernel_width, kernel_width), np.uint8)  # カーネルの設定
    # mask_closing = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)  # クロージング処理
    #
    # # TV 最小化適用
    # known_coords = np.where(mask1 == 1)
    # ip = cs.Inpaint(verbose=True, store=False)
    # painted, objective, iterates = ip.transform(im, known_coords)
    # painted = painted.astype(float)
    # # print("Norm of error: {}".format(la.norm(im_origin.flatten() - painted[i].flatten())))
    #
    # # 大規模欠損部の補間
    # im2 = np.where(mask_closing == 0, painted, im)
    #
    # # csv ファイルとして保存
    # np.savetxt("dem_102_02/DEM_102_after_tv_02_3.csv", im2, delimiter=',')
    #
    # # DEM の可視化と保存
    # demT = im2.T
    # visualize_dem.visualize_dem(demT, title="DEM_102", filename="dem_102_02/DEM_102_after_tv_02_3.html")

    # TV 補間後の DEM を読み込む
    im2 = np.array(pd.read_csv("files/DEM_102_after_tv_3.csv", header=None))
    # mask2 の作成
    mask2 = np.where(im2 == 0, 0, 1)

    # 最小値をゼロにする
    h_min = np.min(im2)
    print("本来の標高の最低値：", h_min)
    h_min = h_min - 20
    im2_zero = im2 - h_min
    new_h_max = np.max(im2_zero)
    print("調整後の最高標高値：", new_h_max)

    """
    冗長DCT辞書の作成
    """
    patch_size = 8  # 要変更
    dict_size = 16  # 要変更
    k0 = 4  # k0スパース
    # 辞書の読み込み
    A_DCT = np.fromfile('files/A_DCT').reshape((patch_size ** 2, dict_size ** 2))
    # 辞書の作成
    # A_1D = np.zeros((patch_size, dict_size))
    # for k in np.arange(dict_size):
    #     for i in np.arange(patch_size):
    #         A_1D[i, k] = np.cos(i * k * np.pi / float(dict_size))
    #     if k != 0:
    #         A_1D[:, k] -= A_1D[:, k].mean()
    #
    # A_DCT = np.kron(A_1D, A_1D)
    # A_DCT.tofile('dem_001_dct_04/A_DCT_12_16')
    # 辞書の画像保存
    # dl.show_dictionary(A_DCT, name='dem_001_dct_04/A_DCT_12_16.png')

    """
    スパース符号化（x_hat）
    """
    # q_DCT_dem_01 = dl.sparse_coding_with_mask(im2_zero, A_DCT, k0=k0, sigma=0, mask=mask2, patch_size=patch_size)
    # q_DCT_dem_01.tofile('dem_102_02/DEM_102_q_DCT_3')
    # print('sparse coding finished')
    q_DCT_dem_01 = np.fromfile('dem_102_02/DEM_102_q_DCT_3').reshape((-1, dict_size ** 2))

    """
    再構成
    """
    # Y_recon_DCT_temp = dl.recon_image(im2_zero, q_DCT_dem_01, A_DCT, lam=0, patch_size=patch_size)
    # Y_recon_DCT = Y_recon_DCT_temp + h_min
    # Y_recon_DCT.tofile('dem_102_02/dem_102_recon_Y_DCT_3')
    # print('reconstruction finished')
    Y_recon_DCT = np.fromfile('dem_102_02/dem_102_recon_Y_DCT_3').reshape(im.shape)

    Y_recon_DCT_1 = np.where(mask1 == 0, Y_recon_DCT, im)  # 補間値の更新

    """
    イメージの保存
    """
    # fig, ax = plt.subplots(2, 2, figsize=(8, 12))
    # ax = ax.flatten()
    # ax[0].imshow(im, cmap='gray', interpolation='Nearest')
    # ax[1].imshow(im2, cmap='gray', interpolation='Nearest')
    # ax[2].imshow(Y_recon_DCT, cmap='gray', interpolation='Nearest')
    # # ax[3].imshow(Y_recon_KSVD_1, cmap='gray', interpolation='Nearest')
    # # ax[4].imshow(im_zero_masked[2], cmap='gray', interpolation='Nearest')
    # # ax[5].imshow(recon[2], cmap='gray', interpolation='Nearest')
    # ax[0].axis('off')
    # ax[1].axis('off')
    # ax[2].axis('off')
    # ax[3].axis('off')
    # # ax[4].axis('off')
    # # ax[5].axis('off')
    # ax[0].set_title('deficit')
    # ax[1].set_title('deficit after TV')
    # ax[2].set_title('reconstructed by TV and K-SVD')
    # # ax[3].set_title('reconstructed by K-SVD and TV')
    # # ax[4].set_title('deficit 20%')
    # # ax[5].set_title('reconstructed by COMPRESSED SENSING\nRMSE = {:.6f}'.format(cs.get_rmse(im, recon[2])))
    # # ax[1].set_title('interpolate with mean')
    # plt.tight_layout()
    # plt.savefig('dem_102_02/DEM_102_tv_dct_02_3.png', dpi=220)
    # plt.clf()
    #
    # # DEM の可視化と保存
    # demT = Y_recon_DCT.T
    # visualize_dem.visualize_dem(demT, title="DEM_102", filename="dem_102_02/DEM_102_tv_dct_02_3.html")

    # csv ファイルとして保存
    np.savetxt("dem_102_02/DEM_102_tv_dct_02_3.csv", Y_recon_DCT, delimiter=',')

    time2 = time.clock()
    time_t = int(time2 - time1)
    print("計算時間 : ", time_t, " 秒")
