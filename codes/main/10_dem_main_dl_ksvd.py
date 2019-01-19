# -*- coding: utf-8 -*-
"""
Implementation of Dictionary Learning(K-SVD) to DEM
Written in python3, created by Imose Kazuki on 2018/11/12
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dictionary_learning as dl
import visualize_dem

if __name__ == "__main__":
    time1 = time.clock()  # 時間の記録
    # DEM ファイルの指定
    dem = pd.read_csv("files/DEM_102.csv", header=None)
    # dem_true = np.array(pd.read_csv("files/DEM_102.csv", header=None))
    im = np.array(dem)
    im_row = im.shape[0]  # DEM の行数
    im_col = im.shape[1]  # DEM の列数
    print('Dimension of DEM : ', im.shape)  # 画像の次元を表示
    # mask の作成
    mask = np.where(im == 0, 0, 1)

    # 最小値をゼロにする
    h_min = np.min(im)
    print("本来の標高の最低値：", h_min)
    h_min = h_min - 20
    im_zero = im - h_min
    new_h_max = np.max(im_zero)
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
    # A_DCT.tofile('dem_001_ksvd_04/A_DCT_12_16')
    # 辞書の画像保存
    # dl.show_dictionary(A_DCT, name='dem_001_ksvd_04/A_DCT_12_16.png')

    """
    辞書学習の実行
    """
    A_KSVD = dl.dictionary_learning_with_mask(im_zero, mask, A_DCT, patch_size=patch_size, dict_size=dict_size, k0=k0)
    A_KSVD.tofile('dem_102_01/ksvd/A_KSVD_102')
    dl.show_dictionary(A_KSVD, name='dem_102_01/ksvd/A_KSVD_102.png')
    print('dictionary learning finished')
    # A_KSVD = np.fromfile('dem_102_01/ksvd/A_KSVD_102').reshape((patch_size ** 2, dict_size ** 2))

    """
    スパース符号化（x_hat）
    """
    q_KSVD = dl.sparse_coding_with_mask(im_zero, A_KSVD, k0, sigma=0, mask=mask, patch_size=patch_size)
    q_KSVD.tofile('dem_102_01/ksvd/DEM_102_q_KSVD')
    print('sparse coding finished')
    # q_KSVD = np.fromfile('dem_102_01/ksvd/DEM_102_q_KSVD').reshape((-1, dict_size ** 2))

    """
    画像再構成
    """
    Y_recon_KSVD_temp = dl.recon_image(im_zero, q_KSVD, A_KSVD, lam=0, patch_size=patch_size)
    Y_recon_KSVD = Y_recon_KSVD_temp + h_min
    Y_recon_KSVD.tofile('dem_102_01/ksvd/dem_102_recon_Y_KSVD')
    print('DEM reconstruction finished')
    # Y_recon_KSVD_temp = np.fromfile('dem_102_01/ksvd/dem_102_recon_Y_KSVD').reshape(im.shape)

    Y_recon_KSVD_1 = np.where(mask == 0, Y_recon_KSVD, im)  # 補間値の更新

    """
    再構成画像の保存
    """
    fig, ax = plt.subplots(2, 1, figsize=(8, 12))
    ax = ax.flatten()
    ax[0].imshow(im_zero, cmap='gray', interpolation='Nearest')
    ax[1].imshow(Y_recon_KSVD, cmap='gray', interpolation='Nearest')
    # ax[2].imshow(Y_masked[1], cmap='gray', interpolation='Nearest')
    # ax[3].imshow(Y_recon_DCT[1], cmap='gray', interpolation='Nearest')
    # ax[4].imshow(Y_masked[2], cmap='gray', interpolation='Nearest')
    # ax[5].imshow(Y_recon_DCT[2], cmap='gray', interpolation='Nearest')
    ax[0].axis('off')
    ax[1].axis('off')
    # ax[2].axis('off')
    # ax[3].axis('off')
    # ax[4].axis('off')
    # ax[5].axis('off')
    ax[0].set_title('deficit')
    ax[1].set_title('reconstructed by K-SVD')
    # ax[0].set_title('deficit\n{:.6f}'.format(dl.get_rmse(dem_true, Y_recon_DCT)))
    # ax[1].set_title('reconstructed by DCT\n{:.6f}'.format(dl.get_rmse(dem_true, Y_recon_DCT_1)))
    # ax[2].set_title('deficit 10%')
    # ax[3].set_title('reconstructed by K-SVD\n{:.6f}'.format(dl.get_rmse(im_zero, Y_recon_DCT[1])))
    # ax[4].set_title('deficit 20%')
    # ax[5].set_title('reconstructed by K-SVD\n{:.6f}'.format(dl.get_rmse(im_zero, Y_recon_DCT[2])))
    plt.tight_layout()
    plt.savefig('dem_102_01/ksvd/dem_102_recon_KSVD.png', dpi=220)
    plt.clf()

    # DEM の可視化と保存
    demT = Y_recon_KSVD.T
    visualize_dem.visualize_dem(demT, title="DEM_102", filename="dem_102_01/ksvd/dem_102_ksvd.html")
    # DEM の可視化と保存
    dem1T = Y_recon_KSVD_1.T
    visualize_dem.visualize_dem(dem1T, title="DEM_102", filename="dem_102_01/ksvd/dem_102_ksvd_1.html")

    # csv ファイルとして保存
    np.savetxt("dem_102_01/ksvd/dem_102_ksvd.csv", Y_recon_KSVD, delimiter=',')
    # csv ファイルとして保存
    np.savetxt("dem_102_01/ksvd/dem_102_ksvd_1.csv", Y_recon_KSVD_1, delimiter=',')

    time2 = time.clock()
    time_t = int(time2 - time1)
    print("計算時間 : ", time_t, " 秒")
