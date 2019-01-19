# -*- coding: utf-8 -*-
"""
DEM に対して CS（圧縮センシング）を適用(fourier transform)
Written in python3, created by Imose Kazuki on 2018/11/9
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import compressed_sensing as cs
import visualize_dem

if __name__ == "__main__":
    time1 = time.clock()  # 時間の記録
    # DEM ファイルの指定
    dem = pd.read_csv("files/DEM_102.csv", header=None)
    im = np.array(dem)
    im_row = im.shape[0]  # DEM の行数
    im_col = im.shape[1]  # DEM の列数
    print('Dimention of DEM : ', im.shape)  # DEM の次元を表示
    # np.savetxt("files/DEM_002_matrix.csv", im, delimiter=',')
    # mask の作成
    mask = np.where(im == 0, 0, 1)

    # 最小値をゼロにする
    h_min = np.min(im)
    print(h_min)
    h_min = h_min - 20  # 最小値を低めに設定
    im_zero = im - h_min
    new_h_max = np.max(im_zero)
    print(new_h_max)

    # 初期配列を作成（圧縮センシングの計算のための初期値となる，今回はTV最小化の結果を用いる）
    # df_init = pd.read_csv("files/dem_102_start.csv", header=None)
    # dem_init = np.array(df_init)
    # dem_init = dem_init - h_min

    # 平均値の代入
    mean_value = np.mean(im_zero[mask == 1])
    print("平均標高：", mean_value)
    dem_init = np.where(mask == 0, mean_value, im_zero)

    """
    圧縮センシング(compressed sensing)の適用
    """
    # パラメータの設定
    n_iter = 500  # 反復回数
    alpha = 0.0005  # 正則化パラメータ(lambda) 0.0005

    # 圧縮センシングの適用
    recon, recon1, _ = cs.cs_fourier(dem_init, mask, im_zero, n_iter, alpha)
    recon = recon + h_min
    recon1 = recon1 + h_min

    # np.savetxt("Lenna_after_CS_matrix.csv", recon1, delimiter=',')

    """
    イメージの保存
    """
    fig, ax = plt.subplots(2, 2, figsize=(8, 12))
    ax = ax.flatten()
    ax[0].imshow(im, cmap='gray', interpolation='Nearest')
    ax[1].imshow(recon, cmap='gray', interpolation='Nearest')
    ax[2].imshow(recon1, cmap='gray', interpolation='Nearest')
    # ax[3].imshow(recon[1], cmap='gray', interpolation='Nearest')
    # ax[4].imshow(im_zero_masked[2], cmap='gray', interpolation='Nearest')
    # ax[5].imshow(recon[2], cmap='gray', interpolation='Nearest')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    # ax[3].axis('off')
    # ax[4].axis('off')
    # ax[5].axis('off')
    ax[0].set_title('deficit')
    ax[1].set_title('reconstructed by COMPRESSED SENSING')
    ax[2].set_title('reconstructed by COMPRESSED SENSING')
    # ax[3].set_title('reconstructed by COMPRESSED SENSING\nRMSE = {:.6f}'.format(cs.get_rmse(im, recon[1])))
    # ax[4].set_title('deficit 20%')
    # ax[5].set_title('reconstructed by COMPRESSED SENSING\nRMSE = {:.6f}'.format(cs.get_rmse(im, recon[2])))
    # ax[1].set_title('interpolate with mean')
    plt.tight_layout()
    plt.savefig('dem_102_f_01/DEM_102_cs_f_01.png', dpi=220)
    plt.clf()

    # DEM の可視化と保存
    demT = recon.T
    visualize_dem.visualize_dem(demT, title="DEM_102", filename="dem_102_f_01/dem_102_f_01.html")
    dem1T = recon1.T
    visualize_dem.visualize_dem(dem1T, title="DEM_102", filename="dem_102_f_01/dem_102_f_01_1.html")

    # csv ファイルとして保存
    np.savetxt("dem_102_f_01/dem_102_f_01.csv", recon, delimiter=',')
    np.savetxt("dem_102_f_01/dem_102_f_01_1.csv", recon1, delimiter=',')

    time2 = time.clock()
    time_t = int(time2 - time1)
    print("計算時間 : ", time_t, " 秒")
