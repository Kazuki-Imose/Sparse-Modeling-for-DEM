# -*- coding: utf-8 -*-
"""
DEM に対して CS（圧縮センシング）を適用(fourier transform)
Written in python3, created by Imose Kazuki on 2018/11/9
"""
import glob
import time

import matplotlib.pyplot as plt
import numpy as np

import compressed_sensing as cs
import visualize_dem


if __name__ == "__main__":
    time1 = time.clock()  # 時間の記録
    # DEM ファイルの指定
    files = glob.glob('E:/20_Compressed_Sensing/files/targetdem_002.txt')
    # DEM の読み込み
    dem = visualize_dem.read_dem(files)
    im = dem
    im_row = im.shape[0]  # DEM の行数
    im_col = im.shape[1]  # DEM の列数
    print('Dimention of DEM : ', im.shape)  # 画像の次元を表示
    # np.savetxt("files/DEM_002_matrix.csv", im, delimiter=',')

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
    Y = im + np.random.randn(im_row, im_col) * sig  # 標準偏差 sig の正規分布ノイズを画像に追加
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
    for mask_num in range(len(deficits)):
        im_zero_masked_temp = im_zero * mask[mask_num]
        im_zero_masked.append(im_zero_masked_temp)

    # 平均値の代入（圧縮センシングの計算のための初期値となる）
    Y_masked_mean = Y_masked.copy()
    for num in range(len(deficits)):
        Y_masked_num = Y_masked[num]
        mask_num = mask[num]
        mean_value = np.mean(Y_masked_num[mask_num == 1])
        print(mean_value)
        Y_masked_mean[num] = np.where(mask_num == 0, mean_value, Y_masked_num)

    """
    圧縮センシング(compressed sensing)の適用
    """
    # パラメータの設定
    n_iter = 10000  # 反復回数
    alpha = 0.0005  # 正則化パラメータ(lambda) 0.0005

    # 圧縮センシングの適用
    recon = []
    for num in range(len(deficits)):  # すべての欠損値に対して適用
        Y_input = Y_masked_mean[num]
        recon_temp, _, _ = cs.cs_fourier(Y_masked_mean[num], mask[num], im, n_iter, alpha)
        recon.append(recon_temp)

    for ii in range(len(deficits)):
        print(cs.get_rmse(im, recon[ii]))

    # np.savetxt("Lenna_after_CS_matrix.csv", recon1, delimiter=',')

    """
    イメージの保存
    """
    fig, ax = plt.subplots(3, 2, figsize=(8, 12))
    ax = ax.flatten()
    ax[0].imshow(im_zero_masked[0], cmap='gray', interpolation='Nearest')
    ax[1].imshow(recon[0], cmap='gray', interpolation='Nearest')
    ax[2].imshow(im_zero_masked[1], cmap='gray', interpolation='Nearest')
    ax[3].imshow(recon[1], cmap='gray', interpolation='Nearest')
    ax[4].imshow(im_zero_masked[2], cmap='gray', interpolation='Nearest')
    ax[5].imshow(recon[2], cmap='gray', interpolation='Nearest')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[3].axis('off')
    ax[4].axis('off')
    ax[5].axis('off')
    ax[0].set_title('deficit 5%')
    ax[1].set_title('reconstructed by COMPRESSED SENSING\nRMSE = {:.6f}'.format(cs.get_rmse(im, recon[0])))
    ax[2].set_title('deficit 10%')
    ax[3].set_title('reconstructed by COMPRESSED SENSING\nRMSE = {:.6f}'.format(cs.get_rmse(im, recon[1])))
    ax[4].set_title('deficit 20%')
    ax[5].set_title('reconstructed by COMPRESSED SENSING\nRMSE = {:.6f}'.format(cs.get_rmse(im, recon[2])))
    # ax[1].set_title('interpolate with mean')
    plt.tight_layout()
    plt.savefig('dem_002_f_nonr_01/DEM_CS_5-20.png', dpi=220)
    plt.clf()

    fig2, ax2 = plt.subplots(3, 2, figsize=(8, 12))
    ax2 = ax2.flatten()
    ax2[0].imshow(im_zero_masked[3], cmap='gray', interpolation='Nearest')
    ax2[1].imshow(recon[3], cmap='gray', interpolation='Nearest')
    ax2[2].imshow(im_zero_masked[4], cmap='gray', interpolation='Nearest')
    ax2[3].imshow(recon[4], cmap='gray', interpolation='Nearest')
    ax2[4].imshow(im_zero_masked[5], cmap='gray', interpolation='Nearest')
    ax2[5].imshow(recon[5], cmap='gray', interpolation='Nearest')
    ax2[0].axis('off')
    ax2[1].axis('off')
    ax2[2].axis('off')
    ax2[3].axis('off')
    ax2[4].axis('off')
    ax2[5].axis('off')
    ax2[0].set_title('deficit 25%')
    ax2[1].set_title('reconstructed by COMPRESSED SENSING\nRMSE = {:.6f}'.format(cs.get_rmse(im, recon[3])))
    ax2[2].set_title('deficit 30%')
    ax2[3].set_title('reconstructed by COMPRESSED SENSING\nRMSE = {:.6f}'.format(cs.get_rmse(im, recon[4])))
    ax2[4].set_title('deficit 40%')
    ax2[5].set_title('reconstructed by COMPRESSED SENSING\nRMSE = {:.6f}'.format(cs.get_rmse(im, recon[5])))
    # ax[1].set_title('interpolate with mean')
    plt.tight_layout()
    plt.savefig('dem_002_f_nonr_01/DEM_CS_25-40.png', dpi=220)
    plt.clf()

    fig3, ax3 = plt.subplots(3, 2, figsize=(8, 12))
    ax3 = ax3.flatten()
    ax3[0].imshow(im_zero_masked[6], cmap='gray', interpolation='Nearest')
    ax3[1].imshow(recon[6], cmap='gray', interpolation='Nearest')
    # ax3[2].imshow(im_zero_masked[7], cmap='gray', interpolation='Nearest')
    # ax3[3].imshow(recon[7], cmap='gray', interpolation='Nearest')
    # ax3[4].imshow(im_zero_masked[8], cmap='gray', interpolation='Nearest')
    # ax3[5].imshow(recon[8], cmap='gray', interpolation='Nearest')
    ax3[0].axis('off')
    ax3[1].axis('off')
    ax3[2].axis('off')
    ax3[3].axis('off')
    ax3[4].axis('off')
    ax3[5].axis('off')
    ax3[0].set_title('deficit 50%')
    ax3[1].set_title('reconstructed by COMPRESSED SENSING\nRMSE = {:.6f}'.format(cs.get_rmse(im, recon[6])))
    # ax3[2].set_title('deficit 60%')
    # ax3[3].set_title('reconstructed by COMPRESSED SENSING\nRMSE = {:.6f}'.format(cs.get_rmse(im, recon[7])))
    # ax3[4].set_title('deficit 70%')
    # ax3[5].set_title('reconstructed by COMPRESSED SENSING\nRMSE = {:.6f}'.format(cs.get_rmse(im, recon[8])))
    # ax[1].set_title('interpolate with mean')
    plt.tight_layout()
    plt.savefig('dem_002_f_nonr_01/DEM_CS_50-70.png', dpi=220)
    plt.clf()

    # fig4, ax4 = plt.subplots(3, 2, figsize=(8, 12))
    # ax4 = ax4.flatten()
    # ax4[0].imshow(im_zero_masked[9], cmap='gray', interpolation='Nearest')
    # ax4[1].imshow(recon[9], cmap='gray', interpolation='Nearest')
    # ax4[2].imshow(im_zero_masked[10], cmap='gray', interpolation='Nearest')
    # ax4[3].imshow(recon[10], cmap='gray', interpolation='Nearest')
    # ax4[4].imshow(im_zero_masked[11], cmap='gray', interpolation='Nearest')
    # ax4[5].imshow(recon[11], cmap='gray', interpolation='Nearest')
    # ax4[0].axis('off')
    # ax4[1].axis('off')
    # ax4[2].axis('off')
    # ax4[3].axis('off')
    # ax4[4].axis('off')
    # ax4[5].axis('off')
    # ax4[0].set_title('deficit 75%')
    # ax4[1].set_title('reconstructed by COMPRESSED SENSING\nRMSE = {:.6f}'.format(cs.get_rmse(im, recon[9])))
    # ax4[2].set_title('deficit 80%')
    # ax4[3].set_title('reconstructed by COMPRESSED SENSING\nRMSE = {:.6f}'.format(cs.get_rmse(im, recon[10])))
    # ax4[4].set_title('deficit 90%')
    # ax4[5].set_title('reconstructed by COMPRESSED SENSING\nRMSE = {:.6f}'.format(cs.get_rmse(im, recon[11])))
    # # ax[1].set_title('interpolate with mean')
    # plt.tight_layout()
    # plt.savefig('dem_002_f_nonr_01/DEM_CS_75-90.png', dpi=220)
    # plt.clf()

    # # RMSE の折れ線グラフを出力
    # fig2, ax2 = plt.subplots(figsize=(8, 12))
    # # ax = ax.flatten()
    # # ax[0].imshow(im, cmap='gray', interpolation='Nearest')
    # left = np.arange(0, n_iter+20, 20)  # 初項0,公差20で終点が n_iter の等差数列
    # ax.plot(left, rmse)
    # # plt.xlabel("the number of iteration")
    # # plt.ylabel("RMSE")
    # # plt.plot(left, rmse)
    # # plt.xlabel("the number of iteration")
    # # plt.ylabel("RMSE")
    # plt.show()
    # plt.savefig('images/figure_rmse_Lenna_recon_by_CS.png', dpi=220)
    #
    # np.savetxt("rmse_by_CS.csv", rmse, delimiter=',')

    time2 = time.clock()
    time_t = int(time2 - time1)
    print("計算時間 : ", time_t, " 秒")
