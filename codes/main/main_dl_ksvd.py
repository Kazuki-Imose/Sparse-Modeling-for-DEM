# -*- coding: utf-8 -*-
"""
Implementation of Dictionary Learning(K-SVD) to DEM
Written in python3, created by Imose Kazuki on 2018/11/12
"""
import glob
import time

import matplotlib.pyplot as plt
import numpy as np

import dictionary_learning as dl
import visualize_dem


if __name__ == "__main__":
    time1 = time.clock()  # 時間の記録
    # DEM ファイルの指定
    files = glob.glob('E:/21_Sparse_Coding/files/targetdem_002.txt')
    # files = glob.glob('C:/Users/kazuk/Desktop/K-SVD_try/files/targetdem_002.txt')
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
    Y = im_zero + np.random.randn(im_col, im_row) * sig  # 標準偏差 sig の正規分布ノイズを画像に追加
    # deficits = [5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]
    deficits = [5]
    # deficits = [11, 12, 13, 14, 15, 16, 17]
    mask = []  # mask の格納されるリスト
    Y_masked = []  # mask された DEM の格納されるリスト
    im_zero_masked = []  # mask された im_zero の格納されるリスト
    # mask の読み込み
    for deficit in deficits:
        mask_temp1 = np.fromfile('files/mask_' + str(deficit), dtype=np.uint8).reshape(im.shape)
        mask.append(mask_temp1)
    # DEM のマスク処理
    for mask_num in range(len(deficits)):
        Y_masked_temp1 = Y * mask[mask_num]
        Y_masked.append(Y_masked_temp1)

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
    # A_DCT.tofile('dem_001_ksvd_07/A_DCT_12_16')
    # 辞書の画像保存
    # dl.show_dictionary(A_DCT, name='dem_002_ksvd_07/A_DCT_12_16.png')

    """
    辞書学習の実行
    """
    A_KSVD = []
    for i in range(len(deficits)):
        A_KSVD_temp1 = dl.dictionary_learning_with_mask(Y_masked[i], mask[i], A_DCT, patch_size=patch_size,
                                                       dict_size=dict_size, k0=k0)
        A_KSVD.append(A_KSVD_temp1)
        A_KSVD[i].tofile('temp1/A_KSVD_' + str(deficits[i]))
        dl.show_dictionary(A_KSVD[i], name='temp1/A_KSVD_' + str(deficits[i]) + '.png')
        print(str(i + 1), '/', len(deficits), ' of dictionary learning finished')
        # A_KSVD_temp1 = np.fromfile('dem_001_ksvd_nonr_01/A_KSVD_' + str(deficits[i])).reshape((patch_size ** 2, dict_size ** 2))
        # A_KSVD.append(A_KSVD_temp1)

    """
    スパース符号化（x_hat）
    """
    q_KSVD = []
    for i in range(len(deficits)):
        q_KSVD_temp1 = dl.sparse_coding_with_mask(Y_masked[i], A_KSVD[i], k0, sigma=0, mask=mask[i],
                                                 patch_size=patch_size)
        q_KSVD.append(q_KSVD_temp1)
        q_KSVD[i].tofile('temp1/DEM_002_q_' + str(deficits[i]) + '_KSVD')
        print(str(i + 1), '/', len(deficits), ' of sparse coding finished')
        # q_KSVD_temp1 = np.fromfile('temp1/DEM_002_q_' + str(deficits[i]) + '_KSVD').reshape((-1, dict_size ** 2))
        # q_KSVD.append(q_KSVD_temp1)

    """
    画像再構成
    """
    Y_recon_KSVD = []
    for i in range(len(deficits)):
        Y_recon_KSVD_temp1 = dl.recon_image(Y_masked[i], q_KSVD[i], A_KSVD[i], lam=0, patch_size=patch_size)
        Y_recon_KSVD.append(Y_recon_KSVD_temp1)
        Y_recon_KSVD[i].tofile('temp1/dem_recon_Y_' + str(deficits[i]) + '_KSVD')
        print(str(i + 1), '/', len(deficits), ' of image reconstruction finished')
        # Y_recon_KSVD_temp1 = np.fromfile('dem_001_ksvd_nonr_01/dem_recon_Y_sig20_' + str(deficits[i]) + '_KSVD').reshape(im.shape)
        # Y_recon_KSVD.append(Y_recon_KSVD_temp1)

    Y_recon_KSVD_1 = []
    for i in range(len(deficits)):
        Y_recon_KSVD_1_temp1 = np.where(mask[i] == 0, Y_recon_KSVD[i], Y_masked[i])  # 補間値の更新
        Y_recon_KSVD_1.append(Y_recon_KSVD_1_temp1)

    for ii in range(7):
        print(dl.get_rmse(im_zero, Y_recon_KSVD[ii]))
    for ii in range(7):
        print(dl.get_rmse(im_zero, Y_recon_KSVD_1[ii]))

    """
    再構成画像の保存
    """
    fig, ax = plt.subplots(3, 2, figsize=(8, 12))
    ax = ax.flatten()
    ax[0].imshow(Y_masked[0], cmap='gray', interpolation='Nearest')
    ax[1].imshow(Y_recon_KSVD[0], cmap='gray', interpolation='Nearest')
    # ax[2].imshow(Y_masked[1], cmap='gray', interpolation='Nearest')
    # ax[3].imshow(Y_recon_KSVD[1], cmap='gray', interpolation='Nearest')
    # ax[4].imshow(Y_masked[2], cmap='gray', interpolation='Nearest')
    # ax[5].imshow(Y_recon_KSVD[2], cmap='gray', interpolation='Nearest')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[3].axis('off')
    ax[4].axis('off')
    ax[5].axis('off')
    ax[0].set_title('deficit 5%')
    ax[1].set_title('reconstructed by K-SVD\n{:.6f}'.format(dl.get_rmse(im_zero, Y_recon_KSVD[0])))
    # ax[2].set_title('deficit 10%')
    # ax[3].set_title('reconstructed by K-SVD\n{:.6f}'.format(dl.get_rmse(im_zero, Y_recon_KSVD[1])))
    # ax[4].set_title('deficit 20%')
    # ax[5].set_title('reconstructed by K-SVD\n{:.6f}'.format(dl.get_rmse(im_zero, Y_recon_KSVD[2])))
    plt.tight_layout()
    plt.savefig('temp1/hosei_nashi/dem_recon_KSVD01.png', dpi=220)
    plt.clf()

    # fig2, ax2 = plt.subplots(3, 2, figsize=(8, 12))
    # ax2 = ax2.flatten()
    # ax2[0].imshow(Y_masked[3], cmap='gray', interpolation='Nearest')
    # ax2[1].imshow(Y_recon_KSVD[3], cmap='gray', interpolation='Nearest')
    # ax2[2].imshow(Y_masked[4], cmap='gray', interpolation='Nearest')
    # ax2[3].imshow(Y_recon_KSVD[4], cmap='gray', interpolation='Nearest')
    # ax2[4].imshow(Y_masked[5], cmap='gray', interpolation='Nearest')
    # ax2[5].imshow(Y_recon_KSVD[5], cmap='gray', interpolation='Nearest')
    # ax2[0].axis('off')
    # ax2[1].axis('off')
    # ax2[2].axis('off')
    # ax2[3].axis('off')
    # ax2[4].axis('off')
    # ax2[5].axis('off')
    # ax2[0].set_title('deficit 25%')
    # ax2[1].set_title('reconstructed by K-SVD\n{:.6f}'.format(dl.get_rmse(im_zero, Y_recon_KSVD[3])))
    # ax2[2].set_title('deficit 30%')
    # ax2[3].set_title('reconstructed by K-SVD\n{:.6f}'.format(dl.get_rmse(im_zero, Y_recon_KSVD[4])))
    # ax2[4].set_title('deficit 40%')
    # ax2[5].set_title('reconstructed by K-SVD\n{:.6f}'.format(dl.get_rmse(im_zero, Y_recon_KSVD[5])))
    # plt.tight_layout()
    # plt.savefig('temp1/hosei_nashi/dem_recon_KSVD02.png', dpi=220)
    # plt.clf()
    #
    # fig3, ax3 = plt.subplots(3, 2, figsize=(8, 12))
    # ax3 = ax3.flatten()
    # ax3[0].imshow(Y_masked[6], cmap='gray', interpolation='Nearest')
    # ax3[1].imshow(Y_recon_KSVD[6], cmap='gray', interpolation='Nearest')
    # # ax3[2].imshow(Y_masked[7], cmap='gray', interpolation='Nearest')
    # # ax3[3].imshow(Y_recon_KSVD[7], cmap='gray', interpolation='Nearest')
    # # ax3[4].imshow(Y_masked[8], cmap='gray', interpolation='Nearest')
    # # ax3[5].imshow(Y_recon_KSVD[8], cmap='gray', interpolation='Nearest')
    # ax3[0].axis('off')
    # ax3[1].axis('off')
    # ax3[2].axis('off')
    # ax3[3].axis('off')
    # ax3[4].axis('off')
    # ax3[5].axis('off')
    # ax3[0].set_title('deficit 50%')
    # ax3[1].set_title('reconstructed by K-SVD\n{:.6f}'.format(dl.get_rmse(im_zero, Y_recon_KSVD[6])))
    # # ax3[2].set_title('deficit 60%')
    # # ax3[3].set_title('reconstructed by K-SVD\n{:.6f}'.format(dl.get_rmse(im_zero, Y_recon_KSVD[7])))
    # # ax3[4].set_title('deficit 70%')
    # # ax3[5].set_title('reconstructed by K-SVD\n{:.6f}'.format(dl.get_rmse(im_zero, Y_recon_KSVD[8])))
    # plt.tight_layout()
    # plt.savefig('temp1/hosei_nashi/dem_recon_KSVD03.png', dpi=220)
    # plt.clf()
    #
    # fig4, ax4 = plt.subplots(3, 2, figsize=(8, 12))
    # ax4 = ax4.flatten()
    # ax4[0].imshow(Y_masked[9], cmap='gray', interpolation='Nearest')
    # ax4[1].imshow(Y_recon_KSVD[9], cmap='gray', interpolation='Nearest')
    # ax4[2].imshow(Y_masked[10], cmap='gray', interpolation='Nearest')
    # ax4[3].imshow(Y_recon_KSVD[10], cmap='gray', interpolation='Nearest')
    # ax4[4].imshow(Y_masked[11], cmap='gray', interpolation='Nearest')
    # ax4[5].imshow(Y_recon_KSVD[11], cmap='gray', interpolation='Nearest')
    # ax4[0].axis('off')
    # ax4[1].axis('off')
    # ax4[2].axis('off')
    # ax4[3].axis('off')
    # ax4[4].axis('off')
    # ax4[5].axis('off')
    # ax4[0].set_title('deficit 75%')
    # ax4[1].set_title('reconstructed by K-SVD\n{:.6f}'.format(dl.get_rmse(im_zero, Y_recon_KSVD[9])))
    # ax4[2].set_title('deficit 80%')
    # ax4[3].set_title('reconstructed by K-SVD\n{:.6f}'.format(dl.get_rmse(im_zero, Y_recon_KSVD[10])))
    # ax4[4].set_title('deficit 90%')
    # ax4[5].set_title('reconstructed by K-SVD\n{:.6f}'.format(dl.get_rmse(im_zero, Y_recon_KSVD[11])))
    # plt.tight_layout()
    # plt.savefig('temp1/hosei_nashi/zdem_recon_KSVD_75-90.png', dpi=220)
    # plt.clf()

    """
    再構成画像の保存
    """
    fig5, ax = plt.subplots(3, 2, figsize=(8, 12))
    ax = ax.flatten()
    ax[0].imshow(Y_masked[0], cmap='gray', interpolation='Nearest')
    ax[1].imshow(Y_recon_KSVD_1[0], cmap='gray', interpolation='Nearest')
    # ax[2].imshow(Y_masked[1], cmap='gray', interpolation='Nearest')
    # ax[3].imshow(Y_recon_KSVD_1[1], cmap='gray', interpolation='Nearest')
    # ax[4].imshow(Y_masked[2], cmap='gray', interpolation='Nearest')
    # ax[5].imshow(Y_recon_KSVD_1[2], cmap='gray', interpolation='Nearest')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[3].axis('off')
    ax[4].axis('off')
    ax[5].axis('off')
    ax[0].set_title('deficit 5%')
    ax[1].set_title('reconstructed by K-SVD\n{:.6f}'.format(dl.get_rmse(im_zero, Y_recon_KSVD_1[0])))
    # ax[2].set_title('deficit 10%')
    # ax[3].set_title('reconstructed by K-SVD\n{:.6f}'.format(dl.get_rmse(im_zero, Y_recon_KSVD_1[1])))
    # ax[4].set_title('deficit 20%')
    # ax[5].set_title('reconstructed by K-SVD\n{:.6f}'.format(dl.get_rmse(im_zero, Y_recon_KSVD_1[2])))
    plt.tight_layout()
    plt.savefig('temp1/hosei_ari/dem_recon_KSVD01_1.png', dpi=220)
    plt.clf()

    # fig6, ax2 = plt.subplots(3, 2, figsize=(8, 12))
    # ax2 = ax2.flatten()
    # ax2[0].imshow(Y_masked[3], cmap='gray', interpolation='Nearest')
    # ax2[1].imshow(Y_recon_KSVD_1[3], cmap='gray', interpolation='Nearest')
    # ax2[2].imshow(Y_masked[4], cmap='gray', interpolation='Nearest')
    # ax2[3].imshow(Y_recon_KSVD_1[4], cmap='gray', interpolation='Nearest')
    # ax2[4].imshow(Y_masked[5], cmap='gray', interpolation='Nearest')
    # ax2[5].imshow(Y_recon_KSVD_1[5], cmap='gray', interpolation='Nearest')
    # ax2[0].axis('off')
    # ax2[1].axis('off')
    # ax2[2].axis('off')
    # ax2[3].axis('off')
    # ax2[4].axis('off')
    # ax2[5].axis('off')
    # ax2[0].set_title('deficit 25%')
    # ax2[1].set_title('reconstructed by K-SVD\n{:.6f}'.format(dl.get_rmse(im_zero, Y_recon_KSVD_1[3])))
    # ax2[2].set_title('deficit 30%')
    # ax2[3].set_title('reconstructed by K-SVD\n{:.6f}'.format(dl.get_rmse(im_zero, Y_recon_KSVD_1[4])))
    # ax2[4].set_title('deficit 40%')
    # ax2[5].set_title('reconstructed by K-SVD\n{:.6f}'.format(dl.get_rmse(im_zero, Y_recon_KSVD_1[5])))
    # plt.tight_layout()
    # plt.savefig('temp1/hosei_ari/dem_recon_KSVD02_1.png', dpi=220)
    # plt.clf()
    #
    # fig7, ax3 = plt.subplots(3, 2, figsize=(8, 12))
    # ax3 = ax3.flatten()
    # ax3[0].imshow(Y_masked[6], cmap='gray', interpolation='Nearest')
    # ax3[1].imshow(Y_recon_KSVD_1[6], cmap='gray', interpolation='Nearest')
    # # ax3[2].imshow(Y_masked[7], cmap='gray', interpolation='Nearest')
    # # ax3[3].imshow(Y_recon_KSVD_1[7], cmap='gray', interpolation='Nearest')
    # # ax3[4].imshow(Y_masked[8], cmap='gray', interpolation='Nearest')
    # # ax3[5].imshow(Y_recon_KSVD_1[8], cmap='gray', interpolation='Nearest')
    # ax3[0].axis('off')
    # ax3[1].axis('off')
    # ax3[2].axis('off')
    # ax3[3].axis('off')
    # ax3[4].axis('off')
    # ax3[5].axis('off')
    # ax3[0].set_title('deficit 50%')
    # ax3[1].set_title('reconstructed by K-SVD\n{:.6f}'.format(dl.get_rmse(im_zero, Y_recon_KSVD_1[6])))
    # # ax3[2].set_title('deficit 60%')
    # # ax3[3].set_title('reconstructed by K-SVD\n{:.6f}'.format(dl.get_rmse(im_zero, Y_recon_KSVD_1[7])))
    # # ax3[4].set_title('deficit 70%')
    # # ax3[5].set_title('reconstructed by K-SVD\n{:.6f}'.format(dl.get_rmse(im_zero, Y_recon_KSVD_1[8])))
    # plt.tight_layout()
    # plt.savefig('temp1/hosei_ari/dem_recon_KSVD03_1.png', dpi=220)
    # plt.clf()
    #
    # fig8, ax4 = plt.subplots(3, 2, figsize=(8, 12))
    # ax4 = ax4.flatten()
    # ax4[0].imshow(Y_masked[9], cmap='gray', interpolation='Nearest')
    # ax4[1].imshow(Y_recon_KSVD_1[9], cmap='gray', interpolation='Nearest')
    # ax4[2].imshow(Y_masked[10], cmap='gray', interpolation='Nearest')
    # ax4[3].imshow(Y_recon_KSVD_1[10], cmap='gray', interpolation='Nearest')
    # ax4[4].imshow(Y_masked[11], cmap='gray', interpolation='Nearest')
    # ax4[5].imshow(Y_recon_KSVD_1[11], cmap='gray', interpolation='Nearest')
    # ax4[0].axis('off')
    # ax4[1].axis('off')
    # ax4[2].axis('off')
    # ax4[3].axis('off')
    # ax4[4].axis('off')
    # ax4[5].axis('off')
    # ax4[0].set_title('deficit 75%')
    # ax4[1].set_title('reconstructed by K-SVD\n{:.6f}'.format(dl.get_rmse(im_zero, Y_recon_KSVD_1[9])))
    # ax4[2].set_title('deficit 80%')
    # ax4[3].set_title('reconstructed by K-SVD\n{:.6f}'.format(dl.get_rmse(im_zero, Y_recon_KSVD_1[10])))
    # ax4[4].set_title('deficit 90%')
    # ax4[5].set_title('reconstructed by K-SVD\n{:.6f}'.format(dl.get_rmse(im_zero, Y_recon_KSVD_1[11])))
    # plt.tight_layout()
    # plt.savefig('temp1/hosei_ari/zdem_recon_KSVD_75-90_1.png', dpi=220)
    # plt.clf()

    time2 = time.clock()
    time_t = int(time2 - time1)
    print("計算時間 : ", time_t, " 秒")
