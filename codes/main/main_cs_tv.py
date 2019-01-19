# -*- coding: utf-8 -*-
"""
画像に対して CS（圧縮センシング）を適用(TV minimizattion)
Written in python3, created by Imose Kazuki on 2018/10/29
"""
import glob

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

import compressed_sensing as cs
import visualize_dem

if __name__ == "__main__":
    # DEM ファイルの指定
    files = glob.glob('E:/21_Sparse_Coding/files/targetdem_002.txt')
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

    # マスク画像の作成
    sig = 0
    Y = im_zero + np.random.randn(im_row, im_col) * sig  # 標準偏差 sig の正規分布ノイズを画像に追加
    # deficits = [5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]
    deficits = [11, 12, 13, 14, 15, 16, 17]
    mask = []  # mask の格納されるリスト
    Y_masked = []  # mask された DEM の格納されるリスト
    # mask の読み込み
    for deficit in deficits:
        mask_temp = np.fromfile('files/mask_' + str(deficit), dtype=np.uint8).reshape(im.shape)
        mask.append(mask_temp)
    # DEM のマスク処理
    for mask_num in range(len(deficits)):
        Y_masked_temp = Y * mask[mask_num]
        Y_masked.append(Y_masked_temp)

    # 欠損画像のリストを作成
    painted = len(Y_masked) * [None]
    im_origin = Y.astype(float)  # cast back to floats for error norm calculation

    for i, Y_masked in enumerate(Y_masked):
        print(i)
        known_coords = np.where(mask[i] == 1)
        ip = cs.Inpaint(verbose=True, store=False)
        painted[i], objective, iterates = ip.transform(Y_masked, known_coords)
        painted[i] = painted[i].astype(float)
        print("Norm of error: {}".format(la.norm(im_origin.flatten() - painted[i].flatten())))

    for j in range(7):
        print(cs.get_rmse(im_zero, painted[j]))
        # Y_masked[j] = np.squeeze(Y_masked[j])
        # painted[j] = np.squeeze(painted[j])

    # recon = painted[0] + h_min

    # csv ファイルとして保存
    # np.savetxt("dem_002_tv.csv", recon, delimiter=',')

    # """
    # イメージの保存
    # """
    # fig, ax = plt.subplots(3, 2, figsize=(8, 12))
    # ax = ax.flatten()
    # ax[0].imshow(Y_masked[0], cmap='gray', interpolation='Nearest')
    # ax[1].imshow(painted[0], cmap='gray', interpolation='Nearest')
    # ax[2].imshow(Y_masked[1], cmap='gray', interpolation='Nearest')
    # ax[3].imshow(painted[1], cmap='gray', interpolation='Nearest')
    # ax[4].imshow(Y_masked[2], cmap='gray', interpolation='Nearest')
    # ax[5].imshow(painted[2], cmap='gray', interpolation='Nearest')
    # ax[0].axis('off')
    # ax[1].axis('off')
    # ax[2].axis('off')
    # ax[3].axis('off')
    # ax[4].axis('off')
    # ax[5].axis('off')
    # ax[0].set_title('deficit 5%')
    # ax[1].set_title('Reconstructed by using TV\nRMSE = {:.6f}'.format(cs.get_rmse(im_zero, painted[0])))
    # ax[2].set_title('deficit 10%')
    # ax[3].set_title('Reconstructed by using TV\nRMSE = {:.6f}'.format(cs.get_rmse(im_zero, painted[1])))
    # ax[4].set_title('deficit 20%')
    # ax[5].set_title('Reconstructed by using TV\nRMSE = {:.6f}'.format(cs.get_rmse(im_zero, painted[2])))
    # # ax[1].set_title('interpolate with mean')
    # plt.tight_layout()
    # plt.savefig('images_TV/dem_002/DEM_002_CS_TV_5-20.png', dpi=220)
    # plt.clf()

    """
    イメージの保存
    """
    fig, ax = plt.subplots(3, 2, figsize=(8, 12))
    ax = ax.flatten()
    # ax[0].imshow(Y_masked[0], cmap='gray', interpolation='Nearest')
    ax[1].imshow(painted[0], cmap='gray', interpolation='Nearest')
    # ax[2].imshow(Y_masked[1], cmap='gray', interpolation='Nearest')
    ax[3].imshow(painted[1], cmap='gray', interpolation='Nearest')
    # ax[4].imshow(Y_masked[2], cmap='gray', interpolation='Nearest')
    ax[5].imshow(painted[2], cmap='gray', interpolation='Nearest')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[3].axis('off')
    ax[4].axis('off')
    ax[5].axis('off')
    ax[0].set_title('deficit 5%')
    ax[1].set_title('Reconstructed by using TV\nRMSE = {:.6f}'.format(cs.get_rmse(im_zero, painted[0])))
    ax[2].set_title('deficit 10%')
    ax[3].set_title('Reconstructed by using TV\nRMSE = {:.6f}'.format(cs.get_rmse(im_zero, painted[1])))
    ax[4].set_title('deficit 20%')
    ax[5].set_title('Reconstructed by using TV\nRMSE = {:.6f}'.format(cs.get_rmse(im_zero, painted[2])))
    # ax[1].set_title('interpolate with mean')
    plt.tight_layout()
    plt.savefig('dem_002_tv_nonr_01/DEM_002_CS_TV01.png', dpi=220)
    plt.clf()

    fig2, ax2 = plt.subplots(3, 2, figsize=(8, 12))
    ax2 = ax2.flatten()
    # ax2[0].imshow(Y_masked[3], cmap='gray', interpolation='Nearest')
    ax2[1].imshow(painted[3], cmap='gray', interpolation='Nearest')
    # ax2[2].imshow(Y_masked[4], cmap='gray', interpolation='Nearest')
    ax2[3].imshow(painted[4], cmap='gray', interpolation='Nearest')
    # ax2[4].imshow(Y_masked[5], cmap='gray', interpolation='Nearest')
    ax2[5].imshow(painted[5], cmap='gray', interpolation='Nearest')
    ax2[0].axis('off')
    ax2[1].axis('off')
    ax2[2].axis('off')
    ax2[3].axis('off')
    ax2[4].axis('off')
    ax2[5].axis('off')
    ax2[0].set_title('deficit 25%')
    ax2[1].set_title('Reconstructed by using TV\nRMSE = {:.6f}'.format(cs.get_rmse(im_zero, painted[3])))
    ax2[2].set_title('deficit 30%')
    ax2[3].set_title('Reconstructed by using TV\nRMSE = {:.6f}'.format(cs.get_rmse(im_zero, painted[4])))
    ax2[4].set_title('deficit 40%')
    ax2[5].set_title('Reconstructed by using TV\nRMSE = {:.6f}'.format(cs.get_rmse(im_zero, painted[5])))
    # ax2[1].set_title('interpolate with mean')
    plt.tight_layout()
    plt.savefig('dem_002_tv_nonr_01/DEM_002_CS_TV02.png', dpi=220)
    plt.clf()

    fig3, ax3 = plt.subplots(3, 2, figsize=(8, 12))
    ax3 = ax3.flatten()
    # ax3[0].imshow(Y_masked[6], cmap='gray', interpolation='Nearest')
    ax3[1].imshow(painted[6], cmap='gray', interpolation='Nearest')
    # ax3[2].imshow(Y_masked[7], cmap='gray', interpolation='Nearest')
    # ax3[3].imshow(painted[7], cmap='gray', interpolation='Nearest')
    # ax3[4].imshow(Y_masked[8], cmap='gray', interpolation='Nearest')
    # ax3[5].imshow(painted[8], cmap='gray', interpolation='Nearest')
    ax3[0].axis('off')
    ax3[1].axis('off')
    ax3[2].axis('off')
    ax3[3].axis('off')
    ax3[4].axis('off')
    ax3[5].axis('off')
    ax3[0].set_title('deficit 50%')
    ax3[1].set_title('Reconstructed by using TV\nRMSE = {:.6f}'.format(cs.get_rmse(im_zero, painted[6])))
    # ax3[2].set_title('deficit 60%')
    # ax3[3].set_title('Reconstructed by using TV\nRMSE = {:.6f}'.format(cs.get_rmse(im_zero, painted[7])))
    # ax3[4].set_title('deficit 70%')
    # ax3[5].set_title('Reconstructed by using TV\nRMSE = {:.6f}'.format(cs.get_rmse(im_zero, painted[8])))
    plt.tight_layout()
    plt.savefig('dem_002_tv_nonr_01/DEM_002_CS_TV03.png', dpi=220)
    plt.clf()
    #
    # fig4, ax4 = plt.subplots(3, 2, figsize=(8, 12))
    # ax4 = ax4.flatten()
    # ax4[0].imshow(Y_masked[9], cmap='gray', interpolation='Nearest')
    # ax4[1].imshow(painted[9], cmap='gray', interpolation='Nearest')
    # ax4[2].imshow(Y_masked[10], cmap='gray', interpolation='Nearest')
    # ax4[3].imshow(painted[10], cmap='gray', interpolation='Nearest')
    # ax4[4].imshow(Y_masked[11], cmap='gray', interpolation='Nearest')
    # ax4[5].imshow(painted[11], cmap='gray', interpolation='Nearest')
    # ax4[0].axis('off')
    # ax4[1].axis('off')
    # ax4[2].axis('off')
    # ax4[3].axis('off')
    # ax4[4].axis('off')
    # ax4[5].axis('off')
    # ax4[0].set_title('deficit 75%')
    # ax4[1].set_title('Reconstructed by using TV\nRMSE = {:.6f}'.format(cs.get_rmse(im_zero, painted[9])))
    # ax4[2].set_title('deficit 80%')
    # ax4[3].set_title('Reconstructed by using TV\nRMSE = {:.6f}'.format(cs.get_rmse(im_zero, painted[10])))
    # ax4[4].set_title('deficit 90%')
    # ax4[5].set_title('Reconstructed by using TV\nRMSE = {:.6f}'.format(cs.get_rmse(im_zero, painted[11])))
    # plt.tight_layout()
    # plt.savefig('images_TV/dem_002/DEM_002_CS_TV_75-90.png', dpi=220)
    # plt.clf()
