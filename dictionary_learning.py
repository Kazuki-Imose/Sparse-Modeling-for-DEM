# -*- coding: utf-8 -*-
"""
Dictionary Learning(K-SVD) のための module
Written in python3, created by Imose Kazuki on 2018/11/12
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from sklearn import cross_validation, preprocessing, linear_model  # 機械学習用のライブラリを利用


def get_psnr(im, recon):
    """ PSNRを得る """
    # return 10. * np.log(im.max() / np.sqrt(np.mean((im - recon) ** 2)))
    return 20. * np.log10(im.max() / np.sqrt(np.mean((im - recon) ** 2)))


def get_rmse(im, recon):
    """ RMSEを得る """
    # return 10. * np.log(im.max() / np.sqrt(np.mean((im - recon) ** 2)))
    return np.sqrt(np.mean((im - recon) ** 2))


def show_dictionary(A, name=None, figsize=(4, 4), vmin=None, vmax=None):
    """ 辞書を表示 """
    n = int(np.sqrt(A.shape[0]))
    m = int(np.sqrt(A.shape[1]))
    A_show = A.reshape((n, n, m, m))
    fig, ax = plt.subplots(m, m, figsize=figsize)
    for row in range(m):
        for col in range(m):
            ax[row, col].imshow(A_show[:, :, col, row], cmap='gray', interpolation='Nearest', vmin=vmin, vmax=vmax)
            ax[row, col].axis('off')
    if name is not None:
        plt.savefig(name, dpi=220)


def OMP(A, b, k0, eps):
    """
    直交マッチング追跡(orthogonal matching pursuit; OMP)

    A nxm行列
    b n要素の観測
    k0 xの非ゼロの要素数
    eps 誤差の閾値

    戻り値
    x m要素のスパース表現
    S m要素のサポートベクトル
    """
    # 初期化
    x = np.zeros(A.shape[1])
    S = np.zeros(A.shape[1], dtype=np.uint8)
    r = b.copy()
    rr = np.dot(r, r)
    for _ in range(k0):
        # 誤差計算
        err = rr - np.dot(A[:, S == 0].T, r) ** 2

        # サポート更新
        ndx = np.where(S == 0)[0]
        S[ndx[err.argmin()]] = 1
        # sum_S = sum(S)

        # 解更新
        As = A[:, S == 1]
        # pinv = np.linalg.pinv(np.dot(As, As.T))  # original
        # x[S == 1] = np.dot(As.T, np.dot(pinv, b))  # original
        pinv = np.linalg.pinv(np.dot(As.T, As))
        x[S == 1] = np.dot(pinv, np.dot(As.T, b))

        # 残差更新
        r = b - np.dot(A, x)
        rr = np.dot(r, r)
        if rr < eps:
            break

    return x, S


def calculatation_by_ElasticNet(A, b, alpha, l1_ratio):
    """
        Elastic Net を用いたスパースコーディング

        A nxm行列
        b n要素の観測
        alpha 正則化の強さ
        l1_ratio L1正則化の重み

        戻り値
        x m要素のスパース表現
        """
    # 初期化
    x = np.zeros(A.shape[1] + 1)

    # Elastic Net Regressor の適用
    clf_er = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    clf_er.fit(A, b)
    x[0] = clf_er.intercept_
    x[1:] = clf_er.coef_

    return x


class DictionaryLearning(object):
    """ 辞書学習 """

    def MOD(self, Y, sig, m, k0, n_iter=50, A0=None, initial_dictionary=None):
        """
        MOD辞書学習アルゴリズム

        Y 信号事例、n×M、nは事例の次元、Mは事例の総数
        sig ノイズレベル
        m 辞書の列数
        k0 非ゼロ要素の個数
        """
        if initial_dictionary is None:
            A = Y[:, :m]
            A = np.dot(A, np.diag(1. / np.sqrt(np.diag(np.dot(A.T, A)))))
        else:
            A = initial_dictionary
        X = np.zeros((A.shape[1], Y.shape[1]))
        eps = A.shape[0] * (sig ** 2)

        log = []
        for k in range(n_iter):
            for i in range(Y.shape[1]):
                X[:, i], _ = OMP(A, Y[:, i], k0, eps=eps)

            opt = np.abs(Y - np.dot(A, X)).mean()

            A = np.dot(Y, np.dot(X.T, np.linalg.pinv(np.dot(X, X.T))))
            A = np.dot(A, np.diag(1. / np.sqrt(np.diag(np.dot(A.T, A)))))

            if A0 is not None:
                opt2 = self.percent_recovery_of_atoms(A, A0)
                log.append((opt, opt2))
            else:
                log.append(opt)
            print(k, log[k])
            print(time.clock())

        return A, np.array(log)

    def KSVD(self, Y, sigma, m, k0, n_iter=50, A0=None, initial_dictionary=None, mask=None):
        """
        K-SVD辞書学習アルゴリズム

        Y 信号事例、n×M、nは信号の次元、Mは事例の総数
        sigma ノイズレベル
        m 辞書の列数
        k0 非ゼロ要素の個数

        参考
        https://github.com/greyhill/pypbip/blob/master/ksvd.py
        """
        if initial_dictionary is None:
            A = Y[:, :m]
            A = np.dot(A, np.diag(1. / np.sqrt(np.diag(np.dot(A.T, A)))))
        else:
            A = initial_dictionary.copy()
        X = np.zeros((A.shape[1], Y.shape[1]))
        eps = A.shape[0] * (sigma ** 2)

        ndx = np.arange(m)
        log = []
        for k in range(n_iter):
            if mask is None:
                for i in range(Y.shape[1]):
                    X[:, i], _ = OMP(A, Y[:, i], k0, eps=eps)
            else:
                # マスクあり
                for i in range(Y.shape[1]):
                    A_mask = A[mask[:, i] == 1, :]
                    Y_mask = Y[mask[:, i] == 1, i]
                    eps = len(Y_mask) * (sigma ** 2) * 1.1
                    X[:, i], _ = OMP(A_mask, Y_mask, k0, eps=eps)
                    Y[mask[:, i] == 0, i] = np.dot(A[mask[:, i] == 0, :], X[:, i])

            for j in ndx:
                x_using = X[j, :] != 0

                # if np.sum(x_using == 0):  # second
                #     error = Y - np.dot(A, X)  # second
                #     norm_error = sum(error ** 2)  # second
                #     min_col = np.argmax(norm_error)  # second
                #     new_atom = Y[:, min_col]  # second
                #     new_atom = new_atom / np.sqrt(np.dot(new_atom.T, new_atom))  # second
                #     A[:, j] = new_atom  # second
                # else:
                #     X[j, x_using] = 0  # second
                #     Residual_err = Y[:, x_using] - np.dot(A, X[:, x_using])  # second
                #     U, s, Vt = np.linalg.svd(Residual_err)  # second
                #     A[:, j] = U[:, 0]  # second
                #     X[j, x_using] = s[0] * Vt.T[:, 0]  # second

                if np.sum(x_using) == 0:  # original
                    continue  # original
                X[j, x_using] = 0  # original
                Residual_err = Y[:, x_using] - np.dot(A, X[:, x_using])  # original
                U, s, Vt = np.linalg.svd(Residual_err)  # original
                A[:, j] = U[:, 0]  # original
                X[j, x_using] = s[0] * Vt.T[:, 0]  # original

            opt = np.abs(Y - np.dot(A, X)).mean()
            # A = self.clear_dictionary(A, X, Y)  # second

            if A0 is not None:
                opt2 = self.percent_recovery_of_atoms(A, A0)
                log.append((opt, opt2))
            else:
                log.append(opt)
            print(k, log[k])

        return A, np.array(log)

    def percent_recovery_of_atoms(self, A, A0, threshold=0.99):
        """ アトムの復元率を測る """
        num = 0
        for m in range(A.shape[1]):
            a = A0[:, m]
            if np.abs(np.dot(a, A)).max() > threshold:
                num += 1
        return float(num) / A.shape[1] * 100

    def clear_dictionary(self, dictionary, code, data):
        """ 似たアトムの初期化 """
        n_features, n_components = dictionary.shape
        n_components, n_samples = code.shape
        norms = np.sqrt(sum(dictionary ** 2))
        norms = norms[:, np.newaxis].T
        dictionary = dictionary / np.dot(np.ones((n_features, 1)), norms)
        code = code * np.dot(norms.T, np.ones((1, n_samples)))

        t1 = 4  # 3
        t2 = 0.9  # 0.999
        error = sum((data - np.dot(dictionary, code)) ** 2)
        gram = np.dot(dictionary.T, dictionary)
        gram = gram - np.diag(np.diag(gram))

        for i in range(0, n_components):
            if (max(gram[i, :]) > t2) or (len(*np.nonzero(abs(code[i, :]) > 1e-7)) <= t1):
                # val = np.max(error)
                pos = np.argmax(error)
                error[pos] = 0
                dictionary[:, i] = data[:, pos] / np.linalg.norm(data[:, pos])
                gram = np.dot(dictionary.T, dictionary)
                gram = gram - np.diag(np.diag(gram))

        return dictionary


def sparse_coding_with_mask(im, A, k0, sigma, mask, patch_size=8):
    """ マスク付きスパース符号化 """
    patches = extract_patches_2d(im, (patch_size, patch_size))
    mask_patches = extract_patches_2d(mask, (patch_size, patch_size))
    q = np.zeros((len(patches), A.shape[1]))
    for i, (patch, mask_patch) in enumerate(zip(patches, mask_patches)):
        if i % 1000 == 0:
            print(i)
        A_mask = A[mask_patch.flatten() == 1, :]
        patch_mask = patch[mask_patch == 1]
        eps = len(patch_mask) * (sigma ** 2) * 1.1
        q[i], _ = OMP(A_mask, patch_mask, k0, eps=eps)
    return q


def recon_image(im, q, A, lam=0.5, patch_size=8):
    """ 画像の再構成 """
    recon_patches = np.dot(A, q.T).T.reshape((-1, patch_size, patch_size))
    recon = reconstruct_from_patches_2d(recon_patches, im.shape)
    return (im * lam + recon) / (lam + 1.)


def sparse_coding_by_ElasticNet_with_mask(im, A, mask, alpha, l1_ratio, patch_size=8):
    """ マスク付きスパース符号化 """
    patches = extract_patches_2d(im, (patch_size, patch_size))
    mask_patches = extract_patches_2d(mask, (patch_size, patch_size))
    q = np.zeros((len(patches), A.shape[1]+1))
    for i, (patch, mask_patch) in enumerate(zip(patches, mask_patches)):
        if i % 1000 == 0:
            print(i)
        A_mask = A[mask_patch.flatten() == 1, :]
        patch_mask = patch[mask_patch == 1]
        q[i] = calculatation_by_ElasticNet(A_mask, patch_mask, alpha, l1_ratio)
    return q


def recon_image_by_ElasticNet(im, q, A, lam=0.5, patch_size=8):
    """ 画像の再構成 """
    c = np.ones((A.shape[0], 1))
    Ac = np.hstack([c, A])
    recon_patches = np.dot(Ac, q.T).T.reshape((-1, patch_size, patch_size))
    recon = reconstruct_from_patches_2d(recon_patches, im.shape)
    return (im * lam + recon) / (lam + 1.)


def dictionary_learning_with_mask(im, mask, A_DCT, patch_size=8, dict_size=16, k0=4, n_iter=15):
    """ マスク付き辞書学習を実行 """
    rate_using_patches = 0.1  # 使うパッチの割合
    dl = DictionaryLearning()
    patches = extract_patches_2d(im, (patch_size, patch_size)).reshape((-1, patch_size ** 2))
    mask_patches = extract_patches_2d(mask, (patch_size, patch_size)).reshape((-1, patch_size ** 2))
    M = len(patches)
    print(M)

    num_using_patches = int(M * rate_using_patches)
    A_KSVD = A_DCT.copy()
    for _ in range(n_iter):
        # ndx = np.random.permutation(M)[:A_DCT.shape[1] * 50]
        ndx = np.random.permutation(M)[:num_using_patches]
        # A_KSVD, _ = dl.KSVD(patches.T, 20., dict_size ** 2, k0, mask=mask_patches.T, n_iter=1,
        #                     initial_dictionary=A_KSVD)
        A_KSVD, _ = dl.KSVD(patches[ndx].T, sigma=0, m=dict_size ** 2, k0=k0, mask=mask_patches[ndx].T, n_iter=1,
                            initial_dictionary=A_KSVD)

    return A_KSVD
