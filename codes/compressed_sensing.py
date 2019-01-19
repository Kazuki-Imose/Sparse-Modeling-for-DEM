# -*- coding: utf-8 -*-
"""
圧縮センシング（Compressed Sensing）のためのモジュール
Written in python3, created by Imose Kazuki on 2018/11/9
"""

import sys

import cv2
import numpy as np
import pywt
from numpy import clip, empty, Inf, mod, sum, vstack, zeros
from numpy.linalg import norm

"""
1)General Function
"""


# PSNR を取得する関数
def get_psnr(im, recon):
    """ PSNRを得る """
    # return 10. * np.log(im.max() / np.sqrt(np.mean((im - recon) ** 2)))
    return 20. * np.log10(im.max() / np.sqrt(np.mean((im - recon) ** 2)))


# RMSE を取得する関数
def get_rmse(im, recon):
    """ RMSEを得る """
    # return 10. * np.log(im.max() / np.sqrt(np.mean((im - recon) ** 2)))
    return np.sqrt(np.mean((im - recon) ** 2))


# ソフト閾値関数
def soft_thresh(b, lam):
    x_hat = np.zeros(b.shape)
    x_hat[b >= lam] = b[b >= lam] - lam
    x_hat[b <= -lam] = b[b <= -lam] + lam
    return x_hat


# 配列内の数値を標準化する関数
def image_normalization(src_img):
    """
    白飛び防止のための正規化処理
    cv2.imshowでwavelet変換された画像を表示するときに必要（大きい値を持つ画像の時だけ）
    """
    norm_img = (src_img - np.min(src_img)) / (np.max(src_img) - np.min(src_img))
    return norm_img


"""
2-1)Compressed Sensing by using Wavelet transformation
"""


# フーリエ変換を用いた圧縮センシングを行う関数
def cs_fourier(Y_imput, mask,  im, n_iter, alpha):
    x_last = Y_imput  # 更新前の x
    rmse = np.empty(n_iter + 1)  # rmse を格納する配列

    RMSE = get_rmse(im, x_last)  # 初期 rmse を計算
    rmse[0] = RMSE
    print('RMSE :', RMSE)
    # 反復計算による圧縮センシングの適用
    for ii in range(1, n_iter+1):
        x_w = fourier_transform_for_image(x_last)  # xをフーリエ変換
        x_w_soft = soft_thresh_to_fourier_coeffs(x_w, alpha)  # ソフト閾値処理
        x_temp = inverse_fourier_transform_for_image(x_w_soft)  # フーリエ逆変換
        x_next = np.where(mask == 0, x_temp, Y_imput)  # 補間値の更新
        RMSE = get_rmse(im, x_next)  # RMSEの計算
        rmse[ii] = RMSE
        diff_rmse = RMSE - rmse[ii - 1]
        if ii % 50 == 0:
            print('RMSE :', RMSE)
        if diff_rmse > - 0.000001:  # 更新速度が一定値以下の場合終了する
            print("反復回数 : ", ii)
            # coeffs_visualization(x_w_soft, "images/x_w_soft.png")  # 最後のフーリエ係数を保存する
            break
        x_last = x_next
    # ゼロ以下の値をゼロにする
    recon = np.where(x_last < 0, 0, x_last)
    return recon, rmse


# ソフト閾値関数をフーリエ変換係数に適用
def soft_thresh_to_fourier_coeffs(b, lam):
    col = b.shape[0]
    row = b.shape[1]
    x_hat = np.zeros((col, row, 2))
    x_hat_temp = np.zeros((col, row))
    b_temp = b[:, :, 0]
    x_hat_temp[b_temp >= lam] = b_temp[b_temp >= lam] - lam
    x_hat_temp[b_temp <= -lam] = b_temp[b_temp <= -lam] + lam
    x_hat[:, :, 0] = x_hat_temp
    b_temp = b[:, :, 1]
    x_hat_temp[b_temp >= lam] = b_temp[b_temp >= lam] - lam
    x_hat_temp[b_temp <= -lam] = b_temp[b_temp <= -lam] + lam
    x_hat[:, :, 1] = x_hat_temp
    return x_hat


# フーリエ変換を行う関数
def fourier_transform_for_image(src_image):
    """
    coeffs: ndarray(2層, 実部と虚部)
    """
    data = src_image.astype(np.float64)
    # coeffs = np.fft.fft2(data)
    coeffs = cv2.dft(data, flags=cv2.DFT_COMPLEX_OUTPUT)
    return coeffs


# フーリエ逆変換を行う関数
def inverse_fourier_transform_for_image(cof):
    # recon = np.fft.ifft2(cof)
    recon = cv2.idft(cof)
    return recon


"""
2-2)Compressed Sensing by using Wavelet transformation
"""


# ウェーブレット変換を用いた圧縮センシングを行う関数
def cs_wavelet(Y_imput, mask,  im, n_iter, alpha, LEVEL, MOTHER_WAVELET="db1"):
    x_last = Y_imput  # 更新前の x
    rmse = np.empty(n_iter + 1)  # rmse を格納する配列

    RMSE = get_rmse(im, x_last)  # 初期 rmse を計算
    rmse[0] = RMSE
    print('RMSE :', RMSE)
    # 反復計算による圧縮センシングの適用
    for ii in range(1, n_iter+1):
        x_w = wavelet_transform_for_image(x_last, LEVEL, M_WAVELET=MOTHER_WAVELET)  # xをウェーブレット変換
        x_w_soft = soft_thresh_to_coeffs(x_w, alpha)  # ソフト閾値処理
        x_temp = inverse_wavelet_transform_for_image(x_w_soft, M_WAVELET=MOTHER_WAVELET)  # ウェーブレット逆変換
        x_next = np.where(mask == 0, x_temp, Y_imput)  # 補間値の更新
        RMSE = get_rmse(im, x_next)  # RMSEの計算
        rmse[ii] = RMSE
        diff_rmse = RMSE - rmse[ii - 1]
        if ii % 50 == 0:
            print('RMSE :', RMSE)
        if diff_rmse > - 0.000001:  # 更新速度が一定値以下の場合終了する
            print("反復回数 : ", ii)
            # coeffs_visualization(x_w_soft, "images/x_w_soft.png")  # 最後のウェーブレット係数を保存する
            break
        x_last = x_next
    # ゼロ以下の値をゼロにする
    recon = np.where(x_last < 0, 0, x_last)
    return recon, rmse


# ソフト閾値関数をウェーブレット係数のリストに適用する関数
def soft_thresh_to_coeffs(cof, lam):
    cof0 = cof[0]
    cof0_thre = soft_thresh(cof0, lam)  # ソフト閾値処理
    coeffs_thre = [cof0_thre]  # リストにする
    # 二番目以降のタプルにも適用
    for i in range(1, len(cof)):
        cH, cV, cD = cof[i]
        cH_temp = soft_thresh(cH, lam)
        cV_temp = soft_thresh(cV, lam)
        cD_temp = soft_thresh(cD, lam)
        coeff_tuple = (cH_temp, cV_temp, cD_temp)  # タプル化
        coeffs_thre.append(coeff_tuple)  # リストにappend
    return coeffs_thre


# 標準化してウェーブレット係数をマージする関数
def merge_images(cA, cH_V_D):
    """numpy.array を４つ(左上、(右上、左下、右下))連結させる"""
    cH, cV, cD = cH_V_D
    cH = image_normalization(cH)  # 外してもok
    cV = image_normalization(cV)  # 外してもok
    cD = image_normalization(cD)  # 外してもok
    cA = cA[0:cH.shape[0], 0:cV.shape[1]]  # 元画像が2の累乗でない場合、端数ができることがあるので、サイズを合わせる。小さい方に合わせます。
    return np.vstack((np.hstack((cA, cH)), np.hstack((cV, cD))))  # 左上、右上、左下、右下、で画素をくっつける


# 標準化せずにウェーブレット係数をマージする関数
def merge_images_without_norm(cA, cH_V_D):
    """numpy.array を４つ(左上、(右上、左下、右下))連結させる"""
    cH, cV, cD = cH_V_D
    cA = cA[0:cH.shape[0], 0:cV.shape[1]]  # 元画像が2の累乗でない場合、端数ができることがあるので、サイズを合わせる。小さい方に合わせます。
    return np.vstack((np.hstack((cA, cH)), np.hstack((cV, cD))))  # 左上、右上、左下、右下、で画素をくっつける


# ウェーブレット係数を可視化する関数
def coeffs_visualization(cof, output_name):
    norm_cof0 = cof[0]
    norm_cof0 = image_normalization(norm_cof0)  # 外してもok
    merge = norm_cof0
    for i in range(1, len(cof)):
        merge = merge_images(merge, cof[i])  # ４つの画像を合わせていく
    cv2.imshow('wavelet transform', merge)
    cv2.waitKey(0)
    # 保存
    merge = merge * 255
    merge = merge.astype(np.int)
    cv2.imwrite(output_name, merge)
    cv2.destroyAllWindows()

    # k = cv2.waitKey(0)
    # if k == 27:  # wait for ESC key to exit
    #     cv2.destroyAllWindows()
    # elif k == ord('s'):  # wait for 's' key to save and exit
    #     cv2.imwrite('messigray.png', merge)
    #     cv2.destroyAllWindows()


# ウェーブレット係数を行列形式(画像サイズ)にする
def create_coeffs_matrix(cof):
    cof0 = cof[0]
    # norm_cof0 = image_normalization(norm_cof0)  # 外してもok
    merge = cof0
    for i in range(1, len(cof)):
        merge = merge_images_without_norm(merge, cof[i])  # ４つの画像を合わせていく
    return merge


# ウェーブレット係数の行列を4分割する関数
def divide_coeffs_matrix_to_4(cof_matrix):
    upper, lower = np.split(cof_matrix, 2, axis=0)
    upper_left, upper_right = np.split(upper, 2, axis=1)
    lower_left, lower_right = np.split(lower, 2, axis=1)
    coeff_tuple = (upper_right, lower_left, lower_right)
    return upper_left, coeff_tuple


# ウェーブレット係数の行列をリストに戻す関数
def divide_coeffs_matrix_to_end(cof_matrix, level):
    upper_left, coeff_tuple = divide_coeffs_matrix_to_4(cof_matrix)
    coeffs_recon = [coeff_tuple]
    for _ in range(level-1):
        upper_left, coeff_tuple = divide_coeffs_matrix_to_4(upper_left)
        coeffs_recon.insert(0, coeff_tuple)
    coeffs_recon.insert(0, upper_left)
    return coeffs_recon


# ウェーブレット変換を行う関数
def wavelet_transform_for_image(src_image, level, M_WAVELET="db1", mode="sym"):
    """
    coeffs: リスト型、0番目の要素はndarray、それ以降の要素はタプル、タプル内には3つの要素がありすべてndarray
    [array([]),(array([]),array([]),array([])), ... ,(array([]),array([]),array([]))]
    """
    data = src_image.astype(np.float64)
    coeffs = pywt.wavedec2(data, M_WAVELET, level=level, mode=mode)
    return coeffs


# ウェーブレット逆変換を行う関数
def inverse_wavelet_transform_for_image(cof, M_WAVELET="db1", mode="sym"):
    recon = pywt.waverec2(cof, M_WAVELET, mode=mode)
    return recon


"""
3)Compressed Sensing by using Total Variation
"""


# 画像のトータルバリエーションの微分を計算する関数
def calculate_differential_total_variation(im, i, j, epsilon):
    diff_tv = (im[i, j] - im[i - 1, j]) / np.sqrt(
        (im[i, j] - im[i - 1, j]) ** 2 + (im[i - 1, j + 1] - im[i - 1, j]) ** 2 + epsilon)
    + (im[i, j] - im[i, j - 1]) / np.sqrt(
        (im[i + 1, j - 1] - im[i, j - 1]) ** 2 + (im[i, j] - im[i, j - 1]) ** 2 + epsilon)
    - (im[i + 1, j] + im[i, j + 1] - 2 * im[i, j]) / np.sqrt(
        (im[i + 1, j] - im[i, j]) ** 2 + (im[i, j + 1] - im[i, j]) ** 2 + epsilon)
    return diff_tv


# 画像のトータルバリエーションの微分行列を取得する関数
def get_differential_total_variation(im, epsilon):
    diff_tv = np.zeros(im.shape)
    row = im.shape[0]
    col = im.shape[1]
    for i in range(1, row-1):
        for j in range(1, col-1):
            diff_tv[i, j] = calculate_differential_total_variation(im, i, j, epsilon)
    return diff_tv


# 画像のトータルヴァリエーションとその勾配を求める関数
def l2tv(image):
    """
    Computes the L2-norm total variation and a subgradient of an image

    References
    ==========
    Adapted from Stanford EE 364B Convex Optimization II final

    """
    m, n = image.shape

    # Pixel value differences across columns
    col_diff = image[: -1, 1:] - image[: -1, : -1]

    # Pixel value differences across rows
    row_diff = image[1:, : -1] - image[: -1, : -1]

    # Compute the L2-norm total variation of the image
    diff_norms = norm(vstack((col_diff.T.flatten(), row_diff.T.flatten())).T, ord=2, axis=1)
    val = sum(diff_norms) / ((m - 1) * (n - 1))

    # Compute a subgradient. When non-differentiable, set to 0
    # by dividing by infinity.
    subgrad = zeros((m, n))
    norms_mat = diff_norms.reshape(n - 1, m - 1).T
    norms_mat[norms_mat == 0] = Inf
    subgrad[: -1, : -1] = - col_diff / norms_mat
    subgrad[: -1, 1:] = subgrad[: -1, 1:] + col_diff / norms_mat
    subgrad[: -1, : -1] = subgrad[: -1, : -1] - row_diff / norms_mat
    subgrad[1:, : -1] = subgrad[1:, : -1] + row_diff / norms_mat

    return val, subgrad


# 既知のインデックスからマスクを作成する関数
def make_mask(shape, known_coords):
    mask = zeros(shape)
    mask[known_coords] = 1
    return mask.astype(bool)  # necessary? probably slower...


# Inpaint クラス
class Inpaint(object):
    """
    Inpaints an image with unknown pixels

    Parameters
    ==========
    alpha : float, optional, default = 200
        Numerator parameter in square-summable-but-not-summable (SSBNS...
        or better non-lame name?) step size

    beta : float, optional, default = 1
        Denominator parameter in SSBNS step size

    max_iter : int, optional, default = 1000
        Maximum number of iterations

    method : str in {'l2tv'}, optional, default = 'l2tv'
        Method for inpainting. Currently only projected subgradient for L2-norm
        total variation minimization supported.

    store : boolean, optional, default = False
        Whether to store and return each iterate

    tol : float, optional, default = 1e-3
        Stopping tolerance on L2 norm of the difference between iterates

    verbose : boolean, optional, default = False
        Whether to print objective per iteration


    References
    ==========
    Something Bertsekas wrote on SSBNS. Need to find

    """

    def __init__(self, alpha=200, beta=1, max_iter=5000, method='l2tv', store=False, tol=1e-4, verbose=False):

        if method not in 'l2tv':
            raise ValueError('Invalid method: got %r instead of one of %r' %
                             (method, 'l2tv'))
        self.method = method

        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.store = store
        self.tol = tol
        self.verbose = verbose

    def _l2tv(self, image, mask):
        # Projected subgradient method for L2-norm total variation (L2TV) minimization.
        # Initialize iterate.
        global n_iter, obj
        painted = image
        painted_best = image
        objective = empty(self.max_iter)
        obj_best = Inf
        if self.store:
            iterates = self.max_iter * [None]
        else:
            iterates = None

        for n_iter in range(self.max_iter):

            # Compute L2TV objective and subgradient.
            obj, subgrad = l2tv(painted)
            objective[n_iter] = obj
            if self.store:
                iterates[n_iter] = painted

            # Update iterate with best objective so far.
            if obj < obj_best:
                obj_best = obj
                painted_best = painted

            if self.verbose:
                if mod(n_iter, 100) == 0:
                    print('Iter: %i. Objective: %f. Best objective: %f.' % (n_iter, obj, obj_best))
                    sys.stdout.flush()

            # Update iterate by stepping in negative subgradient direction.

            # TODO: Try searching for step size that produces sufficient decrease
            # ("Armijo rule along the projection arc" in Bertsekas (1999),
            # using shortcut condition in Lin (2007) Eq. (17)).

            painted_prev = painted
            painted = painted - (self.alpha / (self.beta + n_iter)) * subgrad

            # Projection onto feasible set, or set all known pixel values in
            # non-fancy speak.
            painted[mask] = image[mask]
            clip(painted, 0, 256, painted)

            # Check for convergence.
            if norm(painted - painted_prev) / norm(painted) < self.tol:
                break

        if self.verbose:
            print('Iter: %i. Final objective %f. Best objective %f.' % (n_iter, obj, obj_best))
            sys.stdout.flush()

        objective = objective[: n_iter + 1]
        if self.store:
            iterates = iterates[: n_iter + 1]
        painted = painted_best

        return painted, objective, iterates

    def transform(self, image, known_coords):

        # TODO: Check for scaling in [0, 1] vs. [0, 255] and set
        # np.clip() parameter in _l2tv() correspondingly

        # Implement L2-norm-squared total variation for kicks? Should converge faster
        # since quadratic, same optimum.

        """
        Inpaints image given unknown pixel coordinates

        Parameters
        ==========
        image : array, shape (m, n, n_channel)
            Input image with unknown pixel values. n_channel is the number
            of color channels, i.e. n_channel = 3 if RGB, n_channel = 4 if RGBA, etc.

        known_coords : tuple (array-like, array-like)
            x- and y-coordinates of known pixel values

        Returns
        =======
        painted : array, shape (m, n)
            Inpainted image

        objective : array, shape (n_iter)
            For each channel, value of objective at each iteration

        iterates : list of n_iter arrays of shape (m, n)
            If self.store == True, a list of each iterate. Else, None.

        """

        global painted, iterates
        image = image.astype(float)
        shape = image.shape

        if len(shape) == 2:

            mask = make_mask(shape, known_coords)
            if self.method == "l2tv":
                painted, objective, iterates = self._l2tv(image, mask)

            # Cast back to ints.
            # painted = painted.astype(int)
            if self.store:
                iterates = [iterate.astype(int) for iterate in iterates]

        elif len(shape) == 3:

            raise ValueError("Color images not supported: naively in-painting "
                             "each channel separately sucks bad. Need to figure out algorithm to do them jointly...")

            m, n, n_channel = shape
            painted = empty(shape)
            objective = n_channel * [None]
            iterates = n_channel * [None]

            # Convert known coordinates into binary mask
            # for easier indexing. Was much faster in MATLAB,
            # need to make sure also faster here.
            mask = make_mask((m, n), known_coords)

            for chan in range(n_channel):

                if self.verbose:
                    print("\nIn-painting channel %i." % chan)

                if self.method == "l2tv":
                    painted[:, :, chan], objective[chan], iterates[chan] = self._l2tv(image[:, :, chan], mask)

        else:
            raise ValueError('Invalid input dimensions: image has %i dimensions, ' +
                             'but needs to have 2 or 3.' % len(shape))

        return painted, objective, iterates
