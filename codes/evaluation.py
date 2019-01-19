# -*- coding: utf-8 -*-
"""
Implementation of Reconstruction by using DCT to DEM
Written in python3, created by Imose Kazuki on 2018/11/12
"""
import glob

import numpy as np
import pandas as pd

import compressed_sensing as cs

if __name__ == "__main__":
    # 深浅測量データ の読み込み
    shinsen = np.array(pd.read_csv("files/shinsen_hyoko_dem_102.csv", header=None))
    mask = np.where(shinsen == 0, 0, 1)

    # ファイルの指定
    # files = glob.glob('evaluation/dem_102/03/*.csv')
    files = glob.glob('normal_method/dem_102/*.csv')
    print(files)
    dems = []  # DEM を格納するリスト
    for file in files:
        dem_temp = np.array(pd.read_csv(file, header=None))
        dems.append(dem_temp)
    num_dems = len(dems)
    print("the number of DEMs : ", num_dems)

    # 精度評価
    results = []  # 精度評価結果を格納するリスト
    for i in range(num_dems):
        result_temp = cs.get_rmse_with_mask(dems[i], shinsen, mask)
        results.append(result_temp)

    # np.savetxt("evaluation/dem_102/03/results.csv", results, fmt='%s', delimiter=',')
    np.savetxt("normal_method/dem_102/results/results.csv", results, fmt='%s', delimiter=',')
