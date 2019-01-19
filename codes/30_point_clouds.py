# -*- coding: utf-8 -*-
"""
DEM に対して CS（圧縮センシング）を適用(wavelet transform)
Written in python3, created by Imose Kazuki on 2018/11/9
"""
import glob
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.io import imread
import math
import visualize_dem
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt


def point_clouds_collecting(files):

    data = np.array([[1, 2, 3]])
    i = 0
    for file in files:
        i += 1
        print(i)
        data_temp = pd.read_csv(file, header=None, delim_whitespace=True)
        data_temp = np.array(data_temp)
        data = np.vstack([data, data_temp])
    data = np.delete(data, 0, 0)


def extract_square(dataframe, list, file_name):
    """
    データフレームを任意の箇所で切り取る
    :param dataframe: pandas dataframe
    :param file_name: character
    :return:
    """
    # データフレームの切り取り
    # df = dataframe[(-76112 <= dataframe[0]) & (dataframe[0] <= -75600) &
    #                (-50700 <= dataframe[1]) & (dataframe[1] <= -50188)]
    df = dataframe[(list[0] <= dataframe[0]) & (dataframe[0] <= list[1]) &
                   (list[2] <= dataframe[1]) & (dataframe[1] <= list[3])]
    # ndarray に変換
    data = np.array(df)
    print("the number of points : ", data.shape[0])
    # txt ファイルとして保存
    # np.savetxt(file_name, data, delimiter=',')
    df.to_csv(file_name)


def average_method_to_point_clouds(dataframe, start, num_col, num_row):
    dem = np.zeros((num_row, num_col))
    for i in range(num_row):
        if i % 10 == 0:
            print(i, " / ", num_row)
        df_temp = dataframe[((start[0] + i) <= dataframe[0]) & (dataframe[0] <= (start[0] + i + 1))]
        for j in range(num_col):
            df_temp_temp = df_temp[((start[1] + j) <= df_temp[1]) & (df_temp[1] <= (start[1] + j + 1))]
            if len(df_temp_temp) != 0:
                dem[num_row - 1 - j, i] = np.mean(df_temp_temp[[2]].values.flatten())

    return dem


if __name__ == "__main__":
    time1 = time.clock()  # 時間の記録

    """ 複数ファイルの統合 """
    # # ファイルの指定
    # files = glob.glob('*_grd.txt')
    # data = np.array([[1, 2, 3]])
    # i = 0
    # for file in files:
    #     i += 1
    #     print(i)
    #     data_temp = pd.read_csv(file, header=None, delim_whitespace=True)
    #     data_temp = np.array(data_temp)
    #     data = np.vstack([data, data_temp])
    # data = np.delete(data, 0, 0)cd

    """ 対象エリアの切り取り """
    # ファイルの読み込み
    # df = pd.read_csv("3d_points_oniike.txt", header=None, delim_whitespace=True)
    # 対象エリアの切り取りおよび保存
    # target_area = [-76112], [-75600], [-50700], [-50188]  # X,Y(01)
    # target_area = [-76674], [-76162], [-50044], [-49532]  # X,Y(02)
    # extract_square(df, target_area,  "3d_points_oniike_02.txt")

    """ 平均法の適用 """
    # ファイルの読み込み
    # df = pd.read_csv("oniike_data/3d_points_oniike_01.txt", header=None, delim_whitespace=True)
    # num_row = 512  # 要変更
    # num_col = 512  # 要変更
    # start = [-76674, -50044]  # 要変更
    # dem = average_method_to_point_clouds(df, start, num_col, num_row)
    # # DEM の可視化と保存
    # demT = dem.T  # 転置しないとダメっぽい
    # visualize_dem.visualize_dem(demT, title="DEM_102", filename="dem_102.html")
    # # csv ファイルとして保存
    # np.savetxt("DEM_102.csv", dem, delimiter=',')

    """ 三次元点群データのサンプリング """
    # # ファイルの読み込み
    # df = pd.read_csv("3d_points_oniike.txt", header=None, delim_whitespace=True)
    # sample_rate = 0.1  # サンプリングの割合
    # df_sampled = df.sample(frac=sample_rate, random_state=0)
    # num_row = 512  # 要変更
    # num_col = 512  # 要変更
    # start = [-76112, -50700]  # 要変更
    # dem = average_method_to_point_clouds(df_sampled, start, num_col, num_row)
    # # DEM の可視化と保存
    # demT = dem.T  # 転置しないとダメっぽい
    # visualize_dem.visualize_dem(demT, title="DEM_101_sample2", filename="dem_101_sample2.html")
    # # csv ファイルとして保存
    # np.savetxt("DEM_101_sample2.csv", dem, delimiter=',')

    """ 深浅測量データ作成 """
    # # ファイルの読み込み
    # df = pd.read_csv("shinsen_data/oniike_shinsen_hyoko_raw_1.csv", header=None)
    # # 対象エリアの切り取りおよび保存
    # # target_area = [-76112], [-75600], [-50700], [-50188]  # X,Y(101)
    # # target_area = [-76674], [-76162], [-50044], [-49532]  # X,Y(102)
    # # target_area = [-76112], [-75822], [-50507], [-50210]  # X,Y(101_01)
    # # target_area = [-75885], [-75710], [-50350], [-50188]  # X,Y(101_02)
    # # target_area = [-76400], [-76222], [-49638], [-49600]  # X,Y(102_01)
    # # target_area = [-76230], [-76191], [-50044], [-49993]  # X,Y(102_02)
    # target_area = [-76327], [-76275], [-50044], [-49998]  # X,Y(102_03)
    # extract_square(df, target_area,  "shinsen_data/3d_points_oniike_shinsen_102_03.csv")

    """ 深浅測量データへの平均法の適用 """
    # # ファイルの読み込み
    # df = pd.read_csv("shinsen_data/3d_points_oniike_shinsen_101.txt", header=None, delim_whitespace=True)
    # num_row = 512  # 要変更
    # num_col = 512  # 要変更
    # start = [-76112, -50700]  # 要変更
    # dem = average_method_to_point_clouds(df, start, num_col, num_row)
    # # csv ファイルとして保存
    # np.savetxt("shinsen_data/shinsen_hyoko_dem_101.csv", dem, delimiter=',')

    """ 深浅測量データによる精度評価 """
    # # 深浅測量データの DEM ファイルの読込
    # shinsen_dem = np.array(pd.read_csv("files/DEM_101.csv", header=None))
    # # mask の作成
    # mask = np.where(shinsen_dem == 0, 0, 1)

    """ 深浅測量データにおける深さから標高を計算 """
    # # データの読込
    # df = pd.read_csv("shinsen_data/oniike_shinsen_raw.csv", encoding="shift-jis")
    # df['日時'] = df['日付'].str.cat(df['時間'], sep=' ')
    # df['日時'] = pd.to_datetime(df['日時'], format='%Y/%m/%d %H:%M:%S')
    # df['標高'] = 1.000000
    # print(df)
    # # 標高を計算
    # for i in range(len(df)):
    #     if i % 100 == 0:
    #         print(i)
    #     if df['日時'][i].hour == 9:
    #         df['標高'][i] = (150 + (36 * df['日時'][i].minute) / 60) / 100 - df['測得水深'][i]
    #     elif df['日時'][i].hour == 10:
    #         df['標高'][i] = (186 + (-7 * df['日時'][i].minute) / 60) / 100 - df['測得水深'][i]
    #     elif df['日時'][i].hour == 11:
    #         df['標高'][i] = (179 + (-48 * df['日時'][i].minute) / 60) / 100 - df['測得水深'][i]
    #     elif df['日時'][i].hour == 12:
    #         df['標高'][i] = (131 + (-73 * df['日時'][i].minute) / 60) / 100 - df['測得水深'][i]
    # # csv出力
    # df.to_csv("oniike_shinsen_hyoko_raw.csv")

    """ 鬼池深浅測量データに推定結果を追加 """
    # ファイルの読み込み
    df = pd.read_csv("shinsen_data/3d_points_oniike_shinsen_102_03.csv")
    # df['new_X'] = np.floor(df['X']) + 76112
    # df['new_Y'] = np.abs(np.ceil(df['Y']) + 50188)
    df['new_X'] = np.floor(df['X']) + 76674
    df['new_Y'] = np.abs(np.ceil(df['Y']) + 49532)
    num_df = len(df)
    print("the length of dataframe : ", num_df)
    # DEM の読み込み
    files = glob.glob('normal_method/dem_102/06_dem_102_ela.csv')  # ファイルの指定
    print(files)
    dems = []  # DEM を格納するリスト
    for file in files:
        dem_temp = np.array(pd.read_csv(file, header=None))
        dems.append(dem_temp)
    num_dems = len(dems)
    print("the number of DEMs : ", num_dems)
    # 推定結果の追加
    for i in range(num_dems):
        new_column = 'H_' + str(i)
        df[new_column] = 1.555555
        column = i + 5
        for j in range(num_df):
            df.iat[j, column] = dems[i][int(df.iat[j, 4]), int(df.iat[j, 3])]
        print(i + 1, " / ", num_dems, " finished")
    # csv出力
    df.to_csv("evaluation/oniike_shinsen_hyoko_102_03.csv")

    """ 鬼池深浅測量データに欠損あり DEM を追加 """
    # # ファイルの読み込み
    # df = pd.read_csv("shinsen_data/3d_points_oniike_shinsen_102_03.csv")
    # # df['new_X'] = np.floor(df['X']) + 76112
    # # df['new_Y'] = np.abs(np.ceil(df['Y']) + 50188)
    # df['new_X'] = np.floor(df['X']) + 76674
    # df['new_Y'] = np.abs(np.ceil(df['Y']) + 49532)
    # num_df = len(df)
    # print("the length of dataframe : ", num_df)
    # # DEM の読み込み
    # dem_data = np.array(pd.read_csv("normal_method/dem_102/DEM_102.csv", header=None))
    # dem = np.where(dem_data == 0, np.nan, dem_data)
    # # 推定結果の追加
    # new_column = 'H_existing'
    # df[new_column] = 1.555555
    # for j in range(num_df):
    #     df.iat[j, 5] = dem[int(df.iat[j, 4])][int(df.iat[j, 3])]
    # # csv出力
    # df.to_csv("evaluation/oniike_shinsen_hyoko_102_03_before.csv")

    """ 鬼池深浅測量データに従来手法の結果（TIFF）を追加 """
    # # ファイルの読み込み
    # df = pd.read_csv("shinsen_data/3d_points_oniike_shinsen_102_03.csv")
    # # df['new_X'] = np.floor(df['X']) + 76112
    # # df['new_Y'] = np.abs(np.ceil(df['Y']) + 50188)
    # df['new_X'] = np.floor(df['X']) + 76674
    # df['new_Y'] = np.abs(np.ceil(df['Y']) + 49532)
    # num_df = len(df)
    # print("the length of dataframe : ", num_df)
    # # ファイルの指定
    # files = glob.glob('conventional/dem_102/*_dem_102.tif')
    # print(files)
    # dems = []  # DEM を格納するリスト
    # for file in files:
    #     dem_temp = imread(file)
    #     dems.append(dem_temp)
    # num_dems = len(dems)
    # print("the number of DEMs : ", num_dems)
    # # 推定結果の追加
    # for i in range(num_dems):
    #     new_column = 'H_' + str(i)
    #     df[new_column] = 1.555555
    #     column = i + 5
    #     for j in range(num_df):
    #         df.iat[j, column] = dems[i][int(df.iat[j, 4]), int(df.iat[j, 3])]
    #     print(i + 1, " / ", num_dems, " finished")
    # # csv出力
    # df.to_csv("evaluation/oniike_shinsen_hyoko_102_03.csv")
    #
    # time2 = time.clock()
    # time_t = int(time2 - time1)
    # print("計算時間 : ", time_t, " 秒")
