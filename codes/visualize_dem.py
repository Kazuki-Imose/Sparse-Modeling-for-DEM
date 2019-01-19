# -*- coding: utf-8 -*-
"""
DEM 配列を 3D モデルとして可視化する
Written in python3, created by Imose Kazuki on 2018/11/9
"""
import glob

import numpy as np
import plotly
import plotly.graph_objs as go


# 複数の DEM をファイルから読み込む関数
def read_dem(files_name):
    dem_data = []
    # データを dem に格納
    for file in files_name:
        for l in open(file).readlines():
            data = l[:-1].split(',')
            lst = list(data)
            lst2 = []
            for i in lst:
                i = float(i)
                lst2.append(i)
            dem_data.append(lst2)
        dem_data = np.array(dem_data)
    return dem_data


# DEM を可視化する関数
def visualize_dem(dem_data, title, filename):
    # 0を nan にする処理
    # dem = np.where(dem_data == 0, np.nan, dem_data)
    dem = dem_data
    data = [
        go.Surface(
            z=dem,
            colorscale="Earth"
            # 'Greys', 'YlGnBu', 'Greens', 'YlOrRd', 'Bluered', 'RdBu', 'Reds', 'Blues', 'Picnic', 'Rainbow',
            # 'Portland', 'Jet', 'Hot', 'Blackbody', 'Earth', 'Electric', 'Viridis', 'Cividis'
        )
    ]
    layout = go.Layout(
        title=title,
        autosize=False,
        width=1200,
        height=900,
        margin=dict(
            l=65,
            r=50,
            b=65,
            t=90
        )
    )
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename=filename, auto_open=False, image='png')


if __name__ == "__main__":
    # DEM ファイルの指定
    files = glob.glob('E:/CS_DEM/files/targetdem_001.txt')
    # DEM の読み込み
    dem = read_dem(files)
    dem = dem.T  # 転置しないとダメっぽい
    # DEM の可視化と保存
    visualize_dem(dem, title="DEM_001", filename="dem_001_2134.html")
