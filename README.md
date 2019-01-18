# SparseModelingforTerrain
These program codes were the implementations of this article.
This article file is in the top directory.

本研究では，手法の適用・検証にあたり，いくつかのプログラムコードを作成した．作成した主なプログラムコードを以下に示す．

・dictionary_learning.py

K-SVD法やDCT法，DCT with ElasticNet法を含むスパースコーディングによる欠損補間手法を用いる際に使用する関数が入っているモジュール


・compressed_sensing.py

Fourier 正則化法やWavelet 正則化法を含む圧縮センシングおよびTV最小化法による欠損補間手法を用いる際に使用する関数が入っているモジュール


・conventional.py

従来手法である平均法による欠損補間手法を用いる際に使用する関数が入っているモジュール


・visualize_dem.py

DEMファイルの読み込みや三次元可視化の際に使用する関数が入っているモジュール


・その他各手法の実行ファイル

各手法を実行するファイルであり，ファイル名に"main"が含まれる．
このファイルと，以上に挙げたモジュールを同じディレクトリに入れることで，処理を行うことができる



