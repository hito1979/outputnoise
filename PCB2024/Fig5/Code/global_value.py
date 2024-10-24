import numpy as np
import global_value as g  #g.を変数の前につければ別ファイルでも参照できる共通のグローバル変数となる
#global 変数
############################################
#目的関数の変数
g.W = 2*np.pi
g.T = 1.0
g.e = np.power(10, -1.0)
g.D = 3.0
g.a = 1.0
g.b = 1.0
g.k = 10.0
#g.list_tcp = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
g.list_tcp = np.linspace(0.0, 1.0, 100)

############################################
#その他グローバル関数
g.best_values = [] #列:0 最良値, 1~ A_1,A_2...B_1,B_2... 行:世代
g.repetition = 1000
g.threshold_std = 1e-6
g.threshold_diff = 1e-7