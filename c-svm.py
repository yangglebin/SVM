#! /opt/local/bin/python2.7
# -"- encoding: utf-8 -"-

# ------ package, module --------
# import matplotlib
# matplotlib.use('GTK')
import numpy as np
from scipy.spatial.distance import cdist
import cvxopt
import cvxopt.solvers
# import matplotlib as mpl
# mpl.use('PDF')
import matplotlib.pyplot as plt
# ---- package, module END -----


# --- function 関数 -------------------------
# カーネル関数 を行列形式で実装 -> グラム行列を直接高速に生成
def gram(kerntype, X, Y, SIGMA):
    P = 3  # 多項式カーネルのパラメータ
    # 関数を辞書に登録
    kerndic = {
        "linear": (lambda: np.dot(X, Y.T)),
        "poly": (lambda: (1+np.dot(X, Y.T))**P),
        "gauss": (lambda:
                  np.exp(-cdist(X, Y, metric='sqeuclidean')
                  / (2*(SIGMA**2)))),
    }
    if kerntype in kerndic.keys():
        return kerndic[kerntype]()
    else:
        print('invarid kerntype')


# 二次形式
def quad(x, K):
    leftX = x * K  # x.T*Q
    rightX = leftX.T * x  # Q*x
    return rightX


# ラグランジュ乗数を二次計画法で求める
def QPwrapper(Kt, N, t, C):
    Q = cvxopt.matrix(Kt)
    p = cvxopt.matrix(-np.ones(N))
    temp1 = np.diag([-1.0]*N)
    temp2 = np.identity(N)
    G = cvxopt.matrix(np.vstack((temp1, temp2)))
    temp1 = np.zeros(N)
    temp2 = np.ones(N) * C
    h = cvxopt.matrix(np.hstack((temp1, temp2)))
    A = cvxopt.matrix(t, (1, N))
    b = cvxopt.matrix(0.0)
    sol = cvxopt.solvers.qp(Q, p, G, h, A, b)  # 二次計画法
    # ラグランジュ乗数 a を返す
    return np.array(sol['x']).reshape(N)


# サポートベクトルのインデックスを抽出
def supVec(a):
    return a > 1e-5


# 識別超平面
def discrimPlain(x, a, t, X, K, kerntype, SIGMA):
    s = supVec(a)
    temp = np.dot(K[:, s], a[s]*t[s])
    b = np.sum(t[s] - temp[s])/np.count_nonzero(s)
    plain = np.dot(gram(kerntype, x, X, SIGMA), a*s*t)
    return plain + b


# 人工データ
def artData(N, kerntype):
    cls1 = []
    cls2 = []
    mean1 = [-1, 2]
    mean2 = [1, -1]
    cov = [[1.0, 0.8], [0.8, 1.0]]
    np.random.seed(seed=1)
    if kerntype == "linear":
        cls1.extend(np.random.multivariate_normal(mean1, cov, N/2))
        cls2.extend(np.random.multivariate_normal(mean2, cov, N/2))
    else:
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cls1.extend(np.random.multivariate_normal(mean1, cov, N/4))
        cls1.extend(np.random.multivariate_normal(mean3, cov, N/4))
        cls2.extend(np.random.multivariate_normal(mean2, cov, N/4))
        cls2.extend(np.random.multivariate_normal(mean4, cov, N/4))
    X = np.vstack((cls1, cls2))
    t = []
    for i in range(N/2):
        t.append(1.0)
    for i in range(N/2):
        t.append(-1.0)
    t = np.array(t)
    return X, t


# 訓練データのロード
def loadData():
    data = np.genfromtxt("classification.txt")
    X = data[:, 0:2]
    t = data[:, 2] * 2 - 1.0  # 教師信号を -1 or 1 に 変換
    return X, t


# プロット
def ploting(a, t, X, K, N, kerntype, C, SIGMA):
    # 識別境界を描画
    X1, X2 = np.meshgrid(np.linspace(-3, 3, 150), np.linspace(-3, 3, 150))
    w, h = X1.shape
    X1.resize(X1.size)
    X2.resize(X2.size)
    Z = np.array([discrimPlain(np.array([[x1, x2]]),
                 a, t, X, K, kerntype, SIGMA) for (x1, x2) in zip(X1, X2)])
    X1.resize((w, h))
    X2.resize((w, h))
    Z.resize((w, h))
    plt.pcolor(X1, X2, Z)
    plt.colorbar()
    plt.contour(X1, X2, Z,
                np.append(np.arange(1, 20, 0.2), np.arange(-20, -1.01, 0.2)),
                colors='k', linewidths=0.3, origin='lower')
    plt.contour(X1, X2, Z, np.arange(-1, 1.01, 0.2), colors='k',
                linewidths=0.6, origin='lower')
    plt.contour(X1, X2, Z, [0.0], colors='w',
                linewidths=1.2, origin='lower')
    # データ点プロット
    for n in range(N):
        if t[n] > 0:
            plt.scatter(X[n, 0], X[n, 1], c='r', marker='x')
        else:
            plt.scatter(X[n, 0], X[n, 1], c='b', marker='x')
    # サポートベクター
    # plt.scatter(X[supVec(a), 0], X[supVec(a), 1],
    #             s=40, c='c', marker='o')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    figname = './demos/C-svm_C_' + str(C) + '_SIGMA_' + str(SIGMA) + '.pdf'
    plt.savefig(figname)
    plt.close()
    # plt.show()


# --- メイン関数 -------------------------
if __name__ == "__main__":
    C = 2000  # スラック変数にかかるペナルティパラメータ
    SIGMA = 0.5  # ガウスカーネルのパラメータ
    kerntype = str(raw_input(' Input Kernel Type > '))  # kernel type
    # kerntype = "linear"
    # データ読み込み
    # data = np.load('dataXYT.npy')
    # N = np.size(data, axis=0)
    # ラベル
    # t = data[:, 2]
    # データ点座標
    # X = data[:, 0:2]
    # X, t = artData(N, kerntype)
    X, t = loadData()
    N = len(t)
    # データのグラム行列を作成
    K = gram(kerntype, X, X, SIGMA)
    # 二次形式
    Kt = quad(t, K)
    # ラグランジュ乗数を二次計画法で求める
    a = QPwrapper(Kt, N, t, C)
    # プロット
    ploting(a, t, X, K, N, kerntype, C, SIGMA)
