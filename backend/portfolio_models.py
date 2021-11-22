import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as sp

import requests
import pathlib
import apimoex
import pandas as pd

from scipy import optimize as op
from sklearn.cluster import AgglomerativeClustering


def plot(data):
    plt.plot(data)
    plt.savefig('Graphic.png')


def getInformation(TICKs):
    board = 'TQBR'
    pathlib.Path("Database/{}".format(board)).mkdir(parents=True, exist_ok=True)
    process = 0
    with requests.Session() as session:
        for TICK in TICKs:
             process = process + 1
             data = apimoex.get_board_history(session, TICK, board=board)
             if data == []:
                 continue
             df = pd.DataFrame(data)
             df = df[['TRADEDATE','CLOSE']]
             df.to_excel("Database/{}/{}.xlsx".format(board,TICK), index=False)


def profits(data):
    profits = [(data[i] - data[i - 1]) / data[i - 1] for i in range(1, len(data))]
    return profits

def week(data):
    weeks_data = []
    for i in range(0, len(data) - 1, 5):
        weeks_data.append(data[i])
    return weeks_data

# n - какую строку дифференцируем, i - по какой переменной
def det_diff(i, Q, QUQ):
    q_i = Q.transpose()[i]
    return q_i.transpose().dot(QUQ).dot(q_i)


def khashinyan_target_gradient(Q, U):
    QUQ = np.linalg.inv(Q.dot(U).dot(Q.transpose()))
    return np.array([det_diff(i, Q.copy(), QUQ) for i in range(len(Q[0]))])


def khashinyan(frame):
    a = np.array(frame)
    Q = np.array([np.array([a[i][j] for j in range(len(a[i]))] + [1]) for i in range(len(a))]).transpose()
    m = len(Q[0])
    n = len(Q)
    u_0 = np.array([1] * m) / m
    u = u_0
    U_0 = np.eye(len(u_0)) * (u_0)
    U = U_0
    eps_0 = 10 ** (-10)
    iteration = 0
    max_iter = 1000
    while iteration < max_iter:
        U_0 = U
        u_0 = u
        gradient = khashinyan_target_gradient(Q, U)
        gradient_max = max(gradient)
        max_index = 0
        for i in range(len(gradient)):
            if gradient[i] == gradient_max:
                max_index = i
                break
        eps = (gradient_max - n) / n
        if abs(eps) < abs(eps_0):
            break
        alpha = (gradient_max - n) / (n * (gradient_max - 1))
        er = np.array([0 + 1 * (i == max_index) for i in range(m)])
        u = (1 - alpha) * u_0 + alpha * er
        U = np.eye(m) * (u)

        iteration += 1
    p = u * n
    P = np.eye(len(p)) * (p)
    B_l = np.linalg.inv(Q.dot(P).dot(Q.transpose()))
    B = B_l[0:n - 1, 0:n - 1]
    b = B_l[n - 1, 0:n - 1]
    b_d = B_l[n - 1][n - 1]
    M = B / (1 + np.linalg.inv(B).dot(b).dot(b) - b_d)
    c = -np.linalg.inv(B).dot(b)
    return M, c


def radius_klaster(m, data):
    return max([sp.euclidean(m, i) for i in data])


def cons_klaster(z, C, m, r):
    return z.dot(m) - r * np.sqrt(C.dot(z).dot(z))


def cons_x(z):
    I = np.array([1] * (len(z) - 1) + [0])
    return I.dot(z)


def object_function(z, delta):
    z_c = np.array([-z[i] for i in range(len(z) - 1)] + [1 / delta])
    return -z.dot(z_c)


def g(x, m, r, C):
    return m - r * C.dot(x) / np.sqrt(C.dot(x).dot(x))


def f(x, m, r, C):
    return x.dot(m) - r * np.sqrt(abs(C.dot(x).dot(x)))


def gradient_down(x_0, m, r, C):
    x = x_0.copy()
    x_k = x.copy()
    iteration = 1
    max_iter = 2000
    eps = 0.00001
    while iteration == 1 or sp.euclidean(x, x_k) > eps and iteration < max_iter:
        x = x_k
        lambd = 1 / iteration

        min_i = 0
        min_f = f(x, m[0], r[0], C[0])
        for i in range(1, len(m)):
            if f(x, m[i], r[i], C[i]) < min_f:
                min_f = f(x, m[i], r[i], C[i])
                min_i = i

        gradient = g(x, m[min_i], r[min_i], C[min_i])

        x_k = x - lambd * gradient
        if sum(x_k) != 1:
            x_k = x_k - 1 / len(x_k) * (sum(x_k) - 1)

        iteration += 1
    return x_k

# TICKs - тикеры акций
def runModel(TICKs, minKlaster, maxKlaster, deposit, criterion, selectModel, filename = ''):
    maxKlaster = maxKlaster + 1

    if not filename:
        filename = 'portfolio'

    if selectModel == 'Робастная модель с параметром':
        return runFirstModel(TICKs, minKlaster, maxKlaster, deposit, criterion, filename)
    elif selectModel == 'Робастная модель и метод эллипсоида минимального объема':
        return runSecondModel(TICKs, minKlaster, maxKlaster, deposit, criterion, filename)
    elif selectModel == 'Модель Марковица':
        return runClassicModel(TICKs, deposit, criterion, filename)


def runClassicModel(TICKs, deposit, criterion, filename):
    getInformation(TICKs)
    data = []
    tickers = []
    data_close = []
    deposit = int(deposit)
    for TICK in TICKs:
        file = 'Database/TQBR/' + TICK + '.xlsx'
        try:
            read = pd.read_excel(file).dropna().reset_index()
            data_ticker = [float(read['CLOSE'][i]) for i in range(len(read['CLOSE']))]
            data_close.append(data_ticker)
            data.append(list(profits(data_ticker)))
            tickers.append(TICK)
        except:
            pass

    min_amount_infromation = min([len(i) for i in data])
    data = [list(data[i][len(data[i]) - min_amount_infromation:]) for i in range(len(data))]
    frame = pd.DataFrame({TICKs[i]: data[i] for i in range(len(TICKs))})
    number_of_stoks = len(TICKs)
    data_close = [list(data_close[i][len(data_close[i]) - min_amount_infromation:]) for i in range(len(data_close))]
    frame_close = pd.DataFrame({TICKs[i]: data_close[i] for i in range(len(TICKs))})
    min_price = [0] * number_of_stoks
    lambdas = np.arange(0, 1.9, 0.2)

    def objective_function(x, frame, lambd):
        return -(frame.mean().dot(x) - lambd * (frame.cov().dot(x)).dot(x))

    frame_teach = frame[:int(len(frame) * 0.75)]

    frame_close_teach = frame_close[:int(len(frame) * 0.75)]
    frame_close_test = frame_close[int(len(frame) * 0.75):]

    x = np.zeros(number_of_stoks)
    results = pd.DataFrame(index=[lambdas], columns=[i for i in range(1, 2)])
    linear_constr = op.LinearConstraint([[1 for i in range(number_of_stoks)]], [1], [1])

    constraints = [linear_constr]
    bound = op.Bounds([min_price[i] / deposit for i in range(number_of_stoks)],
                      [np.inf for i in range(number_of_stoks)])
    best_x = []

    frame_teach_prime = frame_teach
    best_criterion = -1
    best_number_clasters = 0
    frame_teach = frame_teach_prime
    cluster = AgglomerativeClustering(n_clusters=1, affinity='euclidean', linkage='average')
    predict = cluster.fit_predict(frame_teach)
    frame_teach.loc[:, 'target'] = predict

    f = frame_teach.groupby(frame_teach['target'])
    clasters = []
    for _, g in f:
        clasters.append(g.drop('target', axis=1))
    for lambd in lambdas:
        i = 0
        for claster in clasters:
            i += 1
            if len(claster) > 3:
                res = op.minimize(objective_function, x0=x, args=(claster, lambd), bounds=bound,
                                  constraints=constraints, method='trust-constr')
                res_x = res.x
                stoks = res_x * deposit / frame_close_teach.iloc[-1]

                if criterion == 'Средний доход':
                    cur_criterion = sum(
                        [(frame_close_test.iloc[i].dot(stoks) - frame_close_test.iloc[i - 1].dot(stoks)) /
                         frame_close_test.iloc[i - 1].dot(stoks) for i in
                         range(1, len(frame_close_test))]) / len(frame_close_test)
                else:
                    cur_criterion = min(
                        np.array([frame_close_test.iloc[i].dot(stoks) for i in range(len(frame_close_test))]))

                if best_criterion < cur_criterion:
                    best_x = res_x
                    best_criterion = cur_criterion
                    best_number_clasters = 1

        print(lambd, ': succesful!')
        results.loc[lambd, 1] = best_criterion
    frame_close_test = frame_close.iloc[int(len(frame_close) * 0.75):]
    stoks = best_x * deposit / frame_close_teach.iloc[-1]
    mas = np.array([frame_close_test.iloc[i].dot(stoks) for i in range(len(frame_close_test))])
    plot(mas)

    print(best_criterion)
    if criterion == 'Средний доход':
        best_criterion = str(round(best_criterion * 100, 3)) + '%'
    else:
        best_criterion = str(round(best_criterion, 3))

    best_x = {TICKs[i]: [round(best_x[i] * 100, 3)] for i in range(len(TICKs))}

    x = pd.DataFrame(best_x)
    x = pd.concat([x, results])

    x.to_excel(filename + '_classic.xlsx')

    doc = open(filename + '_classic.xlsx', 'rb')
    doc.close()

    return best_x, best_criterion


def runFirstModel(TICKs, minKlaster, maxKlaster, deposit, criterion, filename):
    getInformation(TICKs)
    data = []
    tickers = []
    data_close = []
    for TICK in TICKs:
        file = 'Database/TQBR/' + TICK + '.xlsx'
        try:
            read = pd.read_excel(file).dropna().reset_index()
            data_ticker = [float(read['CLOSE'][i]) for i in range(len(read['CLOSE']))]
            data_close.append(data_ticker)
            data.append(list(profits(data_ticker)))
            tickers.append(TICK)
        except:
            pass

    min_amount_infromation = min([len(i) for i in data])
    data = [list(data[i][len(data[i]) - min_amount_infromation:]) for i in range(len(data))]
    frame = pd.DataFrame({TICKs[i]: data[i] for i in range(len(TICKs))})
    number_of_stoks = len(TICKs)
    data_close = [list(data_close[i][len(data_close[i]) - min_amount_infromation:]) for i in range(len(data_close))]
    frame_close = pd.DataFrame({TICKs[i]: data_close[i] for i in range(len(TICKs))})
    min_price = [0] * number_of_stoks
    lambdas = np.arange(0, 1.9, 0.2)

    def objective_function(x, frame, lambd):
        return -(frame.mean().dot(x) - lambd * (frame.cov().dot(x)).dot(x))

    frame_teach = frame[:int(len(frame) * 0.75)]

    frame_close_teach = frame_close[:int(len(frame) * 0.75)]
    frame_close_test = frame_close[int(len(frame) * 0.75):]

    x = np.zeros(number_of_stoks)
    results = pd.DataFrame(index=[lambdas], columns=[i for i in range(minKlaster, maxKlaster)])
    linear_constr = op.LinearConstraint([[1 for i in range(number_of_stoks)]], [1], [1])

    constraints = [linear_constr]
    bound = op.Bounds([min_price[i] / deposit for i in range(number_of_stoks)],
                      [np.inf for i in range(number_of_stoks)])
    best_x = []

    frame_teach_prime = frame_teach
    best_criterion = -1
    best_number_clasters = 0
    for number_of_clasters in range(minKlaster, maxKlaster):
        frame_teach = frame_teach_prime
        cluster = AgglomerativeClustering(n_clusters=number_of_clasters, affinity='euclidean', linkage='average')
        predict = cluster.fit_predict(frame_teach)
        frame_teach.loc[:, 'target'] = predict

        f = frame_teach.groupby(frame_teach['target'])
        clasters = []
        for _, g in f:
            clasters.append(g.drop('target', axis=1))
        for lambd in lambdas:
            i = 0
            for claster in clasters:
                i += 1
                if len(claster) > 3:
                    res = op.minimize(objective_function, x0=x, args=(claster, lambd), bounds=bound,
                                      constraints=constraints, method='trust-constr')
                    res_x = res.x
                    stoks = res_x * deposit / frame_close_teach.iloc[-1]

                    if criterion == 'Средний доход':
                        cur_criterion = sum(
                            [(frame_close_test.iloc[i].dot(stoks) - frame_close_test.iloc[i - 1].dot(stoks)) /
                             frame_close_test.iloc[i - 1].dot(stoks) for i in
                             range(1, len(frame_close_test))]) / len(frame_close_test)
                    else:
                        cur_criterion = min(
                            np.array([frame_close_test.iloc[i].dot(stoks) for i in range(len(frame_close_test))]))

                    if best_criterion < cur_criterion:
                        best_x = res_x
                        best_criterion = cur_criterion
                        best_number_clasters = number_of_clasters

            print(lambd, ': succesful!')
            results.loc[lambd, number_of_clasters] = best_criterion
    frame_close_test = frame_close.iloc[int(len(frame_close) * 0.75):]
    stoks = best_x * deposit / frame_close_teach.iloc[-1]
    mas = np.array([frame_close_test.iloc[i].dot(stoks) for i in range(len(frame_close_test))])
    plot(mas)

    if criterion == 'Средний доход':
        best_criterion = str(round(best_criterion * 100, 3)) + '%'
    else:
        best_criterion = str(round(best_criterion, 3))

    best_x = {TICKs[i]: [round(best_x[i] * 100, 3)] for i in range(len(TICKs))}
    x = pd.DataFrame(best_x)
    x = pd.concat([x, results])
    x.to_excel(filename + '_parameter.xlsx')

    doc = open(filename + '_parameter.xlsx', 'rb')
    doc.close()

    return best_x, best_criterion


def runSecondModel(TICKs, minKlaster, maxKlaster, deposit, criterion, filename):
    getInformation(TICKs)
    data = []
    tickers = []
    data_close = []
    for TICK in TICKs:
        file = 'Database/TQBR/' + TICK + '.xlsx'
        try:
            read = pd.read_excel(file).dropna().reset_index()
            data_ticker = [float(read['CLOSE'][i]) for i in range(len(read['CLOSE']))]
            data_close.append(data_ticker)
            data.append(list(profits(data_ticker)))
            tickers.append(TICK)
        except:
            pass

    min_amount_infromation = min([len(i) for i in data])
    data = [list(data[i][len(data[i]) - min_amount_infromation:]) for i in range(len(data))]
    frame = pd.DataFrame({TICKs[i]: data[i] for i in range(len(TICKs))})
    number_of_stoks = len(TICKs)

    data_close = [list(data_close[i][len(data_close[i]) - min_amount_infromation:]) for i in range(len(data_close))]
    frame_close = pd.DataFrame({TICKs[i]: data_close[i] for i in range(len(TICKs))})
    min_price = [0] * number_of_stoks

    frame_close_teach = frame_close[:int(len(frame) * 0.75)]
    frame_close_test = frame_close[int(len(frame) * 0.75):]

    best_x = []
    best_criterion = -1
    best_number_clasters = 0
    results = pd.DataFrame(columns=[i for i in range(minKlaster, maxKlaster)])
    for number_of_clasters in range(minKlaster, maxKlaster):
        frame_teach = frame.iloc[0:int(len(frame) * 0.75)]
        frame_close_teach = frame_close.iloc[0:int(len(frame_close) * 0.75)]
        frame_close_test = frame_close.iloc[int(len(frame_close) * 0.75):]

        cluster = AgglomerativeClustering(n_clusters=number_of_clasters, affinity='euclidean', linkage='average')
        predict = cluster.fit_predict(frame_teach)
        frame_teach.loc[:, 'target'] = predict

        target = [i for i in range(0, number_of_clasters)]
        clasters = []
        for j in range(len(target)):
            claster = [frame_teach.drop('target', axis=1).iloc[i] for i in range(len(frame_teach)) if
                       frame_teach.iloc[i]['target'] == target[j]]
            if len(claster) > number_of_stoks:
                clasters.append(claster)
        # clasters = np.array(clasters)
        clasters_frames = [pd.DataFrame(i) for i in clasters]
        M = np.array([np.linalg.inv(khashinyan(i)[0]) for i in clasters_frames])
        m_c = np.array([khashinyan(i)[1] for i in clasters_frames])
        r = [radius_klaster(m_c[i], clasters[i]) for i in range(len(m_c))]

        x = gradient_down(np.array([0.1] * number_of_stoks), m_c, r, M)

        stoks = x * deposit / frame_close_teach.iloc[-1]
        if criterion == 'Средний доход':
            cur_criterion = sum(
                [(frame_close_test.iloc[i].dot(stoks) - frame_close_test.iloc[i - 1].dot(stoks)) /
                 frame_close_test.iloc[i - 1].dot(stoks) for i in
                 range(1, len(frame_close_test))]) / len(frame_close_test)
        else:
            cur_criterion = min(
                np.array([frame_close_test.iloc[i].dot(stoks) for i in range(len(frame_close_test))]))
        results.loc[number_of_clasters] = cur_criterion
        if cur_criterion > best_criterion:
            best_criterion = cur_criterion
            best_number_clasters = number_of_clasters
            best_x = x

    stoks = best_x * deposit / frame_close_teach.iloc[-1]
    mas = np.array([frame_close_test.iloc[i].dot(stoks) for i in range(len(frame_close_test))])

    plot(mas)

    if criterion == 'Средний доход':
        best_criterion = str(round(best_criterion * 100, 3)) + '%'
    else:
        best_criterion = str(round(best_criterion, 3))

    best_x = {TICKs[i]: [round(best_x[i] * 100, 3)] for i in range(len(TICKs))}
    x = pd.DataFrame(best_x)
    x = pd.concat([x, results])
    x.to_excel(filename + '_ellips.xlsx')

    doc = open(filename + '_ellips.xlsx', 'rb')
    doc.close()

    return best_x, best_criterion