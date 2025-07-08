# 첫 번째 데이터셋의 경우 합쳐놨을 때 평균 100, 분산 20300/3 나옴. 이대로 normalize 함수 이용해서 계산식 적용하기.



from math import sqrt
from random import normalvariate, uniform

import numpy as np
from matplotlib import pyplot as plt


class MiddlePos:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.values = []
    
    def step(self, dataset_x, dataset_y):
        if len(self.values) > 0:
            sum_x = 0
            sum_y = 0
            for i in self.values:
                sum_x += dataset_x[i]
                sum_y += dataset_y[i]
            self.x = sum_x / len(self.values)
            self.y = sum_y / len(self.values)
            self.values = []


def classify(model, dataset_x, dataset_y, data_size):
    loss = 0
    for i in range(data_size):
        min_point = [0, (dataset_x[i] - model[0].x) ** 2 + (dataset_y[i] - model[0].y) ** 2]
        for point in range(1, len(model)):
            dist = (dataset_x[i] - model[point].x) ** 2 + (dataset_y[i] - model[point].y) ** 2
            if dist < min_point[1]:
                min_point = [point, dist]
        model[min_point[0]].values.append(i)
        loss += min_point[1]
    return loss


def train(model, dataset_x, dataset_y, epoch, data_size=12000):
    loss = -1
    val = [[] for _ in range(len(model))]
    for i in range(epoch):
        prev = loss
        loss = classify(model, dataset_x, dataset_y, data_size)
        print(f"Loss = {loss} @ epoch {i + 1}")
        if loss == prev:
            break
        for m in range(len(model)):
            val[m] = model[m].values
            model[m].step(dataset_x, dataset_y)
    
    return val


def normalize(arr, mu, sigma):
    normalized = []
    for i in arr:
        normalized.append((i - mu) / sigma)
    return normalized


dataset_1_x, dataset_1_y = np.random.normal(0, 10, 4000), np.random.normal(0, 10, 4000)
for i in range(2):
    dataset_1_x = np.hstack((dataset_1_x, np.random.normal((i + 1) * 100, 10, 4000)))
    dataset_1_y = np.hstack((dataset_1_y, np.random.normal((i + 1) * 100, 10, 4000)))
dataset_1_x_norm, dataset_1_y_norm = normalize(dataset_1_x, 100, sqrt(20300 / 3)), normalize(dataset_1_y, 100, sqrt(20300 / 3))

dataset_2_x, dataset_2_y = np.random.normal(0, 0.1, 4000), np.random.normal(0, 0.1, 4000)
for i in range(2):
    dataset_2_x = np.hstack((dataset_2_x, np.random.normal((i + 1) * 100, 0.01, 4000)))
    dataset_2_y = np.hstack((dataset_2_y, np.random.normal((i + 1) * 100, 0.01, 4000)))
dataset_2_x_norm, dataset_2_y_norm = normalize(dataset_2_x, 100, sqrt(20003 / 300)), normalize(dataset_2_y, 100, sqrt(20003 / 300))

init_middle_points = [MiddlePos(uniform(-100, 300), uniform(-100, 300)) for _ in range(3)]
sm, powered_sm = [0, 0], [0, 0]
for i in range(3):
    sm[0] += init_middle_points[i].x
    sm[1] += init_middle_points[i].y
    powered_sm[0] += init_middle_points[i].x ** 2
    powered_sm[1] += init_middle_points[i].y ** 2
mu = [sm[0] / 3, sm[1] / 3]
sigma = [sqrt(powered_sm[0] / 3 - mu[0] ** 2), sqrt(powered_sm[1] / 3 - mu[1] ** 2)]
init_middle_points_norm = [MiddlePos((init_middle_points[i].x - mu[0]) / sigma[0], (init_middle_points[i].y - mu[1]) / sigma[1]) for i in range(3)]

print(f"Starting Session 1x1")
middle_points = init_middle_points
plt.title("1x1")
groups = train(middle_points, dataset_1_x, dataset_1_y, 100)
group_1_x, group_1_y = [dataset_1_x[i] for i in groups[0]], [dataset_1_y[i] for i in groups[0]]
group_2_x, group_2_y = [dataset_1_x[i] for i in groups[1]], [dataset_1_y[i] for i in groups[1]]
group_3_x, group_3_y = [dataset_1_x[i] for i in groups[2]], [dataset_1_y[i] for i in groups[2]]
plt.scatter(group_1_x, group_1_y, s=0.1, c="blue")
plt.scatter(group_2_x, group_2_y, s=0.1, c="red")
plt.scatter(group_3_x, group_3_y, s=0.1, c="green")
pos_x, pos_y = [], []
for i in middle_points:
    pos_x.append(i.x)
    pos_y.append(i.y)
plt.scatter(pos_x, pos_y, c="black")
plt.show()

print(f"Starting Session 2x2")
middle_points = init_middle_points
plt.title("2x2")
groups = train(middle_points, dataset_2_x, dataset_2_y, 100)
group_1_x, group_1_y = [dataset_2_x[i] for i in groups[0]], [dataset_2_y[i] for i in groups[0]]
group_2_x, group_2_y = [dataset_2_x[i] for i in groups[1]], [dataset_2_y[i] for i in groups[1]]
group_3_x, group_3_y = [dataset_2_x[i] for i in groups[2]], [dataset_2_y[i] for i in groups[2]]
plt.scatter(group_1_x, group_1_y, s=0.1, c="blue")
plt.scatter(group_2_x, group_2_y, s=0.1, c="red")
plt.scatter(group_3_x, group_3_y, s=0.1, c="green")
pos_x, pos_y = [], []
for i in middle_points:
    pos_x.append(i.x)
    pos_y.append(i.y)
plt.scatter(pos_x, pos_y, c="black")
plt.show()

print(f"Starting Session 1x2")
middle_points = init_middle_points
plt.title("1x2")
groups = train(middle_points, dataset_1_x, dataset_2_y, 100)
group_1_x, group_1_y = [dataset_1_x[i] for i in groups[0]], [dataset_2_y[i] for i in groups[0]]
group_2_x, group_2_y = [dataset_1_x[i] for i in groups[1]], [dataset_2_y[i] for i in groups[1]]
group_3_x, group_3_y = [dataset_1_x[i] for i in groups[2]], [dataset_2_y[i] for i in groups[2]]
plt.scatter(group_1_x, group_1_y, s=0.1, c="blue")
plt.scatter(group_2_x, group_2_y, s=0.1, c="red")
plt.scatter(group_3_x, group_3_y, s=0.1, c="green")
pos_x, pos_y = [], []
for i in middle_points:
    pos_x.append(i.x)
    pos_y.append(i.y)
plt.scatter(pos_x, pos_y, c="black")
plt.show()

print(f"Starting Session 1x1 (Normalized)")
middle_points = init_middle_points_norm
plt.title("1x1 (Normalizezd)")
groups = train(middle_points, dataset_1_x_norm, dataset_1_y_norm, 100)
group_1_x, group_1_y = [dataset_1_x_norm[i] for i in groups[0]], [dataset_1_y_norm[i] for i in groups[0]]
group_2_x, group_2_y = [dataset_1_x_norm[i] for i in groups[1]], [dataset_1_y_norm[i] for i in groups[1]]
group_3_x, group_3_y = [dataset_1_x_norm[i] for i in groups[2]], [dataset_1_y_norm[i] for i in groups[2]]
plt.scatter(group_1_x, group_1_y, s=0.1, c="blue")
plt.scatter(group_2_x, group_2_y, s=0.1, c="red")
plt.scatter(group_3_x, group_3_y, s=0.1, c="green")
pos_x, pos_y = [], []
for i in middle_points:
    pos_x.append(i.x)
    pos_y.append(i.y)
plt.scatter(pos_x, pos_y, c="black")
plt.show()

print(f"Starting Session 2x2 (Normalized)")
middle_points = init_middle_points_norm
plt.title("2x2 (Normalized)")
groups = train(middle_points, dataset_2_x_norm, dataset_2_y_norm, 100)
group_1_x, group_1_y = [dataset_2_x_norm[i] for i in groups[0]], [dataset_2_y_norm[i] for i in groups[0]]
group_2_x, group_2_y = [dataset_2_x_norm[i] for i in groups[1]], [dataset_2_y_norm[i] for i in groups[1]]
group_3_x, group_3_y = [dataset_2_x_norm[i] for i in groups[2]], [dataset_2_y_norm[i] for i in groups[2]]
plt.scatter(group_1_x, group_1_y, s=0.1, c="blue")
plt.scatter(group_2_x, group_2_y, s=0.1, c="red")
plt.scatter(group_3_x, group_3_y, s=0.1, c="green")
plt.scatter(dataset_2_x_norm, dataset_2_y_norm, s=0.1)
pos_x, pos_y = [], []
for i in middle_points:
    pos_x.append(i.x)
    pos_y.append(i.y)
plt.scatter(pos_x, pos_y, c="black")
plt.show()

print(f"Starting Session 1x2 (Normalized)")
middle_points = init_middle_points_norm
plt.title("1x2 (Normalized)")
groups = train(middle_points, dataset_1_x_norm, dataset_2_y_norm, 100)
group_1_x, group_1_y = [dataset_1_x_norm[i] for i in groups[0]], [dataset_2_y_norm[i] for i in groups[0]]
group_2_x, group_2_y = [dataset_1_x_norm[i] for i in groups[1]], [dataset_2_y_norm[i] for i in groups[1]]
group_3_x, group_3_y = [dataset_1_x_norm[i] for i in groups[2]], [dataset_2_y_norm[i] for i in groups[2]]
plt.scatter(group_1_x, group_1_y, s=0.1, c="blue")
plt.scatter(group_2_x, group_2_y, s=0.1, c="red")
plt.scatter(group_3_x, group_3_y, s=0.1, c="green")
pos_x, pos_y = [], []
for i in middle_points:
    pos_x.append(i.x)
    pos_y.append(i.y)
plt.scatter(pos_x, pos_y, c="black")
plt.show()
