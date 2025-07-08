# Normalization_with_K-means_Clustering

## 실험 조건
1. K-means clustering 알고리즘의 중심점 초기값은 랜덤으로 지정한다.
2. 초기에 지정된 중심점 초기값은 모든 실험에 동일하게 적용한다. (단, 정규화한 결과의 경우 초기값도 정규화하여 적용한다.)
3. K-means clustering 알고리즘의 학습은 최대 100세대까지 진행되며 중심점 이동이 없는 경우에 중단된다.
4. 학습이 끝나면 학습 결과를 그래프로 보여준다.
5. 학습 결과 그래프에서 같은 군집은 같은 색의 점으로 표현하고, 중심점은 흑색으로 표현한다.
<br></br>
## 실험 결과
![1x1](https://github.com/user-attachments/assets/0a850ea0-7a54-4578-9ea8-a0d5b5e3a714)  
정규화 이전 dataset_1_x를 x축, dataset_1_y를 y축으로 설정. (모두 분산 20300/3 적용)  
잘 군집화되지 않은 모습이 나타남.  
<br></br>
![2x2](https://github.com/user-attachments/assets/aeac5bbd-f95a-4bb7-a794-fe7304bbb104)  
정규화 이전 dataset_2_x를 x축, dataset_2_y를 y축으로 설정. (모두 분산 20003/300 적용)  
중심점 위치를 볼 때 잘 군집화되지 않은 모습이 나타남.  
<br></br>
![1x2](https://github.com/user-attachments/assets/67243ea0-6f27-4447-bdf6-365a460fd63d)  
정규화 이전 dataset_1_x를 x축, dataset_2_y를 y축으로 설정. (x축 자료 분산 20300/3, y축 자료 분산 20003/300 적용)  
잘 군집화되지 않은 모습이 나타남.  
<br></br>
![1x1_norm](https://github.com/user-attachments/assets/72d26f80-d816-4bf2-a9a4-80529b216e82)  
정규화 이후 dataset_1_x를 x축, dataset_1_y를 y축으로 설정. (모두 표준편차 1 적용)  
정규화 이전과 비교하여 군집화 실력이 향상됨.  
<br></br>
![2x2_norm](https://github.com/user-attachments/assets/c99400ad-6b91-44cb-84a6-f14e249690e5)  
정규화 이후 dataset_2_x를 x축, dataset_2_y를 y축으로 설정. (모두 표준편차 1 적용)  
중심점 좌표를 고려하면 정규화 이전과 비교하여 군집화 실력이 향상됨.  
<br></br>
![1x2_norm](https://github.com/user-attachments/assets/a54d3e2f-bb50-4579-a2f0-9897b786b7d3)  
정규화 이후 dataset_1_x를 x축, dataset_2_y를 y축으로 설정. (모두 표준편차 1 적용)  
정규화 이전과 비교하여 군집화 실력이 향상됨.  
<br></br>
## 실험 결과 분석
정규화 이전에 제대로 군집화를 하지 못하던 모델이 정규화 이후 뛰어난 군집화 성능을 발휘했다.  
성능 향상은 타 예시들에 비해 dataset_1_x, dataset_2_x를 비교한 실험에서 크게 발생했다.  
따라서 서로 표준편차가 다른 2가지 변수에 따라 군집화를 수행할 때 정규화가 2가지 변수의 중요도를 비슷하게 맞춰준다는 점을 알 수 있다.
