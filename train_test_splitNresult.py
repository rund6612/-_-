import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# CSV 파일 경로

file_path = "./전처리_0508.csv"

# CSV 파일 불러오기
df = pd.read_csv(file_path)

#Train, Test 데이터 분리
df = df.sample(frac=1)
train_len = round(len(df)*0.7)
train_dataframe = df[:train_len]
test_dataframe = df[train_len:]
train_Power_usage=list(train_dataframe['Power_Usage'])
test_Power_usage=list(test_dataframe['Power_Usage'])
train_man_cost = list(train_dataframe['Man_Cost'])
train_power_cost = list(train_dataframe['Power_Cost'])
test_man_cost = list(test_dataframe['Man_Cost'])
test_power_cost = list(test_dataframe['Power_Cost'])
drop_col_list = ['Datetime', 'Man_Cost', 'Power_Usage', 'Power_Cost']


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 'Power_Usage' 열을 출력 변수(y)로 설정하고 나머지를 입력 변수(X)로 설정합니다.
X_train = train_dataframe.drop(drop_col_list,axis=1)
y_train = train_dataframe['Power_Usage']
X_test = test_dataframe.drop(drop_col_list,axis=1)
y_test = test_dataframe['Power_Usage']

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 스케일링된 데이터와 원래의 타겟 값을 출력해 확인합니다.
print("X_train_scaled:\n", X_train_scaled)
print("y_train:\n", y_train)
print("X_test_scaled:\n", X_test_scaled)
print("y_test:\n", y_test)

# 모델 생성
model = Sequential()

# 입력층과 은닉층 추가
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))

# 출력층 추가 (연속형 데이터를 예측하기 때문에 활성화 함수 사용 안 함)
model.add(Dense(1))

# 모델 컴파일
model.compile(loss='mean_squared_error', optimizer='adam')

# 모델 학습
model.fit(X_train_scaled, y_train, epochs=50, batch_size=10, validation_split=0.2)

# 모델 평가
loss = model.evaluate(X_test_scaled, y_test)
print(f'Mean Squared Error on Test Data: {loss}')

#train_result 산출
train_result = model.predict(X_train_scaled)

# 예측
predictions = model.predict(X_test_scaled)

# 예측 결과와 실제 값 비교
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': predictions.flatten()})
# print(comparison.head())

# 예측 결과와 실제 값 비교
comparison1 = pd.DataFrame({'Actual': y_train, 'Predicted': train_result.flatten()})
print(comparison1.head())

comparison1 = comparison1.head(100)
plt.figure(figsize=(10, 5))
plt.plot(comparison1['Actual'].values, color='red', label='Actual', linestyle='-')
plt.plot(comparison1['Predicted'].values, color='blue', label='Predicted', linestyle='-')
plt.title('Actual vs Predicted Power Usage')
plt.xlabel('Sample Index')
plt.ylabel('Power Usage')
plt.legend()
plt.show()

#test_result
test_result = model.predict(X_test_scaled)

# 예측 결과와 실제 값 비교
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': test_result.flatten()})
print(comparison.head())

comparison = comparison.head(100)
plt.figure(figsize=(10, 5))
plt.plot(comparison['Actual'].values, color='red', label='Actual', linestyle='-')
plt.plot(comparison['Predicted'].values, color='blue', label='Predicted', linestyle='-')
plt.title('Actual vs Predicted Power Usage')
plt.xlabel('Sample Index')
plt.ylabel('Power Usage')
plt.legend()
plt.show()

from ortools.linear_solver import pywraplp
import numpy as np

solution_list = []
work_list = []
sol_work_list = []

for i, cost in enumerate(test_man_cost):
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        print("Solver not created")
        continue

    # 변수 정의: x는 10 이상인 정수 변수
    x = solver.IntVar(10, solver.infinity(), 'x')

    # cost와 test_result[i]의 합을 상수로 정의
    cost_value = cost if not isinstance(cost, np.ndarray) else cost.item()
    test_result_value = test_result[i] if not isinstance(test_result[i], np.ndarray) else test_result[i].item()
    total_cost = float(cost_value + test_result_value)
    power_usage_cost = float(test_power_cost[i] * test_Power_usage[i])

    # 목적 함수: 비용을 최소화
    solver.Minimize(x * total_cost)

    # 제약 조건 추가
    solver.Add(x * total_cost >= power_usage_cost)
    solver.Add(x >= 10)

    # 솔버 실행
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        solution_list.append(x.solution_value())
        work_list.append(power_usage_cost)
        sol_work_list.append(int(x.solution_value() + 1) * total_cost)
    else:
        print(f"Solver did not find an optimal solution for index {i}")

# # 결과 출력
# print("Solutions:", solution_list)
# print("Work List:", work_list)
# print("Solution Work List:", sol_work_list)
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(sol_work_list[:100],color='red',label='model output')
ax.plot(work_list[:100],label='label')
ax.set_xlabel('Optimization solutions')
ax.set_ylabel('Product Label')
ax.set_title('Linear programming result')
ax.xaxis.label.set_fontsize(15)
ax.yaxis.label.set_fontsize(15)
ax.title.set_fontsize(15)
plt.legend(fontsize=15)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 데이터 준비
test_data_x = test_dataframe.drop(drop_col_list, axis=1)
train_data_x = train_dataframe.drop(drop_col_list, axis=1)
y_test = test_dataframe['Power_Usage']
y_train = train_dataframe['Power_Usage']

# 랜덤 포레스트 모델 학습 및 예측
rf_reg = RandomForestRegressor(max_depth=5)
rf_reg.fit(train_data_x, y_train)
train_result_rf = rf_reg.predict(train_data_x)
test_result_rf = rf_reg.predict(test_data_x)

# 랜덤 포레스트 평가
rf_train_mse = mean_squared_error(y_train, train_result_rf)
rf_test_mse = mean_squared_error(y_test, test_result_rf)
print(f"Random Forest Train MSE: {rf_train_mse}")
print(f"Random Forest Test MSE: {rf_test_mse}")

# 인공신경망 모델의 예측 결과를 이미 얻은 상태로 가정
# test_result_nn = 인공신경망 모델의 예측 결과
# train_result_nn = 인공신경망 모델의 학습 데이터 예측 결과

import matplotlib.pyplot as plt

# 평가 지표 계산
rf_mae = mean_absolute_error(y_test, test_result_rf)
rf_r2 = r2_score(y_test, test_result_rf)

# 인공신경망 평가
nn_train_mse = mean_squared_error(y_train, train_result)
nn_test_mse = mean_squared_error(y_test, test_result)
nn_mae = mean_absolute_error(y_test, test_result)
nn_r2 = r2_score(y_test, test_result)

print("Random Forest Performance:")
print(f"Mean Squared Error: {rf_test_mse}")
print(f"Mean Absolute Error: {rf_mae}")
print(f"R-squared: {rf_r2}")

print("\nNeural Network Performance:")
print(f"Mean Squared Error: {nn_test_mse}")
print(f"Mean Absolute Error: {nn_mae}")
print(f"R-squared: {nn_r2}")

# 시각화
plt.figure(figsize=(12, 6))

# 실제 값과 예측 값 비교
plt.subplot(1, 2, 1)
plt.plot(y_test.values[:100], label='Actual', color='blue', marker='o')
plt.plot(test_result_rf[:100], label='RF Predicted', color='green', linestyle='dashed', marker='x')
plt.plot(test_result[:100], label='NN Predicted', color='red', linestyle='dashed', marker='x')
plt.title('Actual vs Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Power Usage')
plt.legend()

# 에러 비교
plt.subplot(1, 2, 2)
plt.bar(['Random Forest', 'Neural Network'], [rf_test_mse, nn_test_mse], color=['green', 'red'])
plt.title('Mean Squared Error Comparison')
plt.ylabel('MSE')

plt.tight_layout()
plt.show()
