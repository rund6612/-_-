import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

csv_path = './Resource_Management_Process.csv'
df = pd.read_csv(csv_path)
###DoW의 값을 숫자로 매핑
day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

dowReplace_df = df
# DoW 열의 값을 변경
dowReplace_df['DoW'] = dowReplace_df['DoW'].map(day_mapping)

# 'Man_Cost', 'Power_Usage' 열을 제외한 데이터프레임 생성
dowReplace_df_without_Man_Cost_Power_Usage = dowReplace_df.drop(columns=['Man_Cost', 'Power_Usage'])

# 상관 계수 계산
corr= dowReplace_df_without_Man_Cost_Power_Usage.corr(numeric_only=True)

# 시각화
corr.style.background_gradient(cmap='coolwarm')
#y종속변수인 Man_Cost와 x간의 상관 계수 분석
corr_Man_Cost = dowReplace_df[['Man_Cost', 'Production', 'Temperature', 'Humidity', 'Power_Cost', 'DoW', 'Worker_Power']].corr().iloc[0]

# 결과 출력
print(corr_Man_Cost)


columns_to_drop1 = ['Temperature', 'Power_Cost', 'Humidity']
columns_to_drop = ['Temperature', 'Power_Cost', 'Humidity']
semifinal_df = dowReplace_df.drop(columns=columns_to_drop)

semifinal_df = dowReplace_df.drop(columns=columns_to_drop)

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 숫자형 데이터만 선택
numeric_columns = dowReplace_df.select_dtypes(include=[np.number])

# 박스 플롯을 4개씩 한 행에 배치하여 출력
num_plots = len(numeric_columns.columns)
num_rows = (num_plots - 1) // 4 + 1

fig, axes = plt.subplots(num_rows, 4, figsize=(15, 4 * num_rows))

for i, column in enumerate(numeric_columns.columns):
    row = i // 4
    col = i % 4
    ax = axes[row, col]
    sns.boxplot(data=dowReplace_df[column], ax=ax)
    ax.set_xlabel(column)

    # 이상치 개수 계산
    outliers_index, outliers_count = detect_and_count_outliers(dowReplace_df, column)
    ax.text(0.95, 0.95, f"Outliers: {outliers_count}", verticalalignment='top', horizontalalignment='right', transform=ax.transAxes, color='red', fontsize=8)

# 마지막으로 그리드의 빈 플롯을 숨기기
for i in range(num_plots, num_rows * 4):
    row = i // 4
    col = i % 4
    fig.delaxes(axes[row, col])

plt.tight_layout()
plt.show()


temperature_outliers, temperature_outliers_count = detect_and_count_outliers(dowReplace_df, 'Temperature')
Production_outliers, Production_outliers_count = detect_and_count_outliers(dowReplace_df, 'Production')
Power_Cost_outliers, Power_Cost_outliers_count = detect_and_count_outliers(dowReplace_df, 'Power_Cost')
Worker_Power_outliers, Worker_Power_outliers_count = detect_and_count_outliers(dowReplace_df, 'Worker_Power')
#이상치 개수
print("Temperature 이상치의 개수:", temperature_outliers_count)
print("Production 이상치의 개수:", Production_outliers_count)
print("Power_Cost 이상치의 개수:", Power_Cost_outliers_count)
print("Worker_Power 이상치의 개수:", Worker_Power_outliers_count)
#이상치의 행
print("Temperature 이상치의 행 index:", temperature_outliers)
print("Production 이상치의 행 index:", Production_outliers)
print("Power_Cost 이상치의 행 index:", Power_Cost_outliers)
print("Worker_Power 이상치의 행 index:", Worker_Power_outliers)


from collections import Counter

# 이상치의 행 index 모으기
all_outliers = np.concatenate([temperature_outliers, Production_outliers, Power_Cost_outliers, Worker_Power_outliers])

# 각 행의 등장 횟수 세기
outliers_counter = Counter(all_outliers)

# Temperature 열의 이상치가 2번 이상 겹치는 행과 그 횟수 추출
common_outliers = {index: count for index, count in outliers_counter.items() if count >= 2 and index in temperature_outliers}

# 행 번호를 리스트에 담기
common_outliers_list = list(common_outliers.keys())

print("Temperature 열의 이상치가 2번 이상 겹치는 이상치의 행 번호와 횟수:")
for index, count in common_outliers.items():
    print(f"행 번호: {index}, 횟수: {count}")

# 행 번호 리스트 출력
print("행 번호 리스트:", common_outliers_list)
