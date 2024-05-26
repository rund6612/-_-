import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

csv_path = 'C:/Users/in/Desktop/Resource_Management_Process.csv'
df = pd.read_csv(csv_path)

###DoW의 값을 숫자로 매핑
day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

dowReplace_df = df
# DoW 열의 값을 변경
dowReplace_df['DoW'] = dowReplace_df['DoW'].map(day_mapping)

#Power_Cost는 Power_Usage를 예측하는데 도움이 안될것이라 판단
dropped_df = dowReplace_df.drop(columns=['Power_Cost'])

# 이상치를 식별하여 개수와 함께 출력하는 함수
def detect_and_count_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers.index, len(outliers)
    
# 숫자형 데이터만 선택
numeric_columns = dropped_df.select_dtypes(include=[np.number])

# 박스 플롯을 4개씩 한 행에 배치하여 출력
num_plots = len(numeric_columns.columns)
num_rows = (num_plots - 1) // 4 + 1

fig, axes = plt.subplots(num_rows, 4, figsize=(15, 4 * num_rows))

for i, column in enumerate(numeric_columns.columns):
    row = i // 4
    col = i % 4
    ax = axes[row, col]
    sns.boxplot(data=dropped_df[column], ax=ax)
    ax.set_xlabel(column)

    # 이상치 개수 계산
    outliers_index, outliers_count = detect_and_count_outliers(dropped_df, column)
    ax.text(0.95, 0.95, f"Outliers: {outliers_count}", verticalalignment='top', horizontalalignment='right', transform=ax.transAxes, color='red', fontsize=8)

# 마지막으로 그리드의 빈 플롯을 숨기기
for i in range(num_plots, num_rows * 4):
    row = i // 4
    col = i % 4
    fig.delaxes(axes[row, col])

// plt.tight_layout()
// plt.show()

##이상치 제거
temperature_outliers, temperature_outliers_count = detect_and_count_outliers(dropped_df, 'Temperature')
Production_outliers, Production_outliers_count = detect_and_count_outliers(dropped_df, 'Production')
Worker_Power_outliers, Worker_Power_outliers_count = detect_and_count_outliers(dropped_df, 'Worker_Power')

from collections import Counter
# 이상치의 행 index 모으기
all_outliers = np.concatenate([temperature_outliers, Production_outliers, Power_Cost_outliers, Worker_Power_outliers])
# all_outliers = np.concatenate([temperature_outliers, Production_outliers, Worker_Power_outliers])
# 각 행의 등장 횟수 세기
outliers_counter = Counter(all_outliers)

# Temperature 열의 이상치가 2번 이상 겹치는 행과 그 횟수 추출
common_outliers = {index: count for index, count in outliers_counter.items() if count >= 2 and index in temperature_outliers}

# 행 번호를 리스트에 담기
common_outliers_list = list(common_outliers.keys())

cleaned_df = dropped_df.drop(common_outliers_list)

