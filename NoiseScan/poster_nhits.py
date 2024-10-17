import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 샘플 데이터프레임 생성 (Vbb, THR, hit 갯수)
data = {
    'Vbb': [-150, -150, -150, -200, -250],
    'THR': [400, 300, 200, 200, 200],
    'hits': [74733, 114643, 143881, 176751, 205781]
}

df = pd.DataFrame(data)

# X축 라벨을 'Vbb = -150 V & THR = 200 mV' 형식으로 생성
df['label'] = df.apply(lambda row: f"$V_{{BB}}$ = {row['Vbb']} V \n THR = {row['THR']} mV", axis=1)

# 그래프 그리기
plt.figure(figsize=(8, 6))
plt.scatter(df['label'], df['hits'], color='black', s=80)

#plt.xlabel("Vbb & THR Setup", fontsize=16)  # x축 라벨 크기 조정
plt.ylabel("Number of Hits per 30 mins", fontsize=16)  # y축 라벨 크기 조정
#plt.xticks(rotation=45, ha='right', fontsize=14)  # X축 라벨 회전 및 크기 조정
plt.xticks(fontsize=14)  # X축 라벨 회전 및 크기 조정
plt.yticks(fontsize=14)  # y축 라벨 크기 조정
#plt.title("Hit Counts for Different Vbb and THR Setups", fontsize=18)  # 제목 크기 조정

# 그래프 보여주기
plt.tight_layout()
plt.show()