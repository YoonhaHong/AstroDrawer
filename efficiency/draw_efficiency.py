import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plottwo_vs_frequency(y1, y2, directory_path, colors):
    # 모든 csv 파일 찾기
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found.")
        return

    # 색상 설정 (파일 수만큼)

    # 그래프 설정
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()  # 오른쪽 y축

    for idx, file_name in enumerate(csv_files):
        full_path = os.path.join(directory_path, file_name)
        try:
            df = pd.read_csv(full_path)

            color = colors[idx]
            print(color)

            # 왼쪽 y축: efficiency
            ax1.plot(df['frequency'], df[y1], label=f'{file_name[0:-4]} - {y1}', color=color, marker='o', linestyle='-', alpha=0.7)

            # 오른쪽 y축: purity
            ax2.plot(df['frequency'], df[y2], label=f'{file_name[0:-4]} - {y2}', color=color, marker='^', linestyle='--', alpha=0.7)

        except Exception as e:
            print(f"Error reading {file_name}: {e}")

    # 축 라벨 및 제목
    ax1.set_xlabel("Frequency")
    ax1.set_ylabel(y1)
    ax2.set_ylabel(y2)
    plt.title(f"{y1}&{y2} vs Frequency")

    # 범례 통합 표시
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    plt.legend(lines_1 + lines_2, labels_1 + labels_2,
               title = "W02S03, HV=-150 V, THR = 250 mV, injection@r10c10 w 300 mV",
               loc='lower right', ncol=2)

    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"./{y1}&{y2}.png")
    plt.show()

colors = ['red', 'blue', 'green', 'orange', 'purple']
plottwo_vs_frequency("efficiency", "purity", "/home/npl/AstroPix/data/injection", colors)
plottwo_vs_frequency("nhit", "ninjhit", "/home/npl/AstroPix/data/injection", colors )
plottwo_vs_frequency("nhit", "nevent", "/home/npl/AstroPix/data/injection", colors )