import os, sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import glob
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utility import yaml_reader  # 상위 폴더의 utility 모듈에서 yaml_reader 함수를 가져옴

NCOL = 16
NROW = 13
plt.style.use('classic')

def main(args):
    # CSV 파일 읽기
    dffpair = pd.read_csv(args.inputfile)
    
    # 'col'과 'row' 열을 기반으로 히트 수 계산
    dfpair = dffpair[['col', 'row']].copy()
    dfpairc = dfpair[['col', 'row']].value_counts().reset_index(name='hits')
    
    # YAML 파일에서 마스킹된 픽셀 정보 가져오기
    file_name = os.path.basename(args.inputfile).rstrip('.csv')
    dir_name = os.path.dirname(args.inputfile)
    date = file_name.split('_')[1]  # 파일 이름에서 날짜 부분 추출
    findyaml = f"{dir_name}/*{date}*.yml"
    yamlpath = glob.glob(findyaml)
    
    if not yamlpath:
        print(f"No YAML file found for date: {date}")
        return
    
    disablepix = yaml_reader(yamlpath[0])  # YAML 파일 읽기
    pixs = pd.DataFrame(disablepix, columns=['col', 'row', 'disable'])  # 마스킹된 픽셀 정보를 DataFrame으로 변환
    
    # 히트맵과 마스킹 맵을 겹쳐서 그리기
    fig, ax = plt.subplots(figsize=(8, 6))
      
    # 마스킹 맵 그리기 (히트맵 위에 겹쳐서 그리기)
    p2 = ax.hist2d(
        x=pixs['col'], 
        y=pixs['row'], 
        bins=[NCOL, NROW], 
        range=[[0, NCOL], [0, NROW]], 
        weights=pixs['disable'], 
        norm=matplotlib.colors.Normalize(vmin=0, vmax=1), 
        cmap='Greys'
    )  
    # 히트맵 그리기
    p1 = ax.hist2d(
        x=dfpairc['col'], 
        y=dfpairc['row'], 
        bins=[NCOL, NROW], 
        range=[[0, NCOL], [0, NROW]], 
        weights=dfpairc['hits'], 
        cmap='YlOrRd', 
        cmin=1.0, 
        norm=matplotlib.colors.LogNorm()
    )

    
    # 그래프 설정
    ax.set_xlabel('Col', fontweight='bold', fontsize=13)
    ax.set_ylabel('Row', fontweight='bold', fontsize=13)
    ax.xaxis.set_tick_params(labelsize=13)
    ax.yaxis.set_tick_params(labelsize=13)
    ax.set_title('Hit Map with Masked Pixels', fontweight='bold', fontsize=16)
    ax.set_aspect('equal')
    ax.grid()
    
    fig.tight_layout()  # 서브플롯 자동 정렬
    fig.subplots_adjust(right=0.85)  # colorbar 공간 확보
    cbar = fig.colorbar(p1[3], ax=ax, fraction=0.046, pad=0.04)  
    cbar.set_label(label='Hit Counts', weight='bold', size=13)
    # 결과 저장
    figdir = args.outdir if args.outdir else dir_name
    os.makedirs(figdir, exist_ok=True)
    plt.savefig(f"{figdir}/{file_name}_hitmap_with_masked.png")
    print(f"Saved at {figdir}/{file_name}_hitmap_with_masked.png")
    # 창 유지
    #plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Hit Map with Masked Pixels from CSV file.')
    parser.add_argument("inputfile", type=str, help='Path to the input CSV file')
    parser.add_argument('-o', '--outdir', default="./fig", help='Output directory for the plot')
    
    args = parser.parse_args()
    main(args)