#!/bin/bash

# 출력 디렉토리 생성
output_dir="./fig/MultiPixels"
mkdir -p "$output_dir"

# 0부터 15까지 반복
for col in {0..15}; do
    echo "열 $col 처리 중..."
    python3.13 InjectionGraph_MultiPixel.py ~/cernbox/AstroPixv4_ANL/InjectionScan/HV200_THR130 -R 0 12 -C $col $col
done

echo "모든 처리가 완료되었습니다."
