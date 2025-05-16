import re

def extract_params_from_log(filepath):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            if len(lines) < 7:
                raise ValueError("Log file has fewer than 7 lines.")
            line = lines[6].strip()

        # 'Namespace(...)' 안의 내용을 추출
        match = re.search(r'Namespace\((.*)\)', line)
        if not match:
            print(line)
            raise ValueError("No 'Namespace(...)' pattern found.")
        content = match.group(1)
        print(content)

        # 각 key=value 형태를 개별적으로 파싱
        # 리스트 값 같이 복잡한 형태를 위해 정규식 개선
        param_pattern = re.findall(r"(\w+)=((?:\[.*?\])|(?:'.*?')|(?:\d+\.\d+)|(?:\d+))", content)

        # 문자열, 숫자, 리스트 형태 처리
        def parse_value(value):
            if value.startswith('['):
                return eval(value)  # 리스트 형태일 때만 eval 허용
            elif value.startswith("'") and value.endswith("'"):
                return value.strip("'")
            elif '.' in value:
                return float(value)
            else:
                return int(value)

        # 딕셔너리로 정리
        parsed = {k: parse_value(v) for k, v in param_pattern}

        # 원하는 키만 추출
        keys_to_extract = ['threshold', 'vinj', 'inject', 'inject_period', 'maxtime']
        return {k: parsed.get(k) for k in keys_to_extract}

    except Exception as e:
        print(f"Error: {e}")
        return None

