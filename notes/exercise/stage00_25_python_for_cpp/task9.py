from pathlib import Path
import subprocess
import sys


def main() -> None:
    # 第 9 题要求运行 Stage 00.25 的配套脚本。
    # 这里用 subprocess 从当前练习文件定位到项目根目录，再运行：
    # python3 scripts/python_for_cpp_smoke.py
    repo_root = Path(__file__).resolve().parents[3]
    smoke_script = repo_root / "scripts" / "python_for_cpp_smoke.py"

    print("Running:", sys.executable, smoke_script, flush=True)
    subprocess.run([sys.executable, str(smoke_script)], check=True)


if __name__ == "__main__":
    main()
