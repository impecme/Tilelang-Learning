def main() -> None:
    print("main() is executing.")
    print("__name__ inside this module:", __name__)


def explain_entry_guard() -> None:
    print("if __name__ == '__main__' is an entry guard.")
    print("Direct run: __name__ == '__main__', so main() executes.")
    print("Imported by another file: __name__ is the module path, so main() does not auto-run.")


# 入口保护：
# - 直接运行 python3 task8.py 时，__name__ 是 "__main__"，会执行 main()。
# - 被其它文件 import 时，__name__ 是模块名，不会自动执行 main()。
# 这样一个文件既可以当脚本运行，也可以被其它文件复用其中的函数。
if __name__ == "__main__":
    explain_entry_guard()
    main()
