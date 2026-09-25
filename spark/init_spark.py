"""
spark/init_spark.py
Environment configuration and bootstrap for PySpark on Windows/Linux.
Resolves Windows path spaces via 8.3 short names and sets HADOOP_HOME / PYSPARK_PYTHON.
"""

import os
import sys

def init_environment():
    # 1. UTF-8 console output
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # 2. Convert to 8.3 short path on Windows to avoid space issues in batch scripts
    def get_short_path(path):
        if sys.platform == "win32" and path and os.path.exists(path):
            try:
                import ctypes
                from ctypes import wintypes
                _GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
                _GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
                _GetShortPathNameW.restype = wintypes.DWORD
                output_buf_size = 0
                while True:
                    output_buf = ctypes.create_unicode_buffer(output_buf_size)
                    needed = _GetShortPathNameW(path, output_buf, output_buf_size)
                    if output_buf_size >= needed:
                        return output_buf.value
                    else:
                        output_buf_size = needed
            except Exception:
                return path
        return path

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hadoop_dir = os.path.join(base_dir, "hadoop")
    if os.path.exists(hadoop_dir):
        short_hadoop = get_short_path(hadoop_dir)
        os.environ["HADOOP_HOME"] = short_hadoop
        os.environ["PATH"] = os.path.join(short_hadoop, "bin") + os.pathsep + os.environ.get("PATH", "")

    short_python = get_short_path(sys.executable)
    os.environ["PYSPARK_PYTHON"] = short_python
    os.environ["PYSPARK_DRIVER_PYTHON"] = short_python

    if "JAVA_HOME" in os.environ:
        clean_java = os.environ["JAVA_HOME"].rstrip("\\/")
        os.environ["JAVA_HOME"] = get_short_path(clean_java)

    try:
        import findspark
        findspark.init()
    except Exception:
        pass

init_environment()
