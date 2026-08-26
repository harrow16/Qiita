"""
orjson_patch.py
orjson が AppLocker でブロックされる環境向けのモンキーパッチ。
標準ライブラリの json で orjson の最低限のインターフェースを再現し、
sys.modules に差し込むことで import orjson を差し替える。

使い方:
    import orjson_patch  # quantum_add.py の先頭で import するだけ
"""

import sys
import json
import types


class _OrjsonModule(types.ModuleType):
    """orjson の dumps / loads を標準 json で代替する擬似モジュール。"""

    # orjson.dumps は bytes を返す
    @staticmethod
    def dumps(obj, default=None, option=None) -> bytes:
        return json.dumps(obj, default=default, ensure_ascii=False).encode("utf-8")

    # orjson.loads は bytes / str どちらも受け取れる
    @staticmethod
    def loads(data) -> object:
        if isinstance(data, (bytes, bytearray)):
            data = data.decode("utf-8")
        return json.loads(data)

    __version__ = "3.0.0-compat"


# sys.modules に差し込む（以降 import orjson はこのモジュールを参照する）
sys.modules["orjson"] = _OrjsonModule("orjson")
