#!/usr/bin/env python3
"""공용 시각화 유틸리티 함수."""


def fix_mpl_toolkits():
    """시스템 mpl_toolkits (3.6)와 venv matplotlib (3.10) 충돌 해결.

    matplotlib를 import한 후, mpl_toolkits.mplot3d를 사용하기 전에 호출한다.
    """
    import sys
    import pathlib
    import matplotlib

    for key in [k for k in list(sys.modules.keys()) if k.startswith('mpl_toolkits')]:
        del sys.modules[key]

    import mpl_toolkits
    venv_path = str(pathlib.Path(matplotlib.__file__).parent.parent / 'mpl_toolkits')
    mpl_toolkits.__path__ = [venv_path]

    from mpl_toolkits.mplot3d import Axes3D
    matplotlib.projections.projection_registry.register(Axes3D)
