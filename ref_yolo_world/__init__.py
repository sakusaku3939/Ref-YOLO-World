"""Ref-YOLO-World: referring expression comprehension on top of YOLO-World.

Importing this package registers every proposed module in the MMEngine
registry. Configs use ``custom_imports = dict(imports=['ref_yolo_world.modules',
'ref_yolo_world.models'], ...)``.
"""

from .modules import *  # noqa: F401,F403
from .models import *  # noqa: F401,F403
