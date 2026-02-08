"""Core modules for Timed Petri Net system."""

#__init__.py 是首先会执行的，加上下面这句话之后，
#from core.config import xx等价于from core import xx
# 加上这句话后，可以不知道内部结构就引用这两个类
from .config import PetriConfig, ModuleSpec
from .action_enable import ActionEnableChecker
from .transition_fire import TransitionFireExecutor

# from core import * 会导入下面这些类
__all__ = ['PetriConfig', 'ModuleSpec', 'ActionEnableChecker', 'TransitionFireExecutor']

