"""Compatibility for checkpoints saved as curriculum_ppo.CurriculumPolicy."""
import sys
from ai2.training import curriculum_ppo as _implementation

sys.modules[__name__] = _implementation
