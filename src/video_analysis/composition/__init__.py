"""Video-to-music composition planning and SongSpec compilation."""

from .compiler import compile_song_spec
from .models import CompositionPlan

__all__ = ["CompositionPlan", "compile_song_spec"]
