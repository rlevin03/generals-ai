"""
Game package: core Generals.io game logic and rendering.
"""

from .generals import (
    Game,
    GameRenderer,
    GRID_WIDTH,
    GRID_HEIGHT,
    CellType,
    Cell,
    Player,
    MoveCommand,
    PLAYER_COLORS,
)

__all__ = [
    "Game",
    "GameRenderer",
    "GRID_WIDTH",
    "GRID_HEIGHT",
    "CellType",
    "Cell",
    "Player",
    "MoveCommand",
    "PLAYER_COLORS",
]
