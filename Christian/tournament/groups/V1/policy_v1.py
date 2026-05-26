"""
AgentV2 — Sin tabla de aperturas
=================================
Sub-estrategias activas:
  ✓ Detección inmediata (ganar/bloquear en 1 jugada)
  ✗ Tabla de aperturas
  ✓ Minimax Alpha-Beta con profundidad dinámica (5→9)

Usado en entrega.ipynb para aislar el aporte de la tabla de aperturas.
"""

import numpy as np
from connect4.policy import Policy

ROWS = 6
COLS = 7

DEPTH_SCHEDULE = [(10, 5), (20, 6), (30, 7), (42, 9)]


def _valid_cols(board):
    return [c for c in range(COLS) if board[0, c] == 0]


def _drop(board, col, player):
    new = board.copy()
    for r in reversed(range(ROWS)):
        if new[r, col] == 0:
            new[r, col] = player
            break
    return new


def _check_win(board, player):
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r, c + i] == player for i in range(4)): return True
    for r in range(ROWS - 3):
        for c in range(COLS):
            if all(board[r + i, c] == player for i in range(4)): return True
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r + i, c + i] == player for i in range(4)): return True
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            if all(board[r + i, c - i] == player for i in range(4)): return True
    return False


def _score_window(window, me):
    opp = -me
    me_c, opp_c, empty = window.count(me), window.count(opp), window.count(0)
    if me_c == 4:              return  1000.0
    if opp_c == 4:             return -1000.0
    if me_c == 3 and empty==1: return    10.0
    if me_c == 2 and empty==2: return     2.0
    if opp_c==3 and empty==1:  return   -15.0
    if opp_c==2 and empty==2:  return    -3.0
    return 0.0


def _score_board(board, me):
    score = 0.0
    opp = -me
    score += int(np.sum(board[:, COLS // 2] == me))  * 6
    score -= int(np.sum(board[:, COLS // 2] == opp)) * 6
    for r in range(ROWS):
        for c in range(COLS - 3):
            score += _score_window(list(board[r, c:c+4]), me)
    for c in range(COLS):
        for r in range(ROWS - 3):
            score += _score_window(list(board[r:r+4, c]), me)
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            score += _score_window([board[r+i, c+i] for i in range(4)], me)
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            score += _score_window([board[r+i, c-i] for i in range(4)], me)
    return score


def _minimax(board, depth, alpha, beta, maximizing, me):
    valid = _valid_cols(board)
    opp = -me
    terminal = _check_win(board, me) or _check_win(board, opp) or not valid
    if depth == 0 or terminal:
        if _check_win(board, me):  return  1_000_000_000.0, None
        if _check_win(board, opp): return -1_000_000_000.0, None
        if not valid:              return  0.0, None
        return _score_board(board, me), None
    ordered = sorted(valid, key=lambda c: abs(c - COLS // 2))
    if maximizing:
        best, best_col = -np.inf, ordered[0]
        for col in ordered:
            score, _ = _minimax(_drop(board, col, me), depth-1, alpha, beta, False, me)
            if score > best: best, best_col = score, col
            alpha = max(alpha, score)
            if alpha >= beta: break
        return best, best_col
    else:
        best, best_col = np.inf, ordered[0]
        for col in ordered:
            score, _ = _minimax(_drop(board, col, opp), depth-1, alpha, beta, True, me)
            if score < best: best, best_col = score, col
            beta = min(beta, score)
            if alpha >= beta: break
        return best, best_col


def _get_depth(board):
    pieces = int(np.sum(board != 0))
    for threshold, depth in DEPTH_SCHEDULE:
        if pieces < threshold:
            return depth
    return 9


class AgentV2(Policy):
    """V2 — Sin tabla de aperturas: detección inmediata + Minimax dinámico."""

    def __init__(self):
        self._me = -1

    def mount(self, timeout=None):
        self._me = -1

    def act(self, s: np.ndarray) -> int:
        neg, pos = int(np.sum(s == -1)), int(np.sum(s == 1))
        self._me = -1 if neg <= pos else 1
        opp = -self._me
        valid = _valid_cols(s)
        if not valid:
            return 0

        # Prioridad 0: ganar inmediatamente
        for col in valid:
            if _check_win(_drop(s, col, self._me), self._me):
                return col

        # Prioridad 1: bloquear victoria del oponente
        for col in valid:
            if _check_win(_drop(s, col, opp), opp):
                return col

        # Minimax con profundidad dinámica
        _, col = _minimax(s, _get_depth(s), -np.inf, np.inf, True, self._me)
        if col is None or col not in valid:
            col = min(valid, key=lambda c: abs(c - COLS // 2))
        return int(col)
