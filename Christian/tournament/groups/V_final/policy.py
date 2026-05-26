"""
MyAgent — Híbrido con Transposition Table + Heurística mejorada
================================================================
Mejoras sobre la versión anterior:

1. Transposition Table (TT) persistente por partida
   La TT se guarda entre jugadas del mismo juego. Esto permite
   que búsquedas posteriores reutilicen resultados ya calculados,
   lo que equivale en la práctica a 1-2 niveles extra de profundidad
   sin costo adicional de tiempo.

2. Heurística con bonus de altura de fila
   Las amenazas en filas bajas (fondo del tablero) son más valiosas
   porque son más difíciles de bloquear. Se aplica un multiplicador
   rb = 1.0 + (ROWS-1-r)*0.15 a las ventanas horizontales.

3. Profundidad ajustada al speedup real de la TT
   Schedule: < 10 piezas → depth 6, < 28 piezas → depth 6,
   >= 28 piezas → depth 7. La TT caliente compensa la diferencia
   con el depth 9 del schedule original.

4. Victoria rápida premiada en el score terminal (+ depth)
   El Minimax prefiere ganar en el turno más próximo posible,
   no dilatar partidas ya ganadas.

5. MoveTracker corregido
   La versión original tenía _extract_move_sequence siempre devolviendo None.
   Ahora reconstruye la secuencia correctamente al detectar diffs.

6. Tabla de aperturas con validación contra Minimax
   Si la jugada del libro es claramente peor que la mejor del motor
   (diferencia > 20 pts), se descarta y se usa el Minimax directamente.
"""

import numpy as np
from connect4.policy import Policy


ROWS = 6
COLS = 7

# ---------------------------------------------------------------------------
# TABLA DE APERTURAS
# ---------------------------------------------------------------------------

OPENING_BOOK: dict[tuple[int, ...], int] = {
    (): 3,
    (3,): 3,
    (3, 3): 2,
    (3, 2): 3,
    (3, 4): 3,
    (3, 1): 2,
    (3, 5): 4,
    (3, 0): 2,
    (3, 6): 4,
    (3, 3, 2): 4,
    (3, 3, 4): 2,
    (3, 2, 3): 2,
    (3, 4, 3): 4,
    (3, 1, 2): 3,
    (3, 5, 4): 3,
    (3, 0, 2): 3,
    (3, 6, 4): 3,
    (3, 2, 2): 3,
    (3, 4, 4): 3,
    (3, 1, 3): 2,
    (3, 5, 3): 4,
    (3, 3, 2, 4): 4,
    (3, 3, 4, 2): 2,
    (3, 3, 2, 2): 4,
    (3, 3, 4, 4): 2,
    (3, 3, 2, 3): 4,
    (3, 3, 2, 1): 4,
    (3, 3, 2, 5): 2,
    (3, 2, 3, 2): 4,
    (3, 4, 3, 4): 2,
}


def _mirror_col(col: int) -> int:
    return COLS - 1 - col


def _mirror_sequence(seq: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(_mirror_col(c) for c in seq)


def lookup_opening(move_sequence: tuple[int, ...]) -> int | None:
    if move_sequence in OPENING_BOOK:
        return OPENING_BOOK[move_sequence]
    mirrored = _mirror_sequence(move_sequence)
    if mirrored in OPENING_BOOK:
        return _mirror_col(OPENING_BOOK[mirrored])
    return None


# ---------------------------------------------------------------------------
# UTILIDADES DE TABLERO
# ---------------------------------------------------------------------------

def _valid_cols(board: np.ndarray) -> list[int]:
    return [c for c in range(COLS) if board[0, c] == 0]


def _drop(board: np.ndarray, col: int, player: int) -> np.ndarray:
    new = board.copy()
    for r in reversed(range(ROWS)):
        if new[r, col] == 0:
            new[r, col] = player
            break
    return new


def _check_win(board: np.ndarray, player: int) -> bool:
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r, c + i] == player for i in range(4)):
                return True
    for r in range(ROWS - 3):
        for c in range(COLS):
            if all(board[r + i, c] == player for i in range(4)):
                return True
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r + i, c + i] == player for i in range(4)):
                return True
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            if all(board[r + i, c - i] == player for i in range(4)):
                return True
    return False


def _count_threats(board: np.ndarray, player: int) -> int:
    """Cuántas ventanas de 3 propias + 1 vacío existen (amenazas reales)."""
    count = 0
    opp = -player

    def _is_threat(window: list) -> bool:
        return window.count(player) == 3 and window.count(0) == 1

    for r in range(ROWS):
        for c in range(COLS - 3):
            if _is_threat(list(board[r, c:c+4])):
                count += 1
    for r in range(ROWS - 3):
        for c in range(COLS):
            if _is_threat([board[r+i, c] for i in range(4)]):
                count += 1
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if _is_threat([board[r+i, c+i] for i in range(4)]):
                count += 1
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            if _is_threat([board[r+i, c-i] for i in range(4)]):
                count += 1
    return count


# ---------------------------------------------------------------------------
# HEURÍSTICA
# ---------------------------------------------------------------------------

def _score_window(window: list, me: int) -> float:
    opp = -me
    me_c  = window.count(me)
    opp_c = window.count(opp)
    empty = window.count(0)
    if me_c == 4:                return  1000.0
    if opp_c == 4:               return -1000.0
    if me_c == 3 and empty == 1: return    10.0
    if me_c == 2 and empty == 2: return     2.0
    if opp_c == 3 and empty == 1: return  -15.0
    if opp_c == 2 and empty == 2: return   -3.0
    return 0.0


def _score_board(board: np.ndarray, me: int) -> float:
    score = 0.0
    opp = -me

    # Bonus columnas centrales
    score += int(np.sum(board[:, 3] == me))  * 6
    score -= int(np.sum(board[:, 3] == opp)) * 6
    score += int(np.sum(board[:, 2] == me))  * 3
    score -= int(np.sum(board[:, 2] == opp)) * 3
    score += int(np.sum(board[:, 4] == me))  * 3
    score -= int(np.sum(board[:, 4] == opp)) * 3

    # Ventanas horizontales con bonus de altura (filas bajas = más valiosas)
    for r in range(ROWS):
        row_bonus = 1.0 + (ROWS - 1 - r) * 0.15
        for c in range(COLS - 3):
            score += _score_window(list(board[r, c:c+4]), me) * row_bonus

    # Verticales, diagonales (sin bonus de altura, ya tienen contexto)
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


# ---------------------------------------------------------------------------
# MINIMAX CON ALPHA-BETA Y TRANSPOSITION TABLE
# ---------------------------------------------------------------------------

def _minimax(
    board: np.ndarray,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    me: int,
    tt: dict,          # transposition table persistente por partida
) -> tuple[float, int | None]:

    # Consultar caché
    key = (board.tobytes(), depth, maximizing)
    if key in tt:
        return tt[key]

    valid = _valid_cols(board)
    opp = -me
    terminal = _check_win(board, me) or _check_win(board, opp) or not valid

    if depth == 0 or terminal:
        if _check_win(board, me):  result = (1_000_000_000.0 + depth, None)
        elif _check_win(board, opp): result = (-1_000_000_000.0 - depth, None)
        elif not valid:            result = (0.0, None)
        else:                      result = (_score_board(board, me), None)
        tt[key] = result
        return result

    ordered = sorted(valid, key=lambda c: abs(c - COLS // 2))

    if maximizing:
        best, best_col = -np.inf, ordered[0]
        for col in ordered:
            score, _ = _minimax(_drop(board, col, me), depth-1, alpha, beta, False, me, tt)
            if score > best:
                best, best_col = score, col
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        result = (best, best_col)
    else:
        best, best_col = np.inf, ordered[0]
        for col in ordered:
            score, _ = _minimax(_drop(board, col, opp), depth-1, alpha, beta, True, me, tt)
            if score < best:
                best, best_col = score, col
            beta = min(beta, score)
            if alpha >= beta:
                break
        result = (best, best_col)

    tt[key] = result
    return result


# ---------------------------------------------------------------------------
# MOVE TRACKER — reconstruye secuencia de jugadas real
# ---------------------------------------------------------------------------

class MoveTracker:
    def __init__(self):
        self._prev_board: np.ndarray | None = None
        self._sequence: list[int] = []

    def reset(self):
        self._prev_board = None
        self._sequence = []

    def update(self, board: np.ndarray) -> list[int]:
        if self._prev_board is None:
            self._prev_board = board.copy()
            # Si el tablero ya tiene piezas en el primer llamado, reconstruir
            if int(np.sum(board != 0)) > 0:
                self._sequence = _reconstruct_sequence(board)
            return list(self._sequence)

        diff = board.astype(int) - self._prev_board.astype(int)
        changed = list(zip(*np.where(diff != 0)))

        if len(changed) == 1:
            _, col = changed[0]
            self._sequence.append(int(col))

        self._prev_board = board.copy()
        return list(self._sequence)


def _reconstruct_sequence(board: np.ndarray) -> list[int]:
    """
    Reconstruye el orden de jugadas desde un tablero ya poblado.
    Solo fiable con pocas piezas (fase de apertura).
    """
    total = int(np.sum(board != 0))
    if total > 10:
        return []
    pieces = []
    for c in range(COLS):
        for r in reversed(range(ROWS)):
            if board[r, c] != 0:
                height = ROWS - 1 - r
                pieces.append((height, c))
    pieces.sort(key=lambda x: x[0])
    return [col for _, col in pieces]


# ---------------------------------------------------------------------------
# POLÍTICA HÍBRIDA — MyAgent
# ---------------------------------------------------------------------------

class MyAgent(Policy):
    """
    Agente híbrido mejorado.

    Árbol de decisión por prioridad:
      0. Ganar inmediatamente
      1. Bloquear victoria inmediata del oponente
      2. Crear doble amenaza propia (fork)
      3. Bloquear doble amenaza del oponente
      4. Tabla de aperturas validada contra Minimax (primeras 8 piezas)
      5. Minimax Alpha-Beta con Transposition Table persistente
    """

    # Profundidad según piezas en el tablero.
    # La TT acumulada entre jugadas equivale en la práctica a 1-2 niveles extra.
    DEPTH_SCHEDULE = [
        (28, 6),   # apertura y medio juego: depth 6 + TT caliente
        (42, 7),   # endgame: depth 7 (menos columnas libres → más rápido)
    ]

    def __init__(self):
        self._me: int = -1
        self._tracker = MoveTracker()
        self._tt: dict = {}          # transposition table persistente por partida

    def _get_depth(self, board: np.ndarray) -> int:
        pieces = int(np.sum(board != 0))
        for threshold, depth in self.DEPTH_SCHEDULE:
            if pieces < threshold:
                return depth
        return 7

    def mount(self, timeout=None) -> None:
        self._me = -1
        self._tracker.reset()
        self._tt = {}                # limpiar TT al inicio de cada partida

    def _find_fork(self, board: np.ndarray, player: int) -> int | None:
        """Devuelve una columna que crea ≥2 amenazas simultáneas, o None."""
        for col in _valid_cols(board):
            nb = _drop(board, col, player)
            if _count_threats(nb, player) >= 2:
                return col
        return None

    def _book_is_safe(self, board: np.ndarray, book_col: int, me: int, depth: int) -> bool:
        """
        Descarta la jugada del libro si el Minimax prefiere otra por > 20 puntos.
        Usa depth-1 para no duplicar el costo computacional.
        """
        valid = _valid_cols(board)
        if book_col not in valid:
            return False
        score_book, _ = _minimax(
            _drop(board, book_col, me), max(depth-1, 1),
            -np.inf, np.inf, False, me, self._tt
        )
        best_score, _ = _minimax(
            board, max(depth-1, 1),
            -np.inf, np.inf, True, me, self._tt
        )
        return (best_score - score_book) <= 20.0

    def act(self, s: np.ndarray) -> int:
        # Detectar color propio
        neg = int(np.sum(s == -1))
        pos = int(np.sum(s == 1))
        self._me = -1 if neg <= pos else 1
        opp = -self._me

        valid = _valid_cols(s)
        if not valid:
            return 0

        # ── Prioridad 0: ganar inmediatamente ──────────────────────────────
        for col in valid:
            if _check_win(_drop(s, col, self._me), self._me):
                return col

        # ── Prioridad 1: bloquear victoria inmediata del oponente ──────────
        for col in valid:
            if _check_win(_drop(s, col, opp), opp):
                return col

        # ── Prioridad 2: crear fork propio ─────────────────────────────────
        fork_col = self._find_fork(s, self._me)
        if fork_col is not None:
            return fork_col

        # ── Prioridad 3: bloquear fork del oponente ────────────────────────
        opp_fork = self._find_fork(s, opp)
        if opp_fork is not None:
            return opp_fork

        # Actualizar secuencia de jugadas
        sequence = tuple(self._tracker.update(s))
        total_pieces = int(np.sum(s != 0))
        depth = self._get_depth(s)

        # ── Fase apertura: tabla validada (≤ 8 piezas) ─────────────────────
        if total_pieces <= 8:
            book_move = lookup_opening(sequence)
            if book_move is not None and book_move in valid:
                if self._book_is_safe(s, book_move, self._me, depth):
                    return book_move
            if total_pieces == 0:
                return 3

        # ── Minimax con TT persistente ─────────────────────────────────────
        _, col = _minimax(s, depth, -np.inf, np.inf, True, self._me, self._tt)

        if col is None or col not in valid:
            col = min(valid, key=lambda c: abs(c - COLS // 2))

        return int(col)
