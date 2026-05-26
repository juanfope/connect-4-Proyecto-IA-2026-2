import numpy as np
from connect4.policy import Policy


ROWS = 6
COLS = 7

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
    """Cuenta cuántas amenazas de 3-en-raya tiene un jugador (ventanas con 3 piezas + 1 vacío)."""
    count = 0
    opp = -player

    def check_window(window):
        pc = window.count(player)
        ec = window.count(0)
        return pc == 3 and ec == 1

    for r in range(ROWS):
        for c in range(COLS - 3):
            w = list(board[r, c:c+4])
            if check_window(w): count += 1
    for r in range(ROWS - 3):
        for c in range(COLS):
            w = [board[r+i, c] for i in range(4)]
            if check_window(w): count += 1
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            w = [board[r+i, c+i] for i in range(4)]
            if check_window(w): count += 1
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            w = [board[r+i, c-i] for i in range(4)]
            if check_window(w): count += 1
    return count


def _score_window(window: list, me: int) -> float:
    opp = -me
    me_c   = window.count(me)
    opp_c  = window.count(opp)
    empty  = window.count(0)

    if me_c == 4:               return  1000.0
    if opp_c == 4:              return -1000.0
    if me_c == 3 and empty == 1: return   10.0
    if me_c == 2 and empty == 2: return    2.0
    if opp_c == 3 and empty == 1: return -15.0
    if opp_c == 2 and empty == 2: return  -3.0
    return 0.0


def _score_board(board: np.ndarray, me: int) -> float:
    score = 0.0
    opp = -me

    # Bonus columna central
    score += int(np.sum(board[:, COLS // 2] == me))  * 6
    score -= int(np.sum(board[:, COLS // 2] == opp)) * 6

    # Bonus columnas adyacentes al centro
    for c in [2, 4]:
        score += int(np.sum(board[:, c] == me))  * 3
        score -= int(np.sum(board[:, c] == opp)) * 3

    # Ventanas horizontales
    for r in range(ROWS):
        for c in range(COLS - 3):
            score += _score_window(list(board[r, c:c+4]), me)

    # Ventanas verticales
    for c in range(COLS):
        for r in range(ROWS - 3):
            score += _score_window(list(board[r:r+4, c]), me)

    # Diagonal ↘
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            score += _score_window([board[r+i, c+i] for i in range(4)], me)

    # Diagonal ↙
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            score += _score_window([board[r+i, c-i] for i in range(4)], me)

    return score


# ---------------------------------------------------------------------------
# MINIMAX CON ALPHA-BETA Y ORDENAMIENTO MEJORADO
# ---------------------------------------------------------------------------

def _col_order(valid: list[int]) -> list[int]:
    """Ordena columnas por cercanía al centro (mejor para poda alfa-beta)."""
    return sorted(valid, key=lambda c: abs(c - COLS // 2))


def _minimax(
    board: np.ndarray,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    me: int,
) -> tuple[float, int | None]:

    valid = _valid_cols(board)
    opp = -me

    # Victoria/derrota inmediata antes de llegar a depth=0
    for col in valid:
        if _check_win(_drop(board, col, me), me):
            return 1_000_000_000.0 + depth, col   # premiar victorias rápidas
    if not maximizing:
        for col in valid:
            if _check_win(_drop(board, col, opp), opp):
                return -1_000_000_000.0 - depth, col

    terminal = _check_win(board, me) or _check_win(board, opp) or not valid

    if depth == 0 or terminal:
        if _check_win(board, me):  return  1_000_000_000.0, None
        if _check_win(board, opp): return -1_000_000_000.0, None
        if not valid:              return  0.0, None
        return _score_board(board, me), None

    ordered = _col_order(valid)

    if maximizing:
        best, best_col = -np.inf, ordered[0]
        for col in ordered:
            score, _ = _minimax(_drop(board, col, me), depth-1, alpha, beta, False, me)
            if score > best:
                best, best_col = score, col
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best, best_col
    else:
        best, best_col = np.inf, ordered[0]
        for col in ordered:
            score, _ = _minimax(_drop(board, col, opp), depth-1, alpha, beta, True, me)
            if score < best:
                best, best_col = score, col
            beta = min(beta, score)
            if alpha >= beta:
                break
        return best, best_col


# ---------------------------------------------------------------------------
# MOVE TRACKER — reconstruye la secuencia real de jugadas
# ---------------------------------------------------------------------------

class MoveTracker:
    """
    Reconstruye la secuencia de jugadas comparando tableros consecutivos.
    FIX: la versión original tenía _extract_move_sequence que siempre
    devolvía None. Ahora update() detecta correctamente el diff.
    """

    def __init__(self):
        self._prev_board: np.ndarray | None = None
        self._sequence: list[int] = []

    def reset(self):
        self._prev_board = None
        self._sequence = []

    def update(self, board: np.ndarray) -> list[int]:
        if self._prev_board is None:
            self._prev_board = board.copy()
            return list(self._sequence)

        diff = board.astype(int) - self._prev_board.astype(int)
        changed = list(zip(*np.where(diff != 0)))

        if len(changed) == 1:
            _, col = changed[0]
            self._sequence.append(int(col))
        elif len(changed) == 0:
            # Sin cambio: jugada repetida o primera llamada con tablero vacío
            pass
        else:
            # Más de 1 cambio: estado inicial o reset; reconstruir desde cero
            self._sequence = _reconstruct_sequence(board)

        self._prev_board = board.copy()
        return list(self._sequence)


def _reconstruct_sequence(board: np.ndarray) -> list[int]:
    """
    Reconstruye el orden aproximado de jugadas desde un tablero ya poblado,
    asignando turno por la altura de cada columna (filas más bajas = más antiguas).
    Solo fiable cuando el tablero tiene pocas piezas (apertura).
    """
    total = int(np.sum(board != 0))
    if total > 10:
        return []

    # Recolectar (fila_desde_abajo, col, player) para cada pieza
    pieces = []
    for c in range(COLS):
        for r in reversed(range(ROWS)):
            if board[r, c] != 0:
                height = ROWS - 1 - r          # 0 = fondo
                pieces.append((height, c, int(board[r, c])))

    # Ordenar por altura ascendente; piezas en el mismo nivel se alternan por turno
    pieces.sort(key=lambda x: x[0])

    seq = []
    # Turno 0 → jugador -1 (primer jugador), turno 1 → jugador +1, etc.
    for idx, (_, col, _) in enumerate(pieces):
        seq.append(col)
    return seq


# ---------------------------------------------------------------------------
# POLÍTICA HÍBRIDA MEJORADA
# ---------------------------------------------------------------------------

class MyAgent(Policy):
    """
    Agente híbrido V1 mejorado.

    Cambios respecto a la versión original:
    1. MoveTracker corregido: ahora detecta diff correctamente y reconstruye
       la secuencia cuando recibe el primer tablero no vacío.
    2. La tabla de aperturas se valida contra Minimax-1: si el libro propone
       una jugada claramente peor que la mejor del motor, se descarta.
       Esto evita que la tabla dañe situaciones que el Minimax ya resuelve bien.
    3. Detección de doble amenaza (fork): antes de entrar al Minimax se busca
       si existe una jugada que cree 2 amenazas simultáneas, o que bloquee
       una doble amenaza del oponente.
    4. Heurística mejorada: bonus a columnas adyacentes al centro (cols 2 y 4).
    5. El Minimax premia victorias rápidas (+ depth al score terminal),
       evitando que dilate partidas ya ganadas.
    """

    DEPTH_SCHEDULE = [
        (10, 5),
        (20, 6),
        (30, 7),
        (42, 9),
    ]

    def __init__(self):
        self._me: int = -1
        self._tracker = MoveTracker()

    def _get_depth(self, board: np.ndarray) -> int:
        pieces = int(np.sum(board != 0))
        for threshold, depth in self.DEPTH_SCHEDULE:
            if pieces < threshold:
                return depth
        return 9

    def mount(self, timeout=None) -> None:
        self._me = -1
        self._tracker.reset()

    def _find_fork(self, board: np.ndarray, player: int) -> int | None:
        """
        Devuelve una columna que crea ≥2 amenazas simultáneas para `player`,
        o None si no existe.
        """
        valid = _valid_cols(board)
        for col in valid:
            nb = _drop(board, col, player)
            if _count_threats(nb, player) >= 2:
                return col
        return None

    def _book_is_safe(
        self,
        board: np.ndarray,
        book_col: int,
        me: int,
        depth: int,
    ) -> bool:
        """
        Comprueba que la jugada del libro no sea claramente peor que la mejor
        jugada del Minimax. Se usa depth-1 para no duplicar el costo.
        Si la diferencia es > 20 puntos en favor del motor, se descarta el libro.
        """
        valid = _valid_cols(board)
        if book_col not in valid:
            return False

        score_book, _ = _minimax(
            _drop(board, book_col, me), depth - 1, -np.inf, np.inf, False, me
        )
        best_score, _ = _minimax(board, depth - 1, -np.inf, np.inf, True, me)

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

        # ── Prioridad 2: crear fork (doble amenaza propia) ─────────────────
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

        # ── Fase 1: Tabla de aperturas validada ────────────────────────────
        if total_pieces <= 8:
            book_move = lookup_opening(sequence)
            if book_move is not None and book_move in valid:
                if self._book_is_safe(s, book_move, self._me, depth):
                    return book_move

            if total_pieces == 0:
                return 3

        # ── Fase 2: Minimax con Alpha-Beta ─────────────────────────────────
        _, col = _minimax(s, depth, -np.inf, np.inf, True, self._me)

        if col is None or col not in valid:
            col = min(valid, key=lambda c: abs(c - COLS // 2))

        return int(col)
