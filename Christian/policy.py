import numpy as np
from connect4.policy import Policy


ROWS = 6
COLS = 7

OPENING_BOOK: dict[tuple[int, ...], int] = {

    (): 3,

    (3,): 3,   # respuesta al espejo: tambien columna 3 es valida para P2

    # Movimiento 3: primer jugador, despues de que P2 responde

    (3, 3): 2,      
    (3, 2): 3,      
    (3, 4): 3,      
    (3, 1): 2,      
    (3, 5): 4,      
    (3, 0): 2,      
    (3, 6): 4,      

    # Movimiento 4: segundo jugador 
    # Linea principal D1 d2 C1 (col3, col3, col2):
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

    # Movimiento 5: primer jugador 
    # Linea principal D1 d2 C1 e2 (cols 3,3,2,4):
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
    """Espeja una columna respecto al centro del tablero."""
    return COLS - 1 - col


def _mirror_sequence(seq: tuple[int, ...]) -> tuple[int, ...]:
    """Espeja toda una secuencia de movimientos."""
    return tuple(_mirror_col(c) for c in seq)


def lookup_opening(move_sequence: tuple[int, ...]) -> int | None:
   
    if move_sequence in OPENING_BOOK:
        return OPENING_BOOK[move_sequence]

    mirrored = _mirror_sequence(move_sequence)
    if mirrored in OPENING_BOOK:
        return _mirror_col(OPENING_BOOK[mirrored])

    return None


def _extract_move_sequence(board: np.ndarray) -> tuple[int, ...] | None:

    total_pieces = int(np.sum(board != 0))
    if total_pieces > 8:
        return None

    moves = []
    piece_positions = []
    for c in range(COLS):
        col = board[:, c]
        for r in reversed(range(ROWS)):  
            if col[r] != 0:
               
                height_at_c = int(np.sum(col != 0))
                
                pos_from_bottom = ROWS - 1 - r
                piece_positions.append((pos_from_bottom, c, col[r]))
    return None

#  MINIMAX CON ALPHA-BETA PRUNING

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
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r, c + i] == player for i in range(4)):
                return True
    # Vertical
    for r in range(ROWS - 3):
        for c in range(COLS):
            if all(board[r + i, c] == player for i in range(4)):
                return True
    # Diagonal ↘
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r + i, c + i] == player for i in range(4)):
                return True
    # Diagonal ↙
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            if all(board[r + i, c - i] == player for i in range(4)):
                return True
    return False


def _score_window(window: list, me: int) -> float:
    opp = -me
    me_c   = window.count(me)
    opp_c  = window.count(opp)
    empty  = window.count(0)

    if me_c == 4:              return  1000.0
    if opp_c == 4:             return -1000.0
    if me_c == 3 and empty==1: return    10.0
    if me_c == 2 and empty==2: return     2.0
    if opp_c==3 and empty==1:  return   -15.0   # bloqueo agresivo
    if opp_c==2 and empty==2:  return    -3.0
    return 0.0


def _score_board(board: np.ndarray, me: int) -> float:
    score = 0.0
    opp = -me

    # Bonus columna central
    score += int(np.sum(board[:, COLS // 2] == me))  * 6
    score -= int(np.sum(board[:, COLS // 2] == opp)) * 6

    # Ventanas horizontales
    for r in range(ROWS):
        for c in range(COLS - 3):
            score += _score_window(list(board[r, c:c+4]), me)

    # Ventanas verticales
    for c in range(COLS):
        for r in range(ROWS - 3):
            score += _score_window(list(board[r:r+4, c]), me)

    # Diagonal 
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            score += _score_window([board[r+i, c+i] for i in range(4)], me)

    # Diagonal 
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            score += _score_window([board[r+i, c-i] for i in range(4)], me)

    return score


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

#  SEGUIMIENTO DE SECUENCIA EN TIEMPO REAL

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
            return list(self._sequence)

        diff = board.astype(int) - self._prev_board.astype(int)
        changed = list(zip(*np.where(diff != 0)))

        if len(changed) == 1:
            _, col = changed[0]
            self._sequence.append(int(col))

        self._prev_board = board.copy()
        return list(self._sequence)



#  POLITICA HIBRIDA

class MyAgent(Policy):

    # Profundidad de busqueda segun piezas en el tablero
    DEPTH_SCHEDULE = [
        (10, 5),   # < 10 piezas | profundidad 5
        (20, 6),   # < 20 piezas | profundidad 6
        (30, 7),   # < 30 piezas | profundidad 7
        (42, 9),   # endgame     | profundidad 9
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

  
    def act(self, s: np.ndarray) -> int:
        # Detectar color
        neg = int(np.sum(s == -1))
        pos = int(np.sum(s == 1))
        self._me = -1 if neg <= pos else 1
        opp = -self._me

        valid = _valid_cols(s)
        if not valid:
            return 0

        # ── Prioridad 0: ganar inmediatamente ────────────────────────────────
        for col in valid:
            if _check_win(_drop(s, col, self._me), self._me):
                return col

        # ── Prioridad 1: bloquear victoria inmediata del oponente ─────────────
        for col in valid:
            if _check_win(_drop(s, col, opp), opp):
                return col

        # Actualizar secuencia de jugadas
        sequence = tuple(self._tracker.update(s))
        total_pieces = int(np.sum(s != 0))

        # ── Fase 1: Tabla de aperturas (primeras ~8 piezas) ──────────────────
        if total_pieces <= 8:
            book_move = lookup_opening(sequence)
            if book_move is not None and book_move in valid:
                return book_move

            # Si no hay entrada exacta pero es el primer movimiento, ir al centro
            if total_pieces == 0:
                return 3
            if total_pieces == 1 and 3 in valid:
                # Segundo jugador: ir a col adyacente al centro
                return 3 if 3 in valid else (2 if 2 in valid else 4)

        # ── Fase 2: Minimax con Alpha-Beta ───────────────────────────────────
        depth = self._get_depth(s)
        _, col = _minimax(s, depth, -np.inf, np.inf, True, self._me)

        if col is None or col not in valid:
            col = min(valid, key=lambda c: abs(c - COLS // 2))

        return int(col)