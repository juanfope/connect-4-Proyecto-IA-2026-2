import numpy as np
from connect4.policy import Policy

ROWS = 6
COLS = 7
WINDOW = 4


def score_window(window: list[int], player: int) -> float:
    """Score a window of 4 cells for the given player."""
    opponent = -player
    player_count = window.count(player)
    empty_count = window.count(0)
    opp_count = window.count(opponent)

    if player_count == 4:
        return 100.0
    if opp_count == 4:
        return -100.0
    if player_count == 3 and empty_count == 1:
        return 5.0
    if opp_count == 3 and empty_count == 1:
        return -4.0
    if player_count == 2 and empty_count == 2:
        return 2.0
    return 0.0


def heuristic(board: np.ndarray, player: int) -> float:
    """Board heuristic evaluation for the given player."""
    score = 0.0

    # Center column preference
    center_col = board[:, COLS // 2].tolist()
    score += center_col.count(player) * 3.0

    # Horizontal windows
    for r in range(ROWS):
        for c in range(COLS - 3):
            w = board[r, c:c + 4].tolist()
            score += score_window(w, player)

    # Vertical windows
    for c in range(COLS):
        for r in range(ROWS - 3):
            w = board[r:r + 4, c].tolist()
            score += score_window(w, player)

    # Diagonal (down-right)
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            w = [board[r + i, c + i] for i in range(4)]
            score += score_window(w, player)

    # Diagonal (down-left)
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            w = [board[r + i, c - i] for i in range(4)]
            score += score_window(w, player)

    return score


def get_valid_cols(board: np.ndarray) -> list[int]:
    return [c for c in range(COLS) if board[0, c] == 0]


def drop_piece(board: np.ndarray, col: int, player: int) -> np.ndarray:
    new_board = board.copy()
    for r in reversed(range(ROWS)):
        if new_board[r, col] == 0:
            new_board[r, col] = player
            break
    return new_board


def check_win(board: np.ndarray, player: int) -> bool:
    """Check if the given player has won."""
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r, c + i] == player for i in range(4)):
                return True
    # Vertical
    for c in range(COLS):
        for r in range(ROWS - 3):
            if all(board[r + i, c] == player for i in range(4)):
                return True
    # Diagonal down-right
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r + i, c + i] == player for i in range(4)):
                return True
    # Diagonal down-left
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            if all(board[r + i, c - i] == player for i in range(4)):
                return True
    return False


def is_terminal(board: np.ndarray) -> bool:
    return (
        check_win(board, -1)
        or check_win(board, 1)
        or len(get_valid_cols(board)) == 0
    )


def minimax(
    board: np.ndarray,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    player: int,
) -> float:
    """Minimax with alpha-beta pruning."""
    if depth == 0 or is_terminal(board):
        if check_win(board, player):
            return 1e9
        if check_win(board, -player):
            return -1e9
        return heuristic(board, player)

    valid_cols = get_valid_cols(board)
    # Search center columns first for better pruning
    valid_cols.sort(key=lambda c: -abs(c - COLS // 2) * -1)

    if maximizing:
        value = -np.inf
        for col in valid_cols:
            child = drop_piece(board, col, player)
            value = max(value, minimax(child, depth - 1, alpha, beta, False, player))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        opponent = -player
        value = np.inf
        for col in valid_cols:
            child = drop_piece(board, col, opponent)
            value = min(value, minimax(child, depth - 1, alpha, beta, True, player))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


class Minimaxer(Policy):
    """
    Connect-4 agent using Minimax search with Alpha-Beta pruning.
    Depth is set to 5 by default, which balances quality vs. speed.
    """

    def __init__(self, depth: int = 5):
        self.depth = depth
        self.player = None  # Determined at first call

    def mount(self, *args) -> None:
        self.player = None  # Reset for each new game

    def act(self, s: np.ndarray) -> int:
        board = s.copy()

        # Determine which player we are on first call
        if self.player is None:
            red_count = np.sum(board == -1)
            yellow_count = np.sum(board == 1)
            # The player whose turn it is has fewer pieces on the board
            # Red (-1) goes first, so if counts are equal it's Red's turn
            self.player = -1 if red_count == yellow_count else 1

        valid_cols = get_valid_cols(board)

        # Immediate win check
        for col in valid_cols:
            test = drop_piece(board, col, self.player)
            if check_win(test, self.player):
                return col

        # Immediate block check
        opponent = -self.player
        for col in valid_cols:
            test = drop_piece(board, col, opponent)
            if check_win(test, opponent):
                return col

        # Run minimax
        best_col = valid_cols[len(valid_cols) // 2]
        best_score = -np.inf

        for col in valid_cols:
            child = drop_piece(board, col, self.player)
            score = minimax(child, self.depth - 1, -np.inf, np.inf, False, self.player)
            if score > best_score:
                best_score = score
                best_col = col

        return best_col