import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState

class GreedyHeuristicAgent(Policy):
    
    def __init__(self):
        self.player = None

    def mount(self, action_timeout: float | None = None) -> None:

        self.player = None

    def act(self, state_or_board) -> int:

        if isinstance(state_or_board, np.ndarray):
            board = state_or_board.copy()
            current_player = -1 if np.sum(board == -1) == np.sum(board == 1) else 1
            state = ConnectState(board=board, player=current_player)
        else:
            state = state_or_board

        self.player = state.player
        valid_cols = state.get_free_cols()

        if state.is_final(): 
            return int(valid_cols[0]) if valid_cols else 0
        if len(valid_cols) == 1: 
            return int(valid_cols[0])

        best_score = -float('inf')
        best_col = valid_cols[0]

        for col in valid_cols:
            next_state = state.transition(int(col))
            score = self.evaluate_board(next_state.board, self.player)
            if score > best_score:
                best_score = score
                best_col = col

        return int(best_col)

    def evaluate_board(self, board, player) -> float:
        score = 0

        center_array = [int(i) for i in list(board[:, 3])]
        center_count = center_array.count(player)
        score += center_count * 3

        for r in range(6):
            row_array = [int(i) for i in list(board[r, :])]
            for c in range(4):
                window = row_array[c:c+4]
                score += self.score_window(window, player)

        for c in range(7):
            col_array = [int(i) for i in list(board[:, c])]
            for r in range(3):
                window = col_array[r:r+4]
                score += self.score_window(window, player)

        for r in range(3):
            for c in range(4):
                window = [int(board[r+i, c+i]) for i in range(4)]
                score += self.score_window(window, player)

        for r in range(3, 6):
            for c in range(4):
                window = [int(board[r-i, c+i]) for i in range(4)]
                score += self.score_window(window, player)

        return score

    def score_window(self, window, player) -> float:
        score = 0
        opp_player = -player
        
        if window.count(player) == 4: 
            score += 100
        elif window.count(player) == 3 and window.count(0) == 1: 
            score += 5
        elif window.count(player) == 2 and window.count(0) == 2: 
            score += 2

        if window.count(opp_player) == 3 and window.count(0) == 1: 
            score -= 4
            
        return score