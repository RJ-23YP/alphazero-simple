
def get_gomoku_board_score(canonicalBoard, action):
    score = 0
    board_size = len(canonicalBoard)
    x, y = divmod(action, board_size)
    player = 1
    opponent = 2
    
    directions = [(1, 0), (0, 1), (1, 1), (1, -1),
                 (-1, 0), (0, -1), (-1, -1), (-1, 1)]
    for dx, dy in directions:
        player_count = 0
        opponent_count = 0

        nx, ny = x + dx, y + dy
        cur_player = canonicalBoard[nx, ny] if 0 <= nx < board_size and 0 <= ny < board_size else 0
        
        if cur_player != 0:
            while 0 <= nx < board_size and 0 <= ny < board_size:
                if canonicalBoard[nx, ny] == cur_player:
                    if cur_player == player:
                        player_count += 1
                    else:
                        opponent_count += 1
                else:
                    break
                nx += dx
                ny += dy
        score += player_count ** 2.5
        score += opponent_count ** 2.5
    centre_x, centre_y = board_size // 2, board_size // 2
    score += (board_size - (abs(x - centre_x) + abs(y - centre_y)))
    return score