def get_gomoku_board_score(canonicalBoard, action):
    """
    Calculates a heuristic score for a given action on the Gomoku board.

    Args:
        canonicalBoard (np.array): The current board state in canonical form.
        action (int): The linear index of the action to evaluate.

    Returns:
        score (float): The heuristic score of the action.
    """
    score = 0
    board_size = len(canonicalBoard)  # Assume square board
    x, y = divmod(action, board_size)  # Convert linear index to 2D coordinates
    player = 1  # Assume the current player is 1
    opponent = 2  # Opponent is player 2
    
    # Define all 8 possible directions on the board
    directions = [(1, 0), (0, 1), (1, 1), (1, -1),
                  (-1, 0), (0, -1), (-1, -1), (-1, 1)]

    # Iterate over all directions to evaluate action's impact
    for dx, dy in directions:
        player_count = 0  # Count consecutive player's stones
        opponent_count = 0  # Count consecutive opponent's stones

        # Calculate the next position in the current direction
        nx, ny = x + dx, y + dy
        cur_player = canonicalBoard[nx, ny] if 0 <= nx < board_size and 0 <= ny < board_size else 0  # Check boundaries

        if cur_player != 0:  # If the direction starts with a stone
            # Traverse in the current direction as long as the stones are the same
            while 0 <= nx < board_size and 0 <= ny < board_size:
                if canonicalBoard[nx, ny] == cur_player:
                    if cur_player == player:
                        player_count += 1
                    else:
                        opponent_count += 1
                else:
                    break  # Stop when the stone type changes
                nx += dx
                ny += dy

        # Assign higher weight to longer chains using a power
        score += player_count ** 2.5  # Player's chain contribution
        score += opponent_count ** 2.5  # Opponent's chain contribution

    # Bonus for proximity to the center of the board (encourages central play)
    centre_x, centre_y = board_size // 2, board_size // 2
    score += (board_size - (abs(x - centre_x) + abs(y - centre_y)))

    return score
