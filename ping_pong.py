import cv2
import numpy as np
import mediapipe as mp
import time
import random

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
STREAM_WIDTH = 1000
STREAM_HEIGHT = 800
CENTER_X = STREAM_WIDTH // 2
CENTER_Y = STREAM_HEIGHT // 2

PADDLE_WIDTH = 20
PADDLE_HEIGHT = 100
PLAYER1_X = 50
PLAYER2_X = STREAM_WIDTH - 70
COMPUTER_X = STREAM_WIDTH - 70
COMPUTER_SPEED = 8

BALL_SIZE = 20
INITIAL_BALL_SPEED_X = 9
INITIAL_BALL_SPEED_Y = 9
BALL_SPEED_INCREMENT = 1.05
BALL_RESET_DELAY = 1.0

WINNING_SCORE = 3

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    exit()

cv2.namedWindow("Pong", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pong", STREAM_WIDTH, STREAM_HEIGHT)
cv2.moveWindow("Pong", (SCREEN_WIDTH - STREAM_WIDTH) // 2, (SCREEN_HEIGHT - STREAM_HEIGHT) // 2)

def create_gradient_background(width, height):
    background = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        color = int(255 * (i / height))
        background[i, :] = (color, 100, 255 - color)
    return background

background_img = create_gradient_background(STREAM_WIDTH, STREAM_HEIGHT)

player_score = 0
computer_score = 0
player1_score = 0
player2_score = 0
player1_paddle_y = CENTER_Y
player2_paddle_y = CENTER_Y
computer_paddle_y = CENTER_Y
ball_x, ball_y = CENTER_X, CENTER_Y
ball_speed_x, ball_speed_y = INITIAL_BALL_SPEED_X, INITIAL_BALL_SPEED_Y
last_hit_time = time.time()
last_score_time = time.time()
state = "start"
game_mode = None
score_side = ""

def reset_ball():
    global ball_x, ball_y, ball_speed_x, ball_speed_y
    ball_x, ball_y = CENTER_X, CENTER_Y
    direction = random.choice([-1, 1])
    ball_speed_x = INITIAL_BALL_SPEED_X * direction
    ball_speed_y = random.choice([-INITIAL_BALL_SPEED_Y, INITIAL_BALL_SPEED_Y])

def reset_game():
    global player_score, computer_score, player1_score, player2_score
    player_score = 0
    computer_score = 0
    player1_score = 0
    player2_score = 0
    reset_ball()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    display_frame = background_img.copy()

    if state == "start":
        overlay = display_frame.copy()
        start_text = "Press 'C' to Play vs Computer or 'F' to Play vs Friend"
        (tw, th), _ = cv2.getTextSize(start_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(overlay, start_text, ((STREAM_WIDTH - tw) // 2, STREAM_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        made_by = "Created by Sean Pesis"
        (mw, mh), _ = cv2.getTextSize(made_by, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(overlay, made_by, ((STREAM_WIDTH - mw) // 2, STREAM_HEIGHT // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        cv2.imshow("Pong", display_frame)
        k = cv2.waitKey(1) & 0xFF
        if k in [ord('c'), ord('C')]:
            game_mode = 'computer'
            reset_game()
            state = "play"
        elif k in [ord('f'), ord('F')]:
            game_mode = 'friend'
            reset_game()
            state = "play"
        elif k in [ord('q'), ord('Q')]:
            break
        continue

    elif state == "play":
        if game_mode == 'computer':
            if res.multi_hand_landmarks:
                hlms = res.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(display_frame, hlms, mp_hands.HAND_CONNECTIONS)
                x_index = int(hlms.landmark[8].x * STREAM_WIDTH)
                y_index = int(hlms.landmark[8].y * STREAM_HEIGHT)
                player1_paddle_y = y_index
        elif game_mode == 'friend':
            if res.multi_hand_landmarks and len(res.multi_hand_landmarks) == 2:
                hlms1, hlms2 = res.multi_hand_landmarks
                mp_draw.draw_landmarks(display_frame, hlms1, mp_hands.HAND_CONNECTIONS)
                mp_draw.draw_landmarks(display_frame, hlms2, mp_hands.HAND_CONNECTIONS)
                x1 = int(hlms1.landmark[8].x * STREAM_WIDTH)
                y1 = int(hlms1.landmark[8].y * STREAM_HEIGHT)
                x2 = int(hlms2.landmark[8].x * STREAM_WIDTH)
                y2 = int(hlms2.landmark[8].y * STREAM_HEIGHT)
                if x1 < x2:
                    player1_paddle_y = y1
                    player2_paddle_y = y2
                else:
                    player1_paddle_y = y2
                    player2_paddle_y = y1

        ball_x += ball_speed_x
        ball_y += ball_speed_y

        if ball_y <= BALL_SIZE // 2 or ball_y >= STREAM_HEIGHT - BALL_SIZE // 2:
            ball_speed_y = -ball_speed_y

        if ball_x <= PLAYER1_X + PADDLE_WIDTH + BALL_SIZE // 2:
            if player1_paddle_y - PADDLE_HEIGHT // 2 <= ball_y <= player1_paddle_y + PADDLE_HEIGHT // 2:
                ball_x = PLAYER1_X + PADDLE_WIDTH + BALL_SIZE // 2
                ball_speed_x = -ball_speed_x * BALL_SPEED_INCREMENT
                ball_speed_y *= BALL_SPEED_INCREMENT
                last_hit_time = time.time()
            else:
                if game_mode == 'computer':
                    computer_score +=1
                    score_side = "Computer"
                    if computer_score >= WINNING_SCORE:
                        state = "result"
                    else:
                        reset_ball()
                        last_score_time = time.time()
                elif game_mode == 'friend':
                    player2_score +=1
                    score_side = "Player 2"
                    if player2_score >= WINNING_SCORE:
                        state = "result"
                    else:
                        reset_ball()
                        last_score_time = time.time()

        if game_mode == 'computer':
            paddle_y = computer_paddle_y
            paddle_x = COMPUTER_X
            if ball_x >= paddle_x - BALL_SIZE // 2:
                if paddle_y - PADDLE_HEIGHT // 2 <= ball_y <= paddle_y + PADDLE_HEIGHT // 2:
                    ball_x = paddle_x - BALL_SIZE // 2
                    ball_speed_x = -ball_speed_x * BALL_SPEED_INCREMENT
                    ball_speed_y *= BALL_SPEED_INCREMENT
                    last_hit_time = time.time()
                else:
                    player_score +=1
                    score_side = "Player"
                    if player_score >= WINNING_SCORE:
                        state = "result"
                    else:
                        reset_ball()
                        last_score_time = time.time()
        elif game_mode == 'friend':
            if res.multi_hand_landmarks and len(res.multi_hand_landmarks) == 2:
                hlms1, hlms2 = res.multi_hand_landmarks
                x1 = int(hlms1.landmark[8].x * STREAM_WIDTH)
                x2 = int(hlms2.landmark[8].x * STREAM_WIDTH)
                if x1 < x2:
                    paddle_y = y2
                else:
                    paddle_y = y1
                if ball_x >= PLAYER2_X - BALL_SIZE // 2:
                    if paddle_y - PADDLE_HEIGHT // 2 <= ball_y <= paddle_y + PADDLE_HEIGHT // 2:
                        ball_x = PLAYER2_X - BALL_SIZE // 2
                        ball_speed_x = -ball_speed_x * BALL_SPEED_INCREMENT
                        ball_speed_y *= BALL_SPEED_INCREMENT
                        last_hit_time = time.time()
                    else:
                        player1_score +=1
                        score_side = "Player 1"
                        if player1_score >= WINNING_SCORE:
                            state = "result"
                        else:
                            reset_ball()
                            last_score_time = time.time()

        if game_mode == 'computer':
            if ball_x < -BALL_SIZE:
                computer_score +=1
                score_side = "Computer"
                if computer_score >= WINNING_SCORE:
                    state = "result"
                else:
                    reset_ball()
                    last_score_time = time.time()
        elif game_mode == 'friend':
            if ball_x > STREAM_WIDTH + BALL_SIZE:
                player2_score +=1
                score_side = "Player 2"
                if player2_score >= WINNING_SCORE:
                    state = "result"
                else:
                    reset_ball()
                    last_score_time = time.time()

        if game_mode == 'computer':
            if computer_paddle_y < ball_y:
                computer_paddle_y += COMPUTER_SPEED
            elif computer_paddle_y > ball_y:
                computer_paddle_y -= COMPUTER_SPEED
        elif game_mode == 'friend':
            pass

        player1_paddle_y = max(PADDLE_HEIGHT // 2, min(STREAM_HEIGHT - PADDLE_HEIGHT // 2, player1_paddle_y))
        if game_mode == 'computer':
            computer_paddle_y = max(PADDLE_HEIGHT // 2, min(STREAM_HEIGHT - PADDLE_HEIGHT // 2, computer_paddle_y))
        elif game_mode == 'friend':
            player2_paddle_y = max(PADDLE_HEIGHT // 2, min(STREAM_HEIGHT - PADDLE_HEIGHT // 2, player2_paddle_y))

        cv2.line(display_frame, (CENTER_X, 0), (CENTER_X, STREAM_HEIGHT), (255, 255, 255), 2)

        cv2.rectangle(display_frame,
                      (PLAYER1_X, int(player1_paddle_y - PADDLE_HEIGHT // 2)),
                      (PLAYER1_X + PADDLE_WIDTH, int(player1_paddle_y + PADDLE_HEIGHT // 2)),
                      (0, 255, 0), -1)
        if game_mode == 'computer':
            cv2.rectangle(display_frame,
                          (COMPUTER_X, int(computer_paddle_y - PADDLE_HEIGHT // 2)),
                          (COMPUTER_X + PADDLE_WIDTH, int(computer_paddle_y + PADDLE_HEIGHT // 2)),
                          (0, 0, 255), -1)
        elif game_mode == 'friend':
            cv2.rectangle(display_frame,
                          (PLAYER2_X, int(player2_paddle_y - PADDLE_HEIGHT // 2)),
                          (PLAYER2_X + PADDLE_WIDTH, int(player2_paddle_y + PADDLE_HEIGHT // 2)),
                          (255, 0, 0), -1)

        cv2.circle(display_frame, (int(ball_x), int(ball_y)), BALL_SIZE // 2, (255, 255, 255), -1)

        if game_mode == 'computer':
            player_score_text = f"Player: {player_score}"
            computer_score_text = f"Computer: {computer_score}"
            cv2.putText(display_frame, player_score_text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, computer_score_text, (STREAM_WIDTH - 300, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif game_mode == 'friend':
            player1_score_text = f"Player 1: {player1_score}"
            player2_score_text = f"Player 2: {player2_score}"
            cv2.putText(display_frame, player1_score_text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, player2_score_text, (STREAM_WIDTH - 300, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if score_side:
            side_text = f"Point to {score_side}"
            (stw, sth), _ = cv2.getTextSize(side_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.putText(display_frame, side_text, ((STREAM_WIDTH - stw) // 2, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            if time.time() - last_score_time > 2:
                score_side = ""

        if game_mode == 'computer':
            if player_score >= WINNING_SCORE or computer_score >= WINNING_SCORE:
                state = "result"
        elif game_mode == 'friend':
            if player1_score >= WINNING_SCORE or player2_score >= WINNING_SCORE:
                state = "result"

        cv2.imshow("Pong", display_frame)
        k = cv2.waitKey(1) & 0xFF
        if state == "play":
            if k in [ord('q'), ord('Q')]:
                break
            elif k in [ord('r'), ord('R')]:
                reset_game()
                state = "play"
            elif k in [ord('m'), ord('M')]:
                reset_game()
                state = "start"

    elif state == "result":
        if game_mode == 'computer':
            if player_score > computer_score:
                msg = "You Win!"
                color = (0, 255, 0)
            else:
                msg = "Computer Wins!"
                color = (0, 0, 255)
        elif game_mode == 'friend':
            if player1_score > player2_score:
                msg = "Player 1 Wins!"
                color = (0, 255, 0)
            else:
                msg = "Player 2 Wins!"
                color = (255, 0, 0)

        result_frame = background_img.copy()
        overlay = result_frame.copy()
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
        cv2.putText(overlay, msg, ((STREAM_WIDTH - tw) // 2, STREAM_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
        info = "Press 'Q' to Quit, 'R' to Restart, or 'M' for Main Menu"
        (iw, ih), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.putText(overlay, info, ((STREAM_WIDTH - iw) // 2, (STREAM_HEIGHT // 2) + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.addWeighted(overlay, 0.7, result_frame, 0.3, 0, result_frame)
        cv2.imshow("Pong", result_frame)
        k = cv2.waitKey(1) & 0xFF
        if k in [ord('q'), ord('Q')]:
            break
        elif k in [ord('r'), ord('R')]:
            reset_game()
            state = "play"
        elif k in [ord('m'), ord('M')]:
            reset_game()
            state = "start"

cap.release()
cv2.destroyAllWindows()
