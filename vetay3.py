import cv2
import mediapipe as mp
import pyautogui
import time
import math
import threading
# Mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5,static_image_mode=False    )

# Mouse smoothing
smooth_x, smooth_y = 0, 0
alpha = 0.20  # giảm để di chuyển chuột mượt hơn

screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)

# Trạng thái chương trình
is_paused = False
last_fist_time = 0
fist_cooldown = 1.0  # cooldown để tránh toggle liên tục
was_fist = False  # theo dõi trạng thái nắm tay trước đó
last_gesture_time = 0
cooldown = 0.4
prev_avg_y = None
is_dragging = False #trạng thái giữ
index_thumb_touch_start = 0  #thời điểm bắt đầu chạm
is_touching_index_thumb = False  #đang chạm index + thumb
drag_threshold = 2.5  #3 giây để chuyển sang drag

def dist(a, b):
    """Khoảng cách Euclid giữa 2 landmark mediapipe"""
    return math.hypot(a.x - b.x, a.y - b.y)


def is_hand_closed(hand):
    index_closed = hand.landmark[8].y > hand.landmark[6].y
    middle_closed = hand.landmark[12].y > hand.landmark[10].y
    ring_closed = hand.landmark[16].y > hand.landmark[14].y
    pinky_closed = hand.landmark[20].y > hand.landmark[18].y
    
    # Ngón cái (4) phải chạm vào điểm 11 (khớp giữa của ngón giữa)
    thumb_tip = hand.landmark[4]
    middle_joint = hand.landmark[7]
    thumb_touch_middle = dist(thumb_tip, middle_joint) < 0.015  # ngưỡng khoảng cách
    
    return index_closed and middle_closed and ring_closed and pinky_closed and thumb_touch_middle
def can_move_cursor_y(hand):
    middle_closed = hand.landmark[12].y > hand.landmark[10].y
    ring_closed = hand.landmark[16].y > hand.landmark[14].y
    pinky_closed = hand.landmark[20].y > hand.landmark[18].y
    
    return middle_closed and ring_closed and pinky_closed 
def can_move_cursor_x(hand):
    middle_closedx = hand.landmark[12].x > hand.landmark[10].x
    ring_closedx = hand.landmark[16].x > hand.landmark[14].x
    pinky_closedx = hand.landmark[20].x > hand.landmark[18].x
    return middle_closedx and ring_closedx and pinky_closedx
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Hiển thị trạng thái RUNNING / PAUSED
    cv2.putText(frame,
                "PAUSED" if is_paused else "RUNNING",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255) if is_paused else (0, 255, 0),
                3)

    if results.multi_hand_landmarks:
        for hand in (results.multi_hand_landmarks):
            
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                # Vị trí các landmark chính
                index = hand.landmark[8]
                middle = hand.landmark[12]
                thumb = hand.landmark[4]
                ring = hand.landmark[16]
                pinky = hand.landmark[20]
                index_mcp = hand.landmark[5]

                middle_closed = hand.landmark[12].y > hand.landmark[10].y
                ring_closed = hand.landmark[16].y > hand.landmark[14].y
                pinky_closed = hand.landmark[20].y > hand.landmark[18].y

                index_up = index.y < hand.landmark[6].y
                middle_up = middle.y < hand.landmark[10].y
                ring_up = ring.y < hand.landmark[14].y
                pinky_up = pinky.y < hand.landmark[18].y
                # PAUSE / RESUME (TOGGLE với nắm tay)
                current_time = time.time()
                is_fist_now = is_hand_closed(hand)
                
                # Phát hiện khi nắm tay (chuyển từ không nắm → nắm)
                if is_fist_now and not was_fist:
                    if current_time - last_fist_time > fist_cooldown:
                        is_paused = not is_paused  # Toggle trạng thái
                        last_fist_time = current_time
                        print( "PAUSED" if is_paused else "RUNNING")
                
                was_fist = is_fist_now
                
                # Nếu đang nắm tay hoặc đang PAUSE (bỏ qua gesture khác)
                if is_fist_now or is_paused:
                    continue

                # DOUBLE CLICK (middle + thumb)
                if dist(middle, thumb) < 0.045:
                    if time.time() - last_gesture_time > cooldown:
                        pyautogui.doubleClick()
                        last_gesture_time = time.time()
                    continue

                # CLICK (index + thumb) 
                dist_index_thumb = dist(index, thumb)
                if dist_index_thumb < 0.05:
                    if not is_touching_index_thumb:
                        is_touching_index_thumb = True
                        index_thumb_touch_start = time.time()
                    hold_duration = time.time() - index_thumb_touch_start
                    if hold_duration >= drag_threshold and not is_dragging:
                        # Giữ >= 3s → Bắt đầu drag
                        pyautogui.mouseDown()
                        is_dragging = True
                    if is_dragging:
                        px, py = int(index.x * w), int(index.y * h)
                        smooth_x = int(alpha * px + (1 - alpha) * smooth_x)
                        smooth_y = int(alpha * py + (1 - alpha) * smooth_y)
                        pyautogui.moveTo(screen_w / w * smooth_x,
                                    screen_h / h * smooth_y)
                    
                    continue
                else:
                    # Tách ngón ra
                    if is_touching_index_thumb:
                        hold_duration = time.time() - index_thumb_touch_start
                        
                        if is_dragging:
                            # Đang drag → Thả chuột
                            pyautogui.mouseUp()
                            is_dragging = False
                            
                        elif hold_duration < drag_threshold:
                            # Giữ < 3s → Click
                            if time.time() - last_gesture_time > cooldown:
                                pyautogui.click()
                                last_gesture_time = time.time()
                                
                        
                        is_touching = False
                # PRESS LEFT (ring + thumb)
                if dist(ring, thumb) < 0.055:
                    if time.time() - last_gesture_time > cooldown:
                        pyautogui.press("left")
                        last_gesture_time = time.time()
                    continue

                # SCROLL (index + middle song song)
                
                if dist(index, middle) < 0.035:
                        avg_y = (index.y + middle.y) / 2

                        if prev_avg_y is not None:
                            delta = avg_y - prev_avg_y
                            if abs(delta) > 0.005:
                                pyautogui.scroll(300 if delta > 0 else -300)
                                if delta > 0:
                                    print("xuống")
                                else:
                                    print("lên")
                        prev_avg_y = avg_y
                        continue
                else:
                        prev_avg_y = None
                    
            
                    
                if dist(pinky, thumb) < 0.065:
                    if time.time() - last_gesture_time > cooldown:
                        pyautogui.rightClick()
                        last_gesture_time = time.time()
                    continue


                prev_avg_y = None

                # MOVE CURSOR (default) 
                if can_move_cursor_x(hand) or can_move_cursor_y(hand):
                    px, py = int(index.x * w), int(index.y * h)

                    smooth_x = int(alpha * px + (1 - alpha) * smooth_x)
                    smooth_y = int(alpha * py + (1 - alpha) * smooth_y)

                    pyautogui.moveTo(screen_w / w * smooth_x,
                                    screen_h / h * smooth_y)

    else:
        # Không phát hiện tay, reset trạng thái
        was_fist = False

    cv2.imshow("Hand Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()