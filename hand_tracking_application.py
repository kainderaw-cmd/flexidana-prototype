import cv2
import mediapipe as mp
import time

class HandDetector:
    """
    A class to detect hands and track landmarks using MediaPipe.
    """
    def __init__(self, mode=False, max_hands=2, detection_con=0.7, track_con=0.7):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        # MediaPipe Hands initialization
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def find_hands(self, img, draw=True):
        """
        Processes an image frame and draws hand landmarks if detected.
        """
        # Convert BGR image to RGB for MediaPipe processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        # Draw landmarks if hands are detected
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.get_default_hand_landmarks_style(),
                        self.mp_draw.get_default_hand_connections_style()
                    )
        return img

    def find_positions(self, img, hand_no=0, draw=True):
        """
        Extracts pixel coordinates of all 21 hand landmarks for a given hand.
        """
        landmark_list = []
        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                my_hand = self.results.multi_hand_landmarks[hand_no]
                h, w, c = img.shape

                for id, lm in enumerate(my_hand.landmark):
                    # Convert normalized positions (0.0 to 1.0) to pixel coordinates
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_list.append([id, cx, cy])
                    
                    if draw:
                        # Draw a small circle on the landmark point
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return landmark_list

def main():
    # Initialize camera (0 is default webcam)
    cap = cv2.VideoCapture(0)
    
    # Adjust camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = HandDetector(max_hands=2, detection_con=0.7, track_con=0.7)

    # Frame rate calculation variables
    p_time = 0
    c_time = 0

    print("Tekan 'q' pada jendela kamera untuk keluar.")

    while True:
        success, img = cap.read()
        if not success:
            print("Gagal mengambil frame dari kamera.")
            break

        # Flip image horizontally for natural mirror display
        img = cv2.flip(img, 1)

        img = detector.find_hands(img)
        lm_list = detector.find_positions(img, draw=False)

        # Optional: Print position of landmark index 8 (index tip)
        if len(lm_list) != 0:
            # lm_list[8] corresponds to Index Finger Tip
            index_finger_x, index_finger_y = lm_list[8][1], lm_list[8][2]
            cv2.circle(img, (index_finger_x, index_finger_y), 12, (0, 255, 0), cv2.FILLED)

        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        # Display FPS text
        cv2.putText(
            img,
            f'FPS: {int(fps)}',
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Display frame window
        cv2.imshow("Hand Tracking - OpenCV & MediaPipe", img)

        # Exit loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam and destroy OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()