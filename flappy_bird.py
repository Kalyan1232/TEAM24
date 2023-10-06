import cv2
import mediapipe 
from random import randint
from numpy import where, ndarray
import imutils
def GenWall(h: int) -> int:
    return randint(101, h-40)

def DrawWallBuffer(img: ndarray, wallBuf: list) -> None:
    """Draw Wall Buffer To img Buffer"""
    h, w, c = img.shape
    for wall in wallBuf:
        cv2.rectangle(img, (wall[0], h), (wall[0]+40, wall[1]), (0, 255, 0), cv2.FILLED) # buttom wall 
        cv2.rectangle(img, (wall[0], 0), (wall[0]+40, wall[1]-100), (0, 255, 0), cv2.FILLED) # top wall

def MoveWall(wallBuf) -> list:
    """Move Wall Right to Left, 10px per frame"""
    newWallBuf = list()
    for wall in wallBuf:
        newWallBuf.append((wall[0]-10, wall[1]))
    return newWallBuf

def isCollision(x: int, y: int, wallBuf: list) -> bool:
    """Check of Collision bird to wall"""
    for wall in wallBuf:
        if x in range(wall[0], wall[0]+40): # bird in location of wall
            if not y in range(wall[1]-100, wall[1]): # bird not in space between top and buttom wall (Collision)
                return True

    return False

def GameOver(score: int) -> None:
    """Show `Game Over!` message and `player score` then exit the game"""
    print("Game Over!")
    print(f"Your Score: {score}")
    print("bye bye :)")
    exit(0)

def ShowBird(img: ndarray, x: int, y: int, bird: ndarray) -> None:
    """show bird in pose nose of player"""
    y = y - 20
    x = x - 30
    # delete the 0 px for transparent
    con = bird != 0
    out = where(con, bird, img[y:y+50, x:x+50])
    img[y:y+50, x:x+50] = out[:]
    

def ReadBirdFile() -> ndarray:
    """read and resize bird photo"""

    bird = cv2.imread("bird.png")
    # bird = cv2.imread("C:\\Users\\pardh\\Desktop\\AMMA\\PARDHU NADELLA\\OTHER THAN STUDIES\\COLLEGE RELATED\\Computer Vision\\Python\\PROJECTS\\Virtual Trial Room\\sleeve.png")
    # bird = imutils.rotate_bound(bird, -33)
    bird = cv2.resize(bird, [50,50])
    return bird


def main() -> None:
    """Setup"""
    face = mediapipe.solutions.face_detection.FaceDetection()
    cap = cv2.VideoCapture(0)
    bird = ReadBirdFile()
    frame = 0
    score = 0
    wallBuf = list()
    try:
        while cap.isOpened():
            success, img = cap.read()
            if not success: 
                print("[ERROR] Read Capture Failed.")
                break
            img = cv2.flip(img, 1)
            h, w, c = img.shape
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = face.process(imgRGB)
            if not res.detections: GameOver(score) # Face Not Found in image, To prevent fraud 
            nose = res.detections[0].location_data.relative_keypoints[2] # get position of player nose
            nose.x, nose.y = nose.x * w, nose.y * h # Convert float position to Real position in Image

            ShowBird(img, int(nose.x), int(nose.y), bird)

            frame += 1
            if frame >= 30: # Genrate a New Wall at 30 frame
                frame = 0
                wallBuf.append((w-50, GenWall(h)))

            DrawWallBuffer(img, wallBuf)

            wallBuf = MoveWall(wallBuf)
            
            if isCollision(int(nose.x), int(nose.y), wallBuf):
                GameOver(score)
            else:
                score += 1

            cv2.putText(img, str(score), (5,20),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,0,0))

            cv2.imshow("Flappy Bird Game!", img)
            cv2.waitKey(30)


    except KeyboardInterrupt:
        GameOver(score)


if __name__ == "__main__":
    main()