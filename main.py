import random
import cv2
import cvzone
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector


class SnakeGameClass:
    def __init__(self, pathFood):
        self.points = []
        self.lengths = []
        self.currentLength = 0
        self.allowedLength = 150
        self.previousHead = 0, 0

        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        self.hfood, self.wfood, trash = self.imgFood.shape
        self.FoodPoint = 0, 0
        self.randomFoodLocation()

        self.score = 0
        self.gameOver = False
        self.PB_score = 0

    def randomFoodLocation(self):
        self.FoodPoint = random.randint(100, 1100), random.randint(100, 600)

    def update(self, img, currentHead):
        if self.gameOver:
            cvzone.putTextRect(img, "Game Over", [300, 400], scale=7, thickness=5, offset=20)
            cvzone.putTextRect(img, f'Your Score: {self.score}', [300, 550], scale=7, thickness=5, offset=20)

            # high score system
            try:
                with open("high score.txt", "r") as file_high_score:
                    high_score = int(file_high_score.read())
                file_high_score.close()
            except IOError:
                with open("high score.txt", "w") as file_high_score:
                    file_high_score.write("0")
                    high_score = 0
                file_high_score.close()

            with open("high score.txt", "w") as file_high_score:
                if self.score > high_score:
                    file_high_score.write(str(self.score))
                else:
                    file_high_score.write(str(high_score))
            file_high_score.close()
            cvzone.putTextRect(img, f'High Score: {high_score}', [700, 150], scale=3, thickness=3, offset=10)

            # Personal best
            if self.score > self.PB_score:
                self.PB_score = self.score
            else:
                pass
            cvzone.putTextRect(img, f'PB: {self.PB_score}', [700, 250], scale=3, thickness=3, offset=10)
        else:
            # snake creation
            px, py = self.previousHead
            cx, cy = currentHead

            self.points.append([cx, cy])
            distance = math.hypot(cx - px, cy - py)
            self.lengths.append(distance)
            self.currentLength += distance
            self.previousHead = cx, cy

            # length reduction
            if self.currentLength > self.allowedLength:
                for i, length in enumerate(self.lengths):
                    self.currentLength -= length
                    self.lengths.pop(i)
                    self.points.pop(i)
                    if self.currentLength <= self.allowedLength:
                        break

            # check if food is eaten
            rx, ry = self.FoodPoint
            if rx + self.wfood // 2 >= cx >= rx - self.wfood // 2 and ry + self.hfood // 2 >= cy >= ry - self.hfood // 2:
                self.allowedLength += 30
                self.randomFoodLocation()
                self.score += 1
            else:
                pass

            # draw snake
            for i, points in enumerate(self.points):
                if i != 0:
                    cv2.line(img, self.points[i - 1], self.points[i], (0, 0, 255), 15)

            if len(self.points) > 0:
                cv2.circle(img, self.points[-1], 15, (0, 255, 0), cv2.FILLED)
            else:
                pass

            # draw food
            img = cvzone.overlayPNG(img, self.imgFood, (rx - self.wfood // 2, ry - self.hfood // 2))
            cvzone.putTextRect(img, f'Score: {self.score}', [50, 80], scale=3, thickness=3, offset=10)

            # collision check
            pts = np.array(self.points[:-2], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], False, (0, 200, 0), 3)
            minDist = cv2.pointPolygonTest(pts, (cx, cy), True)

            if -1 <= minDist <= 1:
                self.gameOver = True
                self.points = []
                self.lengths = []
                self.currentLength = 0
                self.allowedLength = 150
                self.previousHead = 0, 0

                self.randomFoodLocation()
        return img



cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8, maxHands=1)


snakefood = "Donut.png"
game = SnakeGameClass(snakefood)
while True:
    # basic setup
    _, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lmlist = hands[0]["lmList"]
        indexPoint = lmlist[8][0:2]
        img = game.update(img, indexPoint)
        cv2.imshow("Image", img)
    else:
        pass
    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k == ord("r"):
        game.score = 0
        game.gameOver = False

cap.release
cv2.destroyAllWindows()
