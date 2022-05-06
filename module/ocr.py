from cv2 import cv2
import numpy as np

from .image import Image

class OCR:
    @staticmethod
    def positions(target, threshold=None, baseImage=None, returnArray=False, debug=False):
        img = baseImage

        w = target.shape[1]
        h = target.shape[0]

        result = cv2.matchTemplate(img, target, cv2.TM_CCOEFF_NORMED)

        yloc, xloc = np.where(result >= threshold)

        rectangles = []
        for (x, y) in zip(xloc, yloc):
            rectangles.append([int(x), int(y), int(w), int(h)])
            rectangles.append([int(x), int(y), int(w), int(h)])

        rectangles, _ = cv2.groupRectangles(rectangles, 1, 0.2)

        if debug == True:
            img2 = img.copy()
            for r in rectangles:
                cv2.rectangle(img2, (r[0], r[1]),
                              (r[0]+w, r[1]+h), (0, 0, 255), 2)
            cv2.imshow("detected", img2)
            cv2.waitKey(0)

        if returnArray is False:
            if len(rectangles) > 0:
                return rectangles
            else:
                return False
        else:
            return rectangles

    def checkCharacter(array, digit):
        exist = False
        for value in array:
            if digit in value['digit']:
                exist = True
                break
        return exist

    def getDigits(img, threshold=0.95):
        digits = []
        for i in range(10):
            template = Image.TARGETS[str(i)]

            positions = OCR.positions(
                target=template, baseImage=img, threshold=threshold, returnArray=True)
            if len(positions) > 0:
                for position in positions:
                    digits.append({'digit': str(i), 'x': position[0]})

            templateDot = Image.TARGETS['dot']
            positionDot = OCR.positions(
                target=templateDot, baseImage=img, threshold=threshold, returnArray=True)
            if len(positionDot) > 0 and OCR.checkCharacter(digits, '.') == False:
                digits.append({'digit': '.', 'x': positionDot[0][0]})

            templateComma = Image.TARGETS['comma']
            positionComma = OCR.positions(
                target=templateComma, baseImage=img, threshold=threshold, returnArray=True)
            if len(positionComma) > 0 and OCR.checkCharacter(digits, ',') == False:
                digits.append({'digit': ',', 'x': positionComma[0][0]})

        def getX(e):
            return e['x']

        digits.sort(key=getX)
        r = list(map(lambda x: x['digit'], digits))
        return(''.join(r))