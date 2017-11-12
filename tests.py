import cv2
import json
import util
import mob_net_cls

img = util.read_image_as_nparr_RGB('./images/elephant.jpeg')

if __name__ == "__main__":
    preds = mob_net_cls.predict(img)
    print(json.dumps(preds))