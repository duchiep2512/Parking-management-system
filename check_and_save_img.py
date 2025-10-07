
import os
import numpy as np
import cv2
from datetime import datetime
import shutil

path_root = 'image_data'

class CheckAndSaveImg():
    def check_exists(self, label):
        return os.path.exists(os.path.join(path_root, label))

    def save_image(self, label, image_array: np.ndarray):
        now = datetime.now()
        name = now.strftime("%H-%M-%S %d-%m-%Y")
        path_dir = os.path.join(path_root, label)
        os.makedirs(path_dir, exist_ok=True)
        cv2.imwrite(os.path.join(path_dir, name + '.jpg'), image_array)

    def delete_image(self, label):
        path_dir = os.path.join(path_root, label)
        try:
            shutil.rmtree(path_dir)
        except FileNotFoundError:
            pass

    def get_data(self, label):
        path = os.path.join(path_root, label)
        if not os.path.exists(path):
            return "Xe chưa vào bãi", None
        imgs = sorted(os.listdir(path))
        if not imgs:
            return "Xe chưa vào bãi", None
        img_name = imgs[0]
        img_array = cv2.imread(os.path.join(path, img_name))
        date_time_img = img_name[:-4].replace('-', ':', 2)
        return date_time_img, img_array
