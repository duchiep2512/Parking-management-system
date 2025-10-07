
import cv2
import numpy as np
from ultralytics import YOLO

def format_vn_plate(s: str):
    s = s.replace(" ", "").replace("_","").upper()
    s = (s.replace("O","0").replace("I","1").replace("L","1")
           .replace("S","5").replace("B","8"))
    if len(s) >= 7:
        head, tail = s[:3], s[3:]
        if len(tail) == 4:  return f"{head}-{tail[:2]}.{tail[2:]}"
        if len(tail) == 5:  return f"{head}-{tail[:3]}.{tail[3:]}"
    return s

def order_quad(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1); rect[0]=pts[np.argmin(s)]; rect[2]=pts[np.argmax(s)]
    d = np.diff(pts, axis=1).ravel(); rect[1]=pts[np.argmin(d)]; rect[3]=pts[np.argmax(d)]
    return rect

def warp_from_poly(img, poly_xy):
    cnt  = np.array(poly_xy, dtype=np.float32)
    rect = cv2.minAreaRect(cnt)
    box  = cv2.boxPoints(rect).astype(np.float32)
    box  = order_quad(box)
    w = int(max(np.linalg.norm(box[0]-box[1]), np.linalg.norm(box[2]-box[3])))
    h = int(max(np.linalg.norm(box[0]-box[3]), np.linalg.norm(box[1]-box[2])))
    w = max(w, 160); h = max(h, 44)
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(box, dst)
    return cv2.warpPerspective(img, M, (w,h))

class OcrPlate:
    def __init__(self, path_model_detect_plate, path_model_char,
                 det_conf_plate=0.25, det_conf_char=0.30, iou_char=0.5,
                 expand_margin=0.08, two_line_thr=0.25, imgsz=832):
        """
        path_model_detect_plate: YOLO seg/det model for license plates (.pt or .onnx)
        path_model_char: YOLO det model for characters (.pt or .onnx)
        """
        self.model_plate = YOLO(path_model_detect_plate)
        self.model_char  = YOLO(path_model_char)
        self.names_char  = self.model_char.names

        self.det_conf_plate = det_conf_plate
        self.det_conf_char  = det_conf_char
        self.iou_char       = iou_char
        self.expand_margin  = expand_margin
        self.two_line_thr   = two_line_thr
        self.imgsz          = imgsz

    def _expand_box(self, x1,y1,x2,y2,W,H,margin):
        cx, cy = (x1+x2)/2, (y1+y2)/2
        bw, bh = (x2-x1)*(1+2*margin), (y2-y1)*(1+2*margin)
        nx1 = int(max(0, cx-bw/2)); ny1 = int(max(0, cy-bh/2))
        nx2 = int(min(W-1, cx+bw/2)); ny2 = int(min(H-1, cy+bh/2))
        return nx1, ny1, nx2, ny2

    def _read_chars_in_roi(self, roi):
        r = self.model_char.predict(roi, conf=self.det_conf_char, iou=self.iou_char, verbose=False)[0]
        if len(r.boxes) == 0:
            return "unknown", 0.0
        xywh = r.boxes.xywh.cpu().numpy()
        cls  = r.boxes.cls.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        keep = conf >= self.det_conf_char
        xywh, cls, conf = xywh[keep], cls[keep], conf[keep]
        if xywh.shape[0] == 0:
            return "unknown", 0.0

        xc, yc = xywh[:,0], xywh[:,1]
        y_range = (yc.max() - yc.min()) / max(1.0, roi.shape[0])
        if y_range > self.two_line_thr:
            y_mid = np.median(yc)
            top_idx = np.where(yc < y_mid)[0]
            bot_idx = np.where(yc >= y_mid)[0]
            top_order = top_idx[np.argsort(xc[top_idx])]
            bot_order = bot_idx[np.argsort(xc[bot_idx])]
            line1 = "".join([str(self.names_char[int(c)]) for c in cls[top_order]])
            line2 = "".join([str(self.names_char[int(c)]) for c in cls[bot_order]])
            raw = f"{line1}-{line2}"
        else:
            order = np.argsort(xc)
            raw = "".join([str(self.names_char[int(c)]) for c in cls[order]])
        return format_vn_plate(raw), float(np.mean(conf)) if conf.size else 0.0

    def infer_image(self, image_bgr):
        H, W = image_bgr.shape[:2]
        res = self.model_plate(image_bgr, conf=self.det_conf_plate, imgsz=self.imgsz, verbose=False)[0]
        annotated = res.plot()

        text, dbg, sc, roi_used = "unknown", "no-plate", 0.0, None

        if res.masks is not None and len(res.masks.xy):
            areas = [cv2.contourArea(np.array(x, dtype=np.float32)) for x in res.masks.xy]
            k = int(np.argmax(areas))
            roi = warp_from_poly(image_bgr, res.masks.xy[k])
            text, sc = self._read_chars_in_roi(roi)
            roi_used = roi
            dbg = f"mask ROI={roi.shape[1]}x{roi.shape[0]} | score={sc:.2f}"
        elif len(res.boxes):
            b = res.boxes
            xyxy = b.xyxy.cpu().numpy()
            conf = b.conf.cpu().numpy()
            areas = (xyxy[:,2]-xyxy[:,0]) * (xyxy[:,3]-xyxy[:,1])
            i = int(np.argmax(areas * conf))
            x1,y1,x2,y2 = map(int, xyxy[i])
            x1,y1,x2,y2 = self._expand_box(x1,y1,x2,y2,W,H,self.expand_margin)
            roi = image_bgr[y1:y2, x1:x2]
            if roi.size:
                text, sc = self._read_chars_in_roi(roi)
                roi_used = roi
                dbg = f"bbox ROI={roi.shape[1]}x{roi.shape[0]} | score={sc:.2f}"

        # vẽ nhãn
        if len(res.boxes):
            x1,y1,x2,y2 = map(int, res.boxes.xyxy[0].tolist())
            cv2.putText(annotated, text, (max(0,x1), max(50,y1-68)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,255), 2, cv2.LINE_AA)

        return annotated, text, sc, roi_used

