"""from ultralytics import YOLO

class TurretVision:
    def __init__(self, model_path):
        #Load model masa sistem mula-mula hidup
        print("[SYSTEM] Loading RENTAP AI Brain...")
        self.model = YOLO(model_path)

    def scan_frame(self,frame):
        # Paksa resolusi 640 supaya laju, macam yang kita bincang sebelum ni
        results = self.model(frame, imgsz=640, conf=0.7, iou=0.45, verbose=False)

        #Ambil frame yang dah siap dilukis bounding box oleh YOLO
        annotated_frame = results[0].plot()
        return annotated_frame, results[0].boxes

from ultralytics import YOLO
import cv2

class TurretVision:
    def __init__ (self,model_path):
        print("[SYSTEM] Loading RENTAP AI Brain...")
        self.model = YOLO(model_path)

        self.box_history = {}

    def scan_frame(self,frame):
        results = self.model.track(frame, imgsz=640, conf=0.25,iou=0.65,max_det=100, persist=True, tracker="bytetrack.yaml", verbose=False)

        annotated_frame = frame.copy()
        boxes_data = results[0].boxes
        current_boxes = {}

        if boxes_data.id is not None:
            track_ids = boxes_data.id.int().cpu().tolist()
            boxes = boxes_data.xyxy.cpu().numpy()
            confs = boxes_data.conf.cpu().numpy()
            classes = boxes_data.cls.cpu().numpy()
            names = self.model.names

            for box,track_id,conf,cls in zip(boxes,track_ids,confs,classes):
                x1,y1,x2,y2 = box
                alpha=0.3
             
                if track_id in self.box_history:
                    old_box = self.box_history[track_id]
                    x1 = (alpha * x1) + ((1-alpha)* old_box[0])
                    y1 = (alpha * y1) + ((1-alpha)* old_box[1])
                    x2 = (alpha * x2) + ((1-alpha)* old_box[2])
                    y2 = (alpha * y2) + ((1-alpha)* old_box[3])
                
                current_boxes[track_id] = [x1,y1,x2,y2]

                x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)

                nama_kelas = names[int(cls)]
                color = (0,0,255) if nama_kelas == "armed_drone" else (0,255,255)

                cv2.rectangle(annotated_frame, (x1,y1), (x2,y2), color, 2)

                # Teks yang lebih kemas
                label = f"ID:{track_id} {nama_kelas} {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Kemas kini ingatan masa lalu
        self.box_history = current_boxes 

        return annotated_frame, boxes_data

           
      
from ultralytics import YOLO
import cv2

class TurretVision:
    def __init__ (self, model_path):
        print("[SYSTEM] Loading RENTAP AI Brain...")
        self.model = YOLO(model_path)

        # Memori ini bertindak sebagai pemegang status Kalman Filter
        self.box_history = {}
        # 30 frame = ~1 saat. Turet akan "teka" kedudukan sasaran selama 1 saat walaupun buta
        self.max_memory_frames = 30 

    def scan_frame(self, frame):
        # conf=0.25 dan iou=0.65 membenarkan ByteTrack (Kalman Filter) berfungsi di tahap optimum
        results = self.model.track(frame, imgsz=640, conf=0.7, iou=0.65, max_det=100, persist=True, tracker="bytetrack.yaml", verbose=False)

        annotated_frame = frame.copy()
        boxes_data = results[0].boxes
        
        # --- 1. FASA PENURUNAN NYAWA (KALMAN DECAY) ---
        # Setiap frame, kita tolak nyawa semua objek yang direkodkan
        for track_id in list(self.box_history.keys()):
            self.box_history[track_id]['life'] -= 1
            # Jika lebih 1 saat tak nampak, baru anggap ia musnah/hilang
            if self.box_history[track_id]['life'] <= 0:
                del self.box_history[track_id] 

        # --- 2. FASA PEMBAHARUAN DATA AI ---
        if boxes_data.id is not None:
            track_ids = boxes_data.id.int().cpu().tolist()
            boxes = boxes_data.xyxy.cpu().numpy()
            confs = boxes_data.conf.cpu().numpy()
            classes = boxes_data.cls.cpu().numpy()
            names = self.model.names

            for box, track_id, conf, cls in zip(boxes, track_ids, confs, classes):
                x1, y1, x2, y2 = box
                alpha = 0.3 # EMA Smoothing
                nama_kelas = names[int(cls)]
             
                # Kalau objek dah ada dalam memori, kita campurkan koordinat (Smooth Motion)
                if track_id in self.box_history:
                    old_box = self.box_history[track_id]['coords']
                    x1 = (alpha * x1) + ((1 - alpha) * old_box[0])
                    y1 = (alpha * y1) + ((1 - alpha) * old_box[1])
                    x2 = (alpha * x2) + ((1 - alpha) * old_box[2])
                    y2 = (alpha * y2) + ((1 - alpha) * old_box[3])
                
                # Masukkan data ke dalam 'Kamus Memori' bersama nyawa penuh
                self.box_history[track_id] = {
                    'coords': [x1, y1, x2, y2],
                    'life': self.max_memory_frames, # Reset nyawa sebab AI nampak dengan jelas
                    'conf': conf,
                    'class_name': nama_kelas
                }

        # --- 3. FASA LUKISAN (RENDER) ---
        # Kita lukis kotak berdasarkan MEMORI, bukan data mentah AI
        for track_id, data in self.box_history.items():
            x1, y1, x2, y2 = map(int, data['coords'])
            nama_kelas = data['class_name']
            conf = data['conf']
            life = data['life']

            color = (0, 0, 255) if nama_kelas == "armed_drone" else (0, 255, 255)

            # Jika nyawa kurang dari maksimum (bermaksud AI sedang buta sementara)
            if life < self.max_memory_frames:
                color_oren = (0, 165, 255) 
                label = f"ID:{track_id} {nama_kelas} [PREDICTING]"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_oren, 2)
                cv2.putText(annotated_frame, label, (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_oren, 2)
            else:
                label = f"ID:{track_id} {nama_kelas} {conf:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return annotated_frame, boxes_data
             
      

from ultralytics import YOLO
import cv2
import numpy as np

class TurretVision:
    def __init__ (self, model_path):
        print("[SYSTEM] Loading RENTAP AI Brain ")
        self.model = YOLO(model_path)

        self.box_history = {}
        self.max_memory_frames = 60 

        self.strict_conf = 0.4
        self.lenient_conf = 0.15

    def scan_frame(self, frame):

        results = self.model.track(
            frame , 
            imgsz = 640,
            conf=self.lenient_conf,
            iou = 0.65,
            max_det=100,
            persist=True,
            tracker="botsort.yaml",
            verbose=False)
        
        annotated_frame = frame.copy()
        boxes_data = results[0].boxes


        for track_id in list(self.box_history.keys()):
            data = self.box_history[track_id]
            data['life'] -=1

            if data['life'] <= 0:
                del self.box_history[track_id]
            else:
                old_coords = data['coords']
                dx ,dy = data.get('velocity', (0,0))
                
                new_x1 = old_coords[0] + dx
                new_y1 = old_coords[1] + dy
                new_x2 = old_coords[2] + dx
                new_y2 = old_coords[3] + dy

                self.box_history[track_id]['coords'] = [new_x1, new_y1, new_x2, new_y2]

        if boxes_data.id is not None:
            tracks_ids = boxes_data.id.int().cpu().tolist()
            boxes = boxes_data.xyxy.cpu().numpy()
            confs = boxes_data.conf.cpu().numpy()
            classes = boxes_data.cls.cpu().numpy()
            names = self.model.names

            for box, track_id, conf, cls in zip(boxes, tracks_ids, confs, classes):
                x1 , y1 , x2, y2 = box
                class_name = names[int(cls)]

                is_new_target = track_id not in self.box_history

                if is_new_target and conf < self.strict_conf:
                    continue

                alpha = 0.3
                dx , dy =0,0

                if track_id in self.box_history:
                    old_box = self.box_history[track_id]['coords']

                    dx = x1 - old_box[0]
                    dy = y1 - old_box[1]

                    x1 = (alpha * x1) + ((1-alpha) * old_box[0])
                    y1 = (alpha * y1) + ((1-alpha) * old_box[1])
                    x2 = (alpha * x2) + ((1-alpha) * old_box[2])
                    y2 = (alpha * y2) + ((1-alpha) * old_box[3])

                self.box_history[track_id] = {
                    'coords': [x1, y1, x2, y2],   
                    'velocity' : (dx,dy),
                    'life': self.max_memory_frames,
                    'conf' : conf,
                    'class_name' : class_name }     
        
        for track_id, data in self.box_history.items():
            x1 ,y1 ,x2,y2 = map(int,data['coords'])
            class_name = data['class_name']
            conf = data['conf']
            life = data['life']

            color = (0,0,255) if class_name == "armed_drone" else (0,255,255)

            if life < self.max_memory_frames:
                orange_color = (0,165,255)
                label = f"ID:{track_id} {class_name} [PREDICTING]"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), orange_color, 2)
                cv2.putText(annotated_frame, label, (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, orange_color, 2)
            else:
                label = f"ID:{track_id} {class_name} {conf:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return annotated_frame, boxes_data
         
           """

from ultralytics import YOLO
import cv2
import numpy as np

class TurretVision:
    def __init__ (self, model_path):
        print("[SYSTEM] Loading RENTAP AI Brain (BoT-SORT Engine)...")
        self.model = YOLO(model_path)

        self.box_history = {}
        # Kita kurangkan memori hantu kepada 15 frame (~0.5 saat). 
        # Cukup untuk cover 'blink' atau silau, tapi tak serabutkan skrin bila dron betul-betul lari.
        self.max_memory_frames = 30 
        
        self.strict_conf = 0.50  
        self.lenient_conf = 0.20 

    def scan_frame(self, frame):
        # BoT-SORT kekal berjalan untuk elak ID bertukar
        results = self.model.track(frame, imgsz=640, conf=self.lenient_conf, iou=0.65, max_det=100, persist=True, tracker="botsort.yaml", verbose=False)

        annotated_frame = frame.copy()
        boxes_data = results[0].boxes
        
        # --- FASA 1: PENURUNAN NYAWA (STATIC GHOST TRACKING) ---
        for track_id in list(self.box_history.keys()):
            self.box_history[track_id]['life'] -= 1
            if self.box_history[track_id]['life'] <= 0:
                del self.box_history[track_id] 
            # LOGIK EKSTRAPOLASI (VELOCITY) TELAH DIBUANG.
            # Kotak kini akan beku dengan kemas di lokasi terakhir ia dilihat.

        # --- FASA 2: PEMBAHARUAN DATA AI ---
        if boxes_data.id is not None:
            track_ids = boxes_data.id.int().cpu().tolist()
            boxes = boxes_data.xyxy.cpu().numpy()
            confs = boxes_data.conf.cpu().numpy()
            classes = boxes_data.cls.cpu().numpy()
            names = self.model.names

            for box, track_id, conf, cls in zip(boxes, track_ids, confs, classes):
                x1, y1, x2, y2 = box
                nama_kelas = names[int(cls)]
                
                is_new_target = track_id not in self.box_history
                if is_new_target and conf < self.strict_conf:
                    continue 
                
                # Kita naikkan alpha sikit (0.4) supaya pergerakan semasa tracking lebih responsif (tak melayang)
                alpha = 0.4 
             
                if track_id in self.box_history:
                    old_box = self.box_history[track_id]['coords']
                    x1 = (alpha * x1) + ((1 - alpha) * old_box[0])
                    y1 = (alpha * y1) + ((1 - alpha) * old_box[1])
                    x2 = (alpha * x2) + ((1 - alpha) * old_box[2])
                    y2 = (alpha * y2) + ((1 - alpha) * old_box[3])
                
                self.box_history[track_id] = {
                    'coords': [x1, y1, x2, y2],
                    'life': self.max_memory_frames,
                    'conf': conf,
                    'class_name': nama_kelas
                }

        # --- FASA 3: LUKISAN (RENDER) MINIMALIS ---
        for track_id, data in self.box_history.items():
            x1, y1, x2, y2 = map(int, data['coords'])
            nama_kelas = data['class_name']
            conf = data['conf']
            life = data['life']

            color = (0, 0, 255) if nama_kelas == "armed_drone" else (0, 255, 255)

            if life < self.max_memory_frames:
                # GHOST BOX STATIK: Warna Oren, teks lebih pendek
                color_oren = (0, 165, 255) 
                label = f"ID:{track_id} [LOST]"
                # Guna FONT saiz 0.5 supaya tak serabut penuh skrin
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_oren, 2)
                cv2.putText(annotated_frame, label, (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_oren, 2)
            else:
                label = f"ID:{track_id} {nama_kelas} {conf:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return annotated_frame, boxes_data