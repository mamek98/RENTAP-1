import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import torch

class TacticalTurretTracker:
    """
    High-Performance Tactical Multi-Object Tracker for Sentry Turrets.
    
    Designed specifically to overcome standard ByteTrack limitations on moving pan/tilt heads:
    1. Centroid & IoU Hybrid Association: Prevents track ID flipping when turret servos rotate fast.
    2. Dual-Threshold Hysteresis: Strict (conf >= 0.40) to initiate new tracks, Lenient (conf >= 0.15) to maintain locks.
    3. Exponential Moving Average (EMA) Smoothing: Eliminates high-frequency coordinate jitter (alpha = 0.35).
    4. Velocity Extrapolation & Coasting: Holds target lock for up to 30 frames (~1 sec) during momentary occlusions,
       sun glare, or banking, preventing servo shudder and lost locks in main_system.py.
    """
    def __init__(self, max_memory=30, alpha=0.35, strict_conf=0.40, track_conf=0.15, max_dist=180.0):
        self.max_memory = max_memory
        self.alpha = alpha
        self.strict_conf = strict_conf
        self.track_conf = track_conf
        self.max_dist = max_dist
        self.tracks = {}  # track_id -> dict
        self.next_id = 1

    def _calc_iou(self, box1, box2):
        """Calculates Intersection over Union between two [x1, y1, x2, y2] boxes."""
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        boxBArea = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        denom = float(boxAArea + boxBArea - interArea)
        return interArea / denom if denom > 0 else 0.0

    def update(self, raw_boxes, raw_confs, raw_clss, class_names=None):
        """
        Updates the tactical tracker state with fresh raw detector outputs.
        Returns: list of [x1, y1, x2, y2, track_id, conf, class_name, is_coasting]
        """
        # Step 1: Age existing tracks & apply velocity coasting
        for tid in list(self.tracks.keys()):
            self.tracks[tid]['life'] -= 1
            if self.tracks[tid]['life'] <= 0:
                del self.tracks[tid]
            else:
                # Coast position with smooth velocity
                vx, vy = self.tracks[tid]['velocity']
                self.tracks[tid]['box'] += np.array([vx, vy, vx, vy], dtype=np.float32)
                self.tracks[tid]['is_coasting'] = True

        if len(raw_boxes) == 0:
            return self._get_active_tracks()

        det_centroids = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in raw_boxes]
        matched_tracks = set()
        matched_dets = set()

        # Step 2: Associate detections to active tracks
        # Stage 2A: High-confidence IoU association (for smooth standard motion)
        for t_id, t_data in self.tracks.items():
            t_box = t_data['box']
            best_det_idx = None
            max_iou = 0.25  # IoU matching threshold

            for d_idx, d_box in enumerate(raw_boxes):
                if d_idx in matched_dets:
                    continue
                if raw_confs[d_idx] < self.track_conf:
                    continue
                iou = self._calc_iou(t_box, d_box)
                if iou > max_iou:
                    max_iou = iou
                    best_det_idx = d_idx

            if best_det_idx is not None:
                matched_tracks.add(t_id)
                matched_dets.add(best_det_idx)
                self._update_track(t_id, raw_boxes[best_det_idx], raw_confs[best_det_idx], raw_clss[best_det_idx], class_names)

        # Stage 2B: Centroid Proximity Association (Tolerates fast camera pan/tilt rotation)
        for t_id, t_data in self.tracks.items():
            if t_id in matched_tracks:
                continue
            t_box = t_data['box']
            tcx, tcy = (t_box[0] + t_box[2]) / 2.0, (t_box[1] + t_box[3]) / 2.0
            
            best_det_idx = None
            min_dist = float('inf')

            for d_idx, (dcx, dcy) in enumerate(det_centroids):
                if d_idx in matched_dets:
                    continue
                if raw_confs[d_idx] < self.track_conf:
                    continue
                dist = np.hypot(dcx - tcx, dcy - tcy)
                if dist < self.max_dist and dist < min_dist:
                    min_dist = dist
                    best_det_idx = d_idx

            if best_det_idx is not None:
                matched_tracks.add(t_id)
                matched_dets.add(best_det_idx)
                self._update_track(t_id, raw_boxes[best_det_idx], raw_confs[best_det_idx], raw_clss[best_det_idx], class_names)

        # Step 3: Spawn new tracks for unmatched detections (must satisfy strict_conf)
        for d_idx, box in enumerate(raw_boxes):
            if d_idx in matched_dets:
                continue
            conf = raw_confs[d_idx]
            if conf >= self.strict_conf:
                new_box = np.array(box, dtype=np.float32)
                cls_idx = raw_clss[d_idx]
                c_name = class_names.get(cls_idx, "drone") if class_names else "drone"

                self.tracks[self.next_id] = {
                    'box': new_box,
                    'velocity': (0.0, 0.0),
                    'conf': conf,
                    'cls': cls_idx,
                    'class_name': c_name,
                    'life': self.max_memory,
                    'is_coasting': False
                }
                self.next_id += 1

        return self._get_active_tracks()

    def _update_track(self, t_id, det_box, det_conf, det_cls, class_names):
        """Applies EMA smoothing and updates velocity state for a matched track."""
        t_data = self.tracks[t_id]
        new_box = np.array(det_box, dtype=np.float32)
        old_box = t_data['box']

        # EMA Coordinate Smoothing: eliminates detector jitter
        smoothed_box = self.alpha * new_box + (1.0 - self.alpha) * old_box

        # Instantaneous & filtered velocity estimation
        new_cx, new_cy = (new_box[0] + new_box[2]) / 2.0, (new_box[1] + new_box[3]) / 2.0
        old_cx, old_cy = (old_box[0] + old_box[2]) / 2.0, (old_box[1] + old_box[3]) / 2.0
        vx = (new_cx - old_cx) * 0.35 + t_data['velocity'][0] * 0.65
        vy = (new_cy - old_cy) * 0.35 + t_data['velocity'][1] * 0.65

        c_name = class_names.get(det_cls, "drone") if class_names else "drone"

        t_data['box'] = smoothed_box
        t_data['velocity'] = (vx, vy)
        t_data['conf'] = det_conf
        t_data['cls'] = det_cls
        t_data['class_name'] = c_name
        t_data['life'] = self.max_memory
        t_data['is_coasting'] = False

    def _get_active_tracks(self):
        """Returns all currently active (live and coasting) tracks."""
        results = []
        for tid, data in self.tracks.items():
            b = data['box']
            results.append([
                int(b[0]), int(b[1]), int(b[2]), int(b[3]),
                tid,
                float(data['conf']),
                data['class_name'],
                data['is_coasting']
            ])
        return results


class TurretVision:
    """
    High-Performance AI Vision Pipeline for RENTAP-1 Sentry Turret.
    
    Optimized for Real-Time Embedded & Laptop GPUs (e.g. RTX 3050 Ti):
    - Mode 1: Tactical Turret Tracker (Default): Runs at 70-80+ FPS using CUDA FP16 Tensor Cores.
              Centroid + IoU hybrid association prevents ID switches when the turret moves.
              EMA coordinate smoothing (alpha=0.35) removes jitter.
              Ghost coasting prevents lost target lock in outdoor sky glare.
    - Mode 2: SAHI + StrongSORT (Legacy/Benchmark): Available via use_sahi=True for offline/thesis testing.
    """
    def __init__(self, model_path, use_sahi=False, conf_threshold=0.35):
        self.use_sahi = use_sahi
        self.conf_threshold = conf_threshold
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.half = (self.device != 'cpu')
        
        print(f"[SYSTEM] Loading RENTAP AI Engine (Device: {self.device}, FP16: {self.half})...")
        
        if not self.use_sahi:
            # High-speed native YOLOv8 with Tactical Turret Tracker (70-80+ FPS)
            self.model = YOLO(model_path)
            self.tracker = TacticalTurretTracker(
                max_memory=30,      # ~1.0 sec coasting buffer
                alpha=0.35,         # Buttery-smooth EMA coordinate filter
                strict_conf=0.40,   # High barrier for new targets (filters false alarms)
                track_conf=0.15,    # Lenient tracking for existing targets (holds locks through glare/turn)
                max_dist=180.0      # Tolerates rapid servo pan/tilt rotation jumps
            )
            print("[SYSTEM] High-Speed Tactical Turret Tracker Pipeline Ready (DJI-Style 70+ FPS).")
        else:
            # Legacy SAHI + StrongSORT (Phase 1 evaluation)
            print("[SYSTEM] Loading Legacy SAHI + StrongSORT Pipeline...")
            from sahi import AutoDetectionModel
            from boxmot import StrongSort
            self.detection_model = AutoDetectionModel.from_pretrained(
                model_type='yolov8',
                model_path=model_path,
                confidence_threshold=conf_threshold,
                device=self.device
            )
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            self.tracker = StrongSort(
                reid_weights=Path('osnet_x0_25_msmt17.pt'),
                device=self.device,
                half=self.half,
                max_cos_dist=0.4,
                max_iou_dist=0.6,
                max_age=30,
                n_init=1,
                nn_budget=100
            )

    def preprocess_frame(self, frame):
        """CLAHE enhancement (Used in low-contrast/night situations if needed)."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        cl = self.clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    def _draw_tactical_box(self, img, x1, y1, x2, y2, color, label, is_coasting=False):
        """Renders crisp military tactical bracket overlays with corner accents."""
        # Main bounding box outline
        thickness = 1 if is_coasting else 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Tactical corner brackets (25% of box dimension)
        corner_len = max(8, min(24, int(min(x2 - x1, y2 - y1) * 0.25)))
        t_corner = 2
        # Top-left
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, t_corner)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, t_corner)
        # Top-right
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, t_corner)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, t_corner)
        # Bottom-left
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, t_corner)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, t_corner)
        # Bottom-right
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, t_corner)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, t_corner)

        # Clean label with dark backdrop for outdoor readability
        font_scale = 0.5
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ty = max(th + 6, y1 - 6)
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + baseline), (20, 20, 20), -1)
        cv2.putText(img, label, (x1 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

    def scan_frame(self, frame):
        """
        Processes a video frame and returns:
        - annotated_frame: clean video frame with tactical bounding box overlays
        - formatted_boxes: list of [x1, y1, x2, y2, track_id, conf, class_name]
        """
        if not self.use_sahi:
            # === HIGH-SPEED TACTICAL TRACKER PIPELINE ===
            # Run YOLOv8 on GPU at 640x640 with FP16 (~11.8ms on RTX 3050 Ti)
            results = self.model.predict(
                frame,
                imgsz=640,
                device=self.device,
                half=self.half,
                conf=self.tracker.track_conf,  # Lenient conf captures small/distant drones
                verbose=False
            )

            raw_boxes = []
            raw_confs = []
            raw_clss = []

            if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                raw_boxes = boxes.xyxy.cpu().numpy().tolist()
                raw_confs = boxes.conf.cpu().numpy().tolist()
                raw_clss = boxes.cls.int().cpu().tolist() if boxes.cls is not None else [0] * len(raw_boxes)

            # Update tactical tracker with EMA smoothing & centroid association
            active_tracks = self.tracker.update(raw_boxes, raw_confs, raw_clss, class_names=self.model.names)

            annotated_frame = frame.copy()
            formatted_boxes = []
            h, w = frame.shape[:2]

            for track in active_tracks:
                x1, y1, x2, y2, tid, conf, c_name, is_coasting = track
                # Clamp coordinates to frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Append to formatted output for main_system target selection
                formatted_boxes.append([x1, y1, x2, y2, tid, conf, c_name])

                # Color coding: Green for locked active target, Amber/Orange for coasting target
                if is_coasting:
                    color = (0, 165, 255)  # Amber / Orange
                    label = f"ID:{tid} {c_name} [HOLD]"
                else:
                    color = (0, 255, 0)    # Tactical Green
                    label = f"ID:{tid} {c_name} {conf:.2f}"

                self._draw_tactical_box(annotated_frame, x1, y1, x2, y2, color, label, is_coasting=is_coasting)

            return annotated_frame, formatted_boxes

        else:
            # === LEGACY SAHI + STRONGSORT PIPELINE ===
            from sahi.predict import get_sliced_prediction
            enhanced_frame = self.preprocess_frame(frame)

            result = get_sliced_prediction(
                enhanced_frame,
                self.detection_model,
                slice_height=720,
                slice_width=720,
                overlap_height_ratio=0.1,
                overlap_width_ratio=0.1,
                verbose=0
            )

            dets = []
            for obj in result.object_prediction_list:
                bbox = obj.bbox
                dets.append([bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, obj.score.value, obj.category.id])

            dets = np.array(dets)

            if len(dets) > 0:
                tracks = self.tracker.update(dets, enhanced_frame)
            else:
                tracks = self.tracker.update(np.empty((0, 6)), enhanced_frame)

            annotated_frame = enhanced_frame.copy()
            formatted_boxes = []

            for track in tracks:
                x1, y1, x2, y2, track_id, conf, cls_id = track[:7]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                track_id = int(track_id)
                class_name = "drone"

                formatted_boxes.append([x1, y1, x2, y2, track_id, conf, class_name])

                color = (0, 255, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{track_id} {class_name} {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, max(10, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            return annotated_frame, formatted_boxes