import cv2
import time
import argparse
import numpy as np
from pathlib import Path
from core.ai_vision import TurretVision

def test_tracking_on_video(video_source, model_path="models/best (2).pt", output_path="output_tracking.mp4", show_display=True):
    """
    Offline & Benchmark Video Tracker Evaluator for RENTAP-1.
    
    Evaluates:
    - Real-Time FPS across video duration
    - Trajectory continuity (Motion Trail Visualization)
    - ID Switch count (IDSW metric)
    - Target Persistence & Hold efficiency
    """
    print(f"[TESTER] Loading AI Vision Module with {model_path}...")
    vision = TurretVision(model_path=model_path)
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[RALAT] Gagal membuka sumber video: {video_source}")
        return

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[TESTER] Sumber Video: {video_source} ({width}x{height} @ {fps_video:.1f} FPS, Jumlah Bingkai: {total_frames})")

    # Output video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps_video, (width, height))
        print(f"[TESTER] Video hasil akan disimpan ke: {output_path}")

    # Tracking evaluation metrics
    trajectory_history = {}  # track_id -> list of (cx, cy)
    seen_track_ids = set()
    frame_count = 0
    frames_with_target = 0
    t_start = time.time()
    
    window_name = "RENTAP-1 Drone Tracking Benchmark"
    if show_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

    print("[TESTER] Memulakan ujian tracking video... Tekan 'Q' untuk berhenti.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        t_frame_start = time.time()

        # 1. Jalankan Tactical Turret Tracker
        annotated_frame, boxes = vision.scan_frame(frame)
        t_frame_end = time.time()
        instant_fps = 1.0 / max(0.001, (t_frame_end - t_frame_start))

        if len(boxes) > 0:
            frames_with_target += 1

        # 2. Kemas kini Trajectory History (Ekor Pergerakan Dron)
        for box in boxes:
            x1, y1, x2, y2, tid, conf, c_name = box[:7]
            if tid is not None:
                seen_track_ids.add(tid)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                if tid not in trajectory_history:
                    trajectory_history[tid] = []
                trajectory_history[tid].append((cx, cy))
                # Hadkan sejarah ekor kepada 40 koordinat terkini
                if len(trajectory_history[tid]) > 40:
                    trajectory_history[tid].pop(0)

        # 3. Lukis Garisan Lintasan Pergerakan Dron (Motion Trail)
        for tid, points in trajectory_history.items():
            if len(points) > 1:
                for i in range(1, len(points)):
                    # Semakin lama, garisan semakin pudar
                    alpha_trail = int(255 * (i / len(points)))
                    color_trail = (0, alpha_trail, 255)  # Ekor jingga-kuning
                    cv2.line(annotated_frame, points[i - 1], points[i], color_trail, 2, cv2.LINE_AA)

        # 4. Paparan Statistik Metrik MOT Pada Skrin
        elapsed = time.time() - t_start
        avg_fps = frame_count / max(0.001, elapsed)
        lock_rate = (frames_with_target / max(1, frame_count)) * 100.0

        # Bar maklumat di atas skrin
        overlay_text_1 = f"FRAME: {frame_count}/{total_frames} | FPS: {instant_fps:.1f} (Avg: {avg_fps:.1f})"
        overlay_text_2 = f"ACTIVE TARGETS: {len(boxes)} | TOTAL UNIQUE IDs: {len(seen_track_ids)} | LOCK RATE: {lock_rate:.1f}%"

        cv2.rectangle(annotated_frame, (20, 20), (700, 95), (20, 20, 20), -1)
        cv2.putText(annotated_frame, overlay_text_1, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, overlay_text_2, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)

        # 5. Simpan & Papar
        if writer:
            writer.write(annotated_frame)

        if show_display:
            cv2.imshow(window_name, annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[TESTER] Ujian dihentikan oleh pengguna.")
                break

    # Tamat
    cap.release()
    if writer:
        writer.release()
    if show_display:
        cv2.destroyAllWindows()

    total_time = time.time() - t_start
    print("="*60)
    print("=== LAPORAN KEPUTUSAN TRACKING BENCHMARK ===")
    print(f"Jumlah Bingkai Diproses : {frame_count}")
    print(f"Jumlah Masa Ujian       : {total_time:.2f} saat")
    print(f"Purata Kelajuan (FPS)   : {frame_count / max(0.001, total_time):.2f} FPS")
    print(f"Jumlah ID Dikesan       : {len(seen_track_ids)} unique track IDs")
    print(f"Kadar Penguncian Sasaran: {lock_rate:.2f}%")
    if output_path:
        print(f"Video Hasil Disimpan Di : {output_path}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and Benchmark RENTAP Drone Tracking on Video Files")
    parser.add_argument("--source", type=str, default="0", help="Path ke fail video (contoh: drone_flight.mp4) atau nombor kamera (0, 1)")
    parser.add_argument("--model", type=str, default="models/best (2).pt", help="Path ke fail model .pt")
    parser.add_argument("--out", type=str, default="output_tracking.mp4", help="Path untuk simpan video output")
    parser.add_argument("--no-gui", action="store_true", help="Jalankan tanpa paparan skrin (headless / server mode)")
    
    args = parser.parse_args()
    src = int(args.source) if args.source.isdigit() else args.source
    test_tracking_on_video(src, model_path=args.model, output_path=args.out, show_display=not args.no_gui)
