import cv2
import threading
import time

class ThreadedCamera:
    def __init__(self,src=1,width=1280,height=720):
        self.src = src
        self.cap = cv2.VideoCapture(self.src,cv2.CAP_DSHOW)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

        (self.grabbed, self.frame) = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.thread = None

    def start(self):
        """Memulakan thread kamera di latar belakang."""
        if self.started:
            return self
        self.started = True
        # Thread dijadikan 'daemon' supaya ia automatik mati jika main_system.py terhenti
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        """Loop latar belakang yang membaca USB Kamera tanpa henti."""
        while self.started:
            grabbed, frame = self.cap.read()
            if not grabbed:
                time.sleep(0.005) # Elak penggunaan CPU melampau jika kamera terputus
                continue
            
            # Kunci pembolehubah semasa kemas kini untuk integriti memori
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        """Main Thread (AI) memanggil fungsi ini untuk dapatkan bingkai TERKINI."""
        with self.read_lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return False, None

    def stop(self):
        """Menghentikan thread dan membebaskan port kamera."""
        self.started = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()