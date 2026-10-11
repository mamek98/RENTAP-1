# Import OpenCV library for computer vision tasks such as capturing video frames from cameras
import cv2
# Import threading module to allow the camera stream to run in a separate background thread without blocking the main program
import threading
# Import time module for adding slight delays to prevent high CPU usage when waiting for frames
import time
# Import numpy for fast matrix/array operations such as calculating image brightness histograms
import numpy as np


class ThreadedCamera:
    """
    A class to handle video capture in a separate background thread.
    This prevents the main application loop from waiting for the camera hardware to provide frames,
    improving overall performance and reducing latency.

    SELF-HEALING VERSION (auto-reconnect) + SMART FIELD AEC/AGC (Auto-Exposure & Gain):
    The turret camera is mounted on the moving pan/tilt head, so the USB-C link can drop for a
    split second when the servos move. When that happens, Windows removes the device and re-adds it
    as a NEW device instance. This class detects the dead link and opens a brand new VideoCapture automatically.

    In addition, it manages hardware auto-exposure and an active background AEC loop so webcams
    (e.g. EMEET Pico) automatically adapt to harsh outdoor field lighting and sky glare without
    requiring manual adjustments in vendor utility software.
    """
    def __init__(self, src=1, width=1920, height=1080, fps=30,
                 stale_timeout=1.5, open_grace=4.0, reconnect_delay=1.0,
                 max_failed_reads=15, black_frame_limit=30, black_threshold=4,
                 use_mjpg=True, auto_exposure=True):
        # The source index for the camera. 0 is usually the built-in webcam, 1 is typically an external USB camera.
        self.src = src
        # Requested frame width
        self.width = width
        # Requested frame height
        self.height = height
        # Requested frame rate
        self.fps = fps
        # If no NEW frame arrives within this many seconds, the camera is treated as disconnected
        self.stale_timeout = stale_timeout
        # Extra time allowed right after (re)opening, because DirectShow cameras take 1-3 s to start streaming
        self.open_grace = open_grace
        # Seconds to wait between reconnect attempts (gives Windows time to re-enumerate the USB device)
        self.reconnect_delay = reconnect_delay
        # Number of consecutive failed cap.read() calls before the link is declared dead
        self.max_failed_reads = max_failed_reads
        # Number of consecutive pure-black frames before the link is declared dead (~1 s at 30 FPS)
        self.black_frame_limit = black_frame_limit
        # A frame whose brightest (sampled) pixel is <= this value is considered "pure black"
        self.black_threshold = black_threshold
        # Request MJPG compression: 1280x720 raw YUY2 needs ~440 Mbps (almost the whole USB 2.0 bus),
        # MJPG needs only a fraction of that, which makes the link much more tolerant of cable glitches
        self.use_mjpg = use_mjpg

        # Threading lock used to prevent race conditions when reading and writing to shared state from different threads
        self.read_lock = threading.Lock()
        # Latest good frame (None until the first frame arrives)
        self.frame = None
        # Boolean indicating whether the latest frame is valid / camera is delivering frames
        self.grabbed = False
        # Timestamp of the last GOOD frame received
        self.last_frame_time = 0.0
        # True while the camera is actively delivering fresh frames
        self.connected = False
        # Number of reconnects performed since start (useful for debugging / HUD display)
        self.reconnect_count = 0
        # Number of consecutive failed reconnect attempts in the current outage
        self.reconnect_attempts = 0
        # Human-readable status string for the HUD
        self.status_text = "INITIALISING"

        # --- SMART FIELD EXPOSURE & GAIN CONTROL (AEC/AGC) ---
        self.auto_exposure = auto_exposure
        self.aec_enabled = auto_exposure
        self.current_exposure = -9.0  # Optimal outdoor starting shutter speed (log2 sec)
        self.current_gain = 0.0       # Minimal sensor gain to eliminate outdoor noise/blowout
        self.target_luminance = 115.0 # Target mean luminance for optimal sky contrast (0-255)
        self.aec_thread = None

        # Boolean flag indicating whether the background threads have been started
        self.started = False
        # Generation counter: every reconnect creates a new generation. Old reader threads see the
        # mismatch, release their own (dead) capture handle and exit on their own.
        self._generation = 0
        # Deadline by which a new frame MUST arrive, otherwise the watchdog triggers a reconnect
        self._deadline = 0.0
        # Placeholder for the reader thread object
        self.thread = None
        # Placeholder for the watchdog thread object
        self.watchdog_thread = None

        # Open the camera for the first time
        self.cap = self._open_capture()

        # Attempt to read the very first frame to verify the camera is working and initialize variables
        if self.cap.isOpened():
            grabbed, frame = self.cap.read()
            if grabbed and frame is not None:
                self.frame = frame
                self.grabbed = True
                self.last_frame_time = time.time()
                self.connected = True
                self.status_text = "ONLINE"

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #
    def _open_capture(self):
        """Create and configure a brand new VideoCapture handle (always returns a VideoCapture object)."""
        # Use the DirectShow backend for faster initialization on Windows
        cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        if cap.isOpened():
            # FOURCC must be set BEFORE the resolution on DirectShow, otherwise it is ignored
            if self.use_mjpg:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            # Set the desired width for the camera frame output
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            # Set the desired height for the camera frame output
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            # Set the desired frame rate
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            # Restrict the buffer size to 1 frame so we always get the most recent frame (less lag)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # AUTO-EXPOSURE & GAIN INITIALIZATION:
            # Force DirectShow camera hardware to enable Auto Exposure and reset any manual overrides
            if self.auto_exposure:
                try:
                    # In DirectShow, 0.75 signals Auto Exposure
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                except Exception:
                    pass
                try:
                    # Reset high gain so it never starts washed out
                    cap.set(cv2.CAP_PROP_GAIN, 0.0)
                except Exception:
                    pass
        return cap

    def _is_black(self, frame):
        """Return True if the frame is (almost) completely black. Samples every 16th pixel for speed."""
        try:
            return int(frame[::16, ::16].max()) <= self.black_threshold
        except Exception:
            return False

    def _reader_loop(self, generation, cap):
        """Background loop that reads ONE specific capture handle until it dies or is superseded."""
        failed_reads = 0
        black_frames = 0
        reason = None

        # Keep reading while the camera is running AND this reader is still the current generation
        while self.started and generation == self._generation:
            # Grab the latest frame from the camera (this call can block / hang if the USB link died)
            grabbed, frame = cap.read()

            # If the watchdog already replaced us while we were blocked inside read(), quit quietly
            if generation != self._generation:
                break

            # Symptom 1: failed reads
            if not grabbed or frame is None:
                failed_reads += 1
                if failed_reads >= self.max_failed_reads:
                    reason = f"{failed_reads} bacaan gagal berturut-turut"
                    break
                # Sleep briefly to avoid maxing out CPU usage while looping empty-handed
                time.sleep(0.01)
                continue
            failed_reads = 0

            # Symptom 2: DirectShow keeps "succeeding" but delivers pure black frames after an unplug
            if self._is_black(frame):
                black_frames += 1
                if black_frames >= self.black_frame_limit:
                    reason = f"{black_frames} bingkai hitam berturut-turut"
                    break
                # Do NOT publish black frames, so the main loop shows the 'signal lost' screen instead
                continue
            black_frames = 0

            # Acquire the lock before updating the shared state to prevent the main thread from reading partial data
            with self.read_lock:
                self.frame = frame
                self.grabbed = True
                self.last_frame_time = time.time()
                self.connected = True
                self.status_text = "ONLINE"
                self._deadline = self.last_frame_time + self.stale_timeout
                self.reconnect_attempts = 0

        # Each reader releases ITS OWN handle (never released from another thread -> no crash)
        try:
            cap.release()
        except Exception:
            pass

        # If we died on our own (not stopped, not superseded), ask the watchdog to reconnect immediately
        if self.started and generation == self._generation and reason is not None:
            print(f"[KAMERA] Sambungan kamera (src={self.src}) terputus: {reason}. Menyambung semula...")
            with self.read_lock:
                self.connected = False
                self.grabbed = False
                self.status_text = "SIGNAL LOST"
                self._deadline = 0.0

    def _spawn_reader(self, cap):
        """Start a new reader thread for the given capture handle as a new generation."""
        self._generation += 1
        generation = self._generation
        self.cap = cap
        with self.read_lock:
            # Give the freshly opened camera some grace time to start streaming
            self._deadline = time.time() + self.open_grace
        self.thread = threading.Thread(target=self._reader_loop, args=(generation, cap), daemon=True)
        self.thread.start()

    def _reconnect(self):
        """Abandon the current (dead) handle and try to open a brand new one."""
        with self.read_lock:
            self.connected = False
            self.grabbed = False
            self.reconnect_attempts += 1
            attempt = self.reconnect_attempts
            self.status_text = f"RECONNECTING (#{attempt})"

        # Bump the generation so any old reader (even one stuck inside read()) will exit and release itself
        self._generation += 1

        if attempt == 1 or attempt % 5 == 0:
            print(f"[KAMERA] Cubaan sambung semula #{attempt} ke kamera src={self.src}...")

        # Open a NEW handle. After a USB re-enumeration this is the only way to get frames again.
        cap = self._open_capture()
        if cap.isOpened():
            self.reconnect_count += 1
            print(f"[KAMERA] Kamera src={self.src} dibuka semula (jumlah reconnect: {self.reconnect_count}).")
            self._spawn_reader(cap)
        else:
            # Device not back yet (Windows still re-enumerating). Try again after reconnect_delay.
            try:
                cap.release()
            except Exception:
                pass
            with self.read_lock:
                self._deadline = time.time() + self.reconnect_delay

    def _watchdog_loop(self):
        """Supervisor thread: triggers a reconnect whenever no fresh frame arrives before the deadline."""
        while self.started:
            time.sleep(0.2)
            if not self.started:
                break
            with self.read_lock:
                deadline = self._deadline
            # Symptom 3 (hung read) and recovery after symptoms 1 & 2 are all handled here
            if time.time() > deadline:
                try:
                    self._reconnect()
                except Exception as e:
                    print(f"[KAMERA] Ralat semasa menyambung semula: {e}")
                    with self.read_lock:
                        self._deadline = time.time() + self.reconnect_delay

    def _aec_worker_loop(self):
        """
        Supervisor Thread: Smart Field Auto-Exposure & Gain Control (AEC/AGC).
        Runs asynchronously every 0.35s so it never slows down the video capture thread.
        Monitors mean brightness and anti-sky blowout clipping so the drone is never lost
        against harsh outdoor field skies.
        """
        while self.started:
            time.sleep(0.35)
            if not self.started or not self.aec_enabled:
                continue

            with self.read_lock:
                if self.frame is None or not self.connected:
                    continue
                # Downsample frame for microsecond luminance evaluation
                sample = self.frame[::16, ::16]
                cap = self.cap

            if sample is None or cap is None:
                continue

            try:
                if not cap.isOpened():
                    continue

                gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
                lum = float(np.mean(gray))
                overexposed_ratio = float(np.mean(gray > 240))

                # ANTI-SKY BLOWOUT: If image is washed out or highlights are clipped
                if lum > (self.target_luminance + 25) or overexposed_ratio > 0.05:
                    if self.current_gain > 5.0:
                        self.current_gain = max(0.0, self.current_gain - 15.0)
                        cap.set(cv2.CAP_PROP_GAIN, self.current_gain)
                    elif self.current_exposure > -13.0:
                        self.current_exposure -= 1.0
                        cap.set(cv2.CAP_PROP_EXPOSURE, self.current_exposure)

                # UNDER-EXPOSURE RECOVERY: If image is too dark
                elif lum < (self.target_luminance - 25) and overexposed_ratio < 0.01:
                    if self.current_exposure < -4.0:
                        self.current_exposure += 1.0
                        cap.set(cv2.CAP_PROP_EXPOSURE, self.current_exposure)
                    elif self.current_gain < 50.0:
                        self.current_gain = min(60.0, self.current_gain + 10.0)
                        cap.set(cv2.CAP_PROP_GAIN, self.current_gain)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    #  Public API (unchanged from the original class)
    # ------------------------------------------------------------------ #
    def start(self):
        """Memulakan thread kamera di latar belakang."""
        # If the thread is already running, do nothing and return the instance
        if self.started:
            return self
        # Set the flag to True indicating the threads are active
        self.started = True

        if self.cap.isOpened():
            # Start reading from the handle that was opened in __init__
            self._spawn_reader(self.cap)
        else:
            # Camera not available at start: let the watchdog keep trying to open it
            with self.read_lock:
                self.status_text = "SIGNAL LOST"
                self._deadline = 0.0

        # Start the watchdog / auto-reconnect supervisor
        self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.watchdog_thread.start()

        # Start the smart field AEC (Auto-Exposure Control) supervisor
        self.aec_thread = threading.Thread(target=self._aec_worker_loop, daemon=True)
        self.aec_thread.start()

        # Return self so methods can be chained if desired
        return self

    def read(self):
        """Main Thread (AI) memanggil fungsi ini untuk dapatkan bingkai TERKINI.
        Returns (False, None) when the camera is disconnected or the frame is stale,
        so the caller never processes a frozen/old image as if it were live."""
        with self.read_lock:
            if self.frame is None or not self.connected:
                return False, None
            if time.time() - self.last_frame_time > self.stale_timeout:
                return False, None
            # Return a copy of the frame to prevent accidental modifications to the original buffer
            return True, self.frame.copy()

    def is_connected(self):
        """True if the camera is currently delivering fresh frames."""
        with self.read_lock:
            return self.connected and (time.time() - self.last_frame_time) <= self.stale_timeout

    def get_status(self):
        """Human-readable status string, e.g. 'ONLINE' or 'RECONNECTING (#3)'."""
        with self.read_lock:
            return self.status_text

    def step_exposure(self, delta):
        """Manually nudge exposure (+1 or -1 EV) and locks to manual mode."""
        self.aec_enabled = False
        self.current_exposure = max(-13.0, min(-1.0, self.current_exposure + delta))
        try:
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.current_exposure)
                print(f"[KAMERA] Pendedahan Manual: {int(self.current_exposure)} EV")
        except Exception as e:
            print(f"[KAMERA] Ralat pelarasan pendedahan: {e}")
        return self.current_exposure

    def toggle_auto_exposure(self):
        """Toggle between smart field Auto-Exposure (AEC) and manual lock."""
        self.aec_enabled = not self.aec_enabled
        status = "AKTIF (AEC)" if self.aec_enabled else "MANUAL"
        print(f"[KAMERA] Mod Pendedahan Auto: {status}")
        if self.aec_enabled:
            try:
                if self.cap and self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            except Exception:
                pass
        return self.aec_enabled

    def reset_auto_exposure(self):
        """Resets camera to native auto exposure and baseline outdoor gain."""
        self.aec_enabled = True
        self.current_exposure = -9.0
        self.current_gain = 0.0
        try:
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                self.cap.set(cv2.CAP_PROP_GAIN, 0.0)
                print("[KAMERA] Pendedahan di-reset kepada Auto & Gain Minima.")
        except Exception as e:
            print(f"[KAMERA] Ralat reset auto-exposure: {e}")

    def get_exposure_info(self):
        """Returns string for HUD display: e.g. 'EXP: AUTO [-9EV] G:0'."""
        mode = "AUTO" if self.aec_enabled else "MAN"
        exp_val = int(self.current_exposure) if self.current_exposure is not None else -9
        gain_val = int(self.current_gain) if self.current_gain is not None else 0
        return f"EXP:{mode}[{exp_val}EV] G:{gain_val}"

    def open_settings(self):
        """Opens native DirectShow camera properties dialog for manual exposure, ISO, focus control."""
        try:
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_SETTINGS, 1)
        except Exception as e:
            print(f"[KAMERA] Ralat membuka dialog tetapan: {e}")

    def stop(self):
        """Menghentikan thread dan membebaskan port kamera."""
        # Set the flag to False, which will break the loops inside the reader, watchdog, and AEC threads
        self.started = False
        # Invalidate the current generation so the reader exits and releases its own handle
        self._generation += 1
        # Wait briefly for the watchdog and AEC threads to finish
        if self.watchdog_thread is not None and self.watchdog_thread.is_alive():
            self.watchdog_thread.join(timeout=1.0)
        if self.aec_thread is not None and self.aec_thread.is_alive():
            self.aec_thread.join(timeout=1.0)
        # Wait for up to 1 second for the reader thread to finish cleanly
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        # If no reader ever owned the handle (or it already finished), make sure the port is released.
        # If the reader is still stuck inside a hung read(), do NOT release from here (could crash OpenCV).
        if self.thread is None or not self.thread.is_alive():
            try:
                if self.cap.isOpened():
                    self.cap.release()
            except Exception:
                pass