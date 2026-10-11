# Import OpenCV library for image processing and displaying the video feed
import cv2
# Import socket module to send UDP network packets to the ESP32 microcontroller
import socket
# Import time module to manage delays and frame timings
import time
# Import os module for interacting with the operating system, like file paths
import os
# Import pygame to handle joystick (Xbox controller) inputs and events
import pygame
# Import XInput specifically to control the rumble/vibration motors on the Xbox controller
import XInput
# Import sys module for system-specific parameters and functions, used for restarting the script
import sys
# Import importlib to allow dynamic reloading of python modules without restarting the whole program
import importlib
# Import the custom TurretVision class that handles AI detection logic
from core.ai_vision import TurretVision
# Import the custom HeadUpDisplay class that handles UI rendering on the screen
from ui.hud_display import HeadUpDisplay
# Import the hud_display module entirely to allow for dynamic reloading
import ui.hud_display
# Import numpy for numerical and array operations
import numpy as np
# Import the custom ThreadedCamera class to capture video frames asynchronously without blocking
from core.camera_stream import ThreadedCamera

def extract_and_select_target(boxes, center_x_screen, center_y_screen, locked_id=None):
    """
    Polymorphic parser and target selector (DJI ActiveTrack logic).
    Supports:
    - List of [x1, y1, x2, y2]
    - List of [x1, y1, x2, y2, track_id, conf, class_name]
    - NumPy ndarray
    - PyTorch Tensor
    - Ultralytics Boxes object (.xyxy, .id, .conf)
    
    Returns:
    - best_target: dict with 'x1', 'y1', 'x2', 'y2', 'cx', 'cy', 'track_id', 'conf' (or None)
    - all_targets: list of all parsed target dicts
    """
    if boxes is None or len(boxes) == 0:
        return None, []

    parsed_targets = []

    # Case 1: Ultralytics YOLO Boxes object
    if hasattr(boxes, 'xyxy'):
        try:
            tensor_boxes = boxes.xyxy.cpu().numpy()
            ids = boxes.id.int().cpu().tolist() if (hasattr(boxes, 'id') and boxes.id is not None) else [None] * len(tensor_boxes)
            confs = boxes.conf.cpu().numpy() if (hasattr(boxes, 'conf') and boxes.conf is not None) else [1.0] * len(tensor_boxes)
            for b, tid, c in zip(tensor_boxes, ids, confs):
                x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                parsed_targets.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cx": (x1 + x2) / 2.0,
                    "cy": (y1 + y2) / 2.0,
                    "track_id": tid,
                    "conf": float(c)
                })
        except Exception as e:
            print(f"[TRACKER ERROR] Failed parsing Ultralytics Boxes: {e}")

    # Case 2: List, Tuple, or NumPy Array
    elif isinstance(boxes, (list, tuple, np.ndarray)):
        for item in boxes:
            try:
                # If item is tensor or ndarray, convert
                if hasattr(item, 'cpu'):
                    item = item.cpu().numpy()
                if hasattr(item, 'tolist'):
                    item = item.tolist()
                
                if len(item) >= 4:
                    x1, y1, x2, y2 = float(item[0]), float(item[1]), float(item[2]), float(item[3])
                    tid = int(item[4]) if len(item) > 4 and item[4] is not None else None
                    conf = float(item[5]) if len(item) > 5 and item[5] is not None else 1.0
                    parsed_targets.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "cx": (x1 + x2) / 2.0,
                        "cy": (y1 + y2) / 2.0,
                        "track_id": tid,
                        "conf": conf
                    })
            except Exception as e:
                continue

    if not parsed_targets:
        return None, []

    # DJI ActiveTrack Target Selection Logic:
    # 1. If currently locked onto a target_id and it still exists in this frame, KEEP TRACKING IT!
    if locked_id is not None:
        for target in parsed_targets:
            if target.get("track_id") is not None and target["track_id"] == locked_id:
                return target, parsed_targets

    # 2. Otherwise (or target was lost), select the target CLOSEST to the screen center crosshair!
    # This prevents false positives at the edges/corners (e.g. clothes hangers) from stealing focus.
    best_target = min(
        parsed_targets,
        key=lambda t: (t["cx"] - center_x_screen) ** 2 + (t["cy"] - center_y_screen) ** 2
    )

    return best_target, parsed_targets

def main():
    """
    Main function that initializes hardware, loads models, connects to peripherals, 
    and runs the primary application loop.
    """
    # IP address of the ESP32 acting as a Wi-Fi Access Point (AP)
    UDP_IP = "192.168.4.1"
    # Port number configured on the ESP32 to receive UDP packets
    UDP_PORT = 1234
    # Create a UDP socket (IPv4, Datagram)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Initialize all imported pygame modules
    pygame.init()
    # Initialize the pygame joystick module to read controller inputs
    pygame.joystick.init()
    # Initialize joystick variable as None
    joystick = None
    
    # Check if at least one joystick/controller is connected
    if pygame.joystick.get_count() > 0:
        # Create a Joystick object for the first detected controller (index 0)
        joystick = pygame.joystick.Joystick(0)
        # Initialize this specific joystick
        joystick.init()
        # Print a system message confirming the controller was detected
        print(f"[SISTEM] Joystick Dikesan: {joystick.get_name()}")
    else:
        # Print a warning if no controller is found
        print("[SISTEM] Tiada Joystick Dikesan. Kawalan mungkin terhad kepada keyboard/mouse.")

    # Speed multiplier for manual panning/tilting via the Xbox controller analog stick
    xbox_speed = 10
    # Sensitivity multiplier for manual panning/tilting via the computer mouse
    mouse_sensitivity = 3     
    # Deadzone threshold to ignore tiny, accidental analog stick movements (stick drift)
    DEADZONE = 0.1
    
    # Track the last sent pan and tilt angles to avoid sending duplicate UDP packets
    last_sent_pan, last_sent_tilt = -1, -1
    # Track the last sent ammo count to trigger updates when ammo changes
    last_sent_ammo = 1000
    # Track the last sent drone detection state to alert the ESP32 when a target appears/disappears
    last_sent_drone = 0
    # Track the timestamp of the last UDP packet sent to prevent flooding the network
    last_udp_time = 0

    # Get the directory where this script (main_system.py) is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path to the YOLO model file (.pt) inside the 'models' folder
    model_path = os.path.join(current_dir, 'models', 'best (2).pt')

    # Check if the AI model file actually exists at the constructed path
    if not os.path.exists(model_path):
        # Print a critical error message and exit the program if the model is missing
        print(f"[Ralat Kritikal] Otak AI tidak dijumpai di: {model_path}")
        return 

    # Initialize the TurretVision object, loading the AI model into memory
    ai_module = TurretVision(model_path)
    # Initialize the HeadUpDisplay object, setting the maximum ammo capacity to 1000
    hud_module = HeadUpDisplay(max_ammo=1000)

    # Attempt to initialize the ThreadedCamera using an external camera (src=1), requesting Full HD 1920x1080 resolution
    print("[SISTEM] Mencuba kamera luaran Full HD 1080p (src=1)...")
    cap = ThreadedCamera(src=1, width=1920, height=1080)
    
    # If the external camera could not be opened
    if not cap.grabbed:
        print("[SISTEM] Kamera luaran gagal. Mencuba kamera laptop (src=0)...")
        # Release the failed camera resource
        cap.cap.release()
        # Fallback to the built-in laptop webcam (src=0) with 1080p resolution
        cap = ThreadedCamera(src=0, width=1920, height=1080)
    
    # Start the background thread that continuously reads frames from the camera
    cap.start()
    # Pause for 1 second to give the camera sensor time to warm up and adjust exposure/white balance
    time.sleep(1.0)

    # Define the title of the display window
    window_name = "RENTAP-1 Prototype"
    # Create an OpenCV window with the WINDOW_NORMAL flag, allowing it to be resized
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Set the initial window size to 1920x1080
    cv2.resizeWindow(window_name, 1920, 1080) 
    # Flag to track whether the window is currently in fullscreen mode
    is_fullscreen = False

    # Dictionary containing all state variables for camera control, zoom, and autonomous tracking
    state = {
        # Current zoom multiplier
        "zoom_level" : 1.0,
        # Target zoom multiplier (useful for smooth zooming transitions if implemented later)
        "target_zoom": 1.0,
        # Available discrete zoom levels that can be toggled through
        "zoom_steps" : [1.0, 2.0, 4.0, 8.0, 15.0],
        # Current index within the zoom_steps array
        "step_idx"   : 0,
        # Current pan angle of the turret (0 to 180 degrees, 90 is center)
        "current_pan" : 90.0,
        # Current tilt angle of the turret (0 to 180 degrees, 90 is center)
        "current_tilt" : 90.0,
        # Last known X coordinate of the mouse cursor, used to calculate relative movement
        "last_mouse_x": 640,
        # Last known Y coordinate of the mouse cursor
        "last_mouse_y": 360,
        # Previous zoom level, used to detect when zoom changes to play a sound effect
        "last_zoom_level": 1.0,
        # Flag indicating if the system is autonomously tracking targets (True) or in manual control (False)
        "autonomous_mode": False,
        # Direction of sweeping when in autonomous mode and no target is found (1 = right, -1 = left)
        "search_direction": 1,
        # Speed at which the turret sweeps while searching for targets
        "search_speed": 1.0,
        # Proportional gain (Kp) for the pan axis PID controller, dictates tracking speed/aggressiveness horizontally
        "pan_kp": 0.025,
        # Proportional gain (Kp) for the tilt axis PID controller, dictates tracking speed/aggressiveness vertically
        "tilt_kp": 0.025,
        # Pixel radius around the screen center with quadratic soft-damping for precision centering
        "tracking_deadzone": 20,
        # Locked target ID for persistence tracking (DJI ActiveTrack style)
        "locked_target_id": None,
        # Counter for consecutive frames where locked target was not seen
        "target_lost_count": 0,
        # Max frames to remember locked target before un-locking (30 frames ~ 1.0 sec coasting)
        "target_lost_threshold": 30,
        # Smooth gimbal step filters (DJI-Style Exponential Smoothing)
        "pan_step_smooth": 0.0,
        "tilt_step_smooth": 0.0
    }

    def mouse_controller(event, x, y, flags, param):
        """
        Callback function executed by OpenCV whenever a mouse event occurs in the window.
        """
        # Declare 'state' as nonlocal so we can modify the dictionary from the outer scope
        nonlocal state
        
        # If the mouse was moved
        if event == cv2.EVENT_MOUSEMOVE: 
            # Only allow mouse aiming if autonomous mode is OFF
            if not state["autonomous_mode"]:
                # Calculate the difference in X position since the last frame
                dx = x - state["last_mouse_x"]
                # Calculate the difference in Y position since the last frame
                dy = y - state["last_mouse_y"]
                
                # Update current pan angle based on mouse movement and sensitivity
                state["current_pan"] += -(dx * mouse_sensitivity)
                # Update current tilt angle based on mouse movement and sensitivity
                state["current_tilt"] -= (dy * mouse_sensitivity)
            
            # Update the last known mouse coordinates for the next frame
            state["last_mouse_x"] = x
            state["last_mouse_y"] = y
        
        # If the left mouse button is pressed down
        if event == cv2.EVENT_LBUTTONDOWN:  
            # Trigger the firing mechanism in the HUD module
            hud_module.start_firing()
        # If the left mouse button is released
        elif event == cv2.EVENT_LBUTTONUP:
            # Stop the firing mechanism
            hud_module.stop_firing()
        # If the right mouse button is clicked or double-clicked
        elif event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_RBUTTONDBLCLK:
            # Cycle through the predefined zoom steps (0 -> 1 -> 2 -> 3 -> 4 -> 0...)
            state["step_idx"] = (state["step_idx"] + 1) % len(state["zoom_steps"])
            # Apply the new zoom level
            state["zoom_level"] = state["zoom_steps"][state["step_idx"]]
        # If the mouse scroll wheel is rotated
        elif event == cv2.EVENT_MOUSEWHEEL:
            # If scrolled forward (zoom in)
            if flags > 0:
                # Increase zoom by 0.5, capping at a maximum of 10.0x
                state["zoom_level"] = min(10.0, state["zoom_level"] + 0.5)
            # If scrolled backward (zoom out)
            else:
                # Decrease zoom by 0.5, capping at a minimum of 1.0x (no zoom)
                state["zoom_level"] = max(1.0, state["zoom_level"] - 0.5)

    # Register the mouse_controller function to handle mouse events for the main display window
    cv2.setMouseCallback(window_name, mouse_controller)
    
    # Variable to track when the controller vibration (rumble) should stop
    rumble_end_time = 0
    # Flag to indicate if the controller is currently vibrating
    is_rumbling = False
    
    print("[SISTEM] RENTAP Dalam Talian. Tekan 'Q' untuk keluar, 'F' untuk tembak, 'R' untuk reload.")

    # Hantar arahan reset ke ESP32 pada permulaan sistem
    try:
        sock.sendto(b"90,90,0,0,1000,0", (UDP_IP, UDP_PORT))
    except OSError:
        pass

    # Last known frame size, used to build the 'signal lost' placeholder screen with the same dimensions
    last_frame_shape = (1080, 1920, 3)

    # FPS Calculation tracking
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 30.0

    # Main Application Loop
    while True:
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
        # Retrieve the latest frame from the threaded camera object
        success, frame = cap.read()
        # True only when the camera delivered a FRESH, valid frame
        camera_ok = success and frame is not None

        # CAMERA RECOVERY FIX:
        # Previously the loop did 'continue' here. That skipped cv2.imshow()/cv2.waitKey(), pygame events
        # and UDP, so the window froze ('Not Responding' / black) and the CPU spun at 100% while the camera
        # was gone. Now we keep the whole loop alive with a placeholder frame while the camera thread
        # reconnects in the background.
        if camera_ok:
            last_frame_shape = frame.shape
        else:
            # Build a dark placeholder frame of the same size as the real camera feed
            frame = np.zeros(last_frame_shape, dtype=np.uint8)
            # Small sleep so the loop doesn't spin at 100% CPU while waiting for the camera to come back
            time.sleep(0.03)

        # If an Xbox controller is connected
        if joystick:
            # Process all events in the pygame event queue
            for event in pygame.event.get():
                # If a button on the controller was pressed down
                if event.type == pygame.JOYBUTTONDOWN:
                    # Button 2 (usually 'X' on Xbox) triggers weapon reload
                    if event.button == 2:
                        hud_module.reload_weapon()
                    # Button 3 (usually 'Y' on Xbox) resets pan and tilt to center (90 degrees)
                    elif event.button == 3:
                        state["current_pan"] = 90
                        state["current_tilt"] = 90
                    # Button 5 (usually Right Bumper / RB) zooms IN
                    elif event.button == 5:
                        # Increment zoom index, ensuring it doesn't exceed the array bounds
                        state["step_idx"] = min(len(state["zoom_steps"]) - 1, state["step_idx"] + 1)
                        state["zoom_level"] = state["zoom_steps"][state["step_idx"]]
                    # Button 4 (usually Left Bumper / LB) zooms OUT
                    elif event.button == 4:  
                        # Decrement zoom index, ensuring it doesn't drop below 0
                        state["step_idx"] = max(0, state["step_idx"] - 1)
                        state["zoom_level"] = state["zoom_steps"][state["step_idx"]]
                    # Button 1 (usually 'B' on Xbox) exits the program
                    elif event.button == 1:
                        # Reset servos to center before exiting
                        state["current_pan"] = 90
                        state["current_tilt"] = 90
                        try:
                            # Hantar arahan reset terus ke ESP32
                            msg = f"90,90,0,0,{hud_module.ammo},0"
                            sock.sendto(msg.encode(), (UDP_IP, UDP_PORT))
                        except OSError:
                            pass
                        # Stop the camera thread
                        cap.stop()
                        # Close the OpenCV display window
                        cv2.destroyAllWindows() 
                        # Exit the loop (and thus the program)
                        sys.exit() 

                # If the D-Pad was pressed
                elif event.type == pygame.JOYHATMOTION:
                    # Get the tuple representing D-Pad state (x, y)
                    hat = joystick.get_hat(0)
                    # If D-Pad UP was pressed
                    if hat[0] == 1:
                        # Toggle firing mode between FULL_AUTO and BURST
                        if hud_module.fire_mode == "FULL_AUTO":
                            hud_module.fire_mode = "BURST"
                            hud_module.mode_sound.play()  
                        else:
                            hud_module.fire_mode = "FULL_AUTO"
                            hud_module.mode_sound.play() 
                    
                    # If D-Pad DOWN was pressed
                    if hat[1] == -1: 
                        # Toggle the autonomous tracking mode on/off
                        state["autonomous_mode"] = not state["autonomous_mode"]
                        # Determine string representation of the mode for logging
                        mode_status = "AKTIF" if state["autonomous_mode"] else "NYAHAKTIF"
                        print(f"[SISTEM] Mod Autonomi: {mode_status}")
                        # If turning ON autonomous mode, reset servos to center first
                        if state["autonomous_mode"]:
                            state["current_pan"] = 90.0
                            state["current_tilt"] = 90.0
                            state["locked_target_id"] = None
                            state["target_lost_count"] = 0

            # If manual control is active (autonomous mode is OFF)
            if not state["autonomous_mode"]:
                # Read the Right Analog Stick vertical axis (Tilt)
                tilt_axis = joystick.get_axis(3)
                # Read the Right Analog Stick horizontal axis (Pan)
                pan_axis = joystick.get_axis(2)

                # Apply deadzone filtering. If the stick is pushed past the deadzone threshold:
                if abs(pan_axis) > DEADZONE:
                    # Adjust the pan angle. Negative sign inverts the axis if needed
                    state["current_pan"] += (-pan_axis) * xbox_speed
                if abs(tilt_axis) > DEADZONE:
                    # Adjust the tilt angle
                    state["current_tilt"] += (-tilt_axis) * xbox_speed

            # Read the Right Trigger (RT) axis. Values range from -1.0 (unpressed) to 1.0 (fully pressed)
            rt_axis = joystick.get_axis(5)
            # If RT is pressed more than halfway
            if rt_axis > 0.5:
                hud_module.start_firing()
            # If RT is released AND the left mouse button is also NOT pressed
            elif rt_axis < 0.5 and not pygame.mouse.get_pressed()[0]: 
                hud_module.stop_firing()  
            
        # Clamp the pan angle between 0 and 180 degrees to prevent exceeding servo limits
        state["current_pan"] = max(0, min(180, state["current_pan"]))
        # Clamp the tilt angle between 0 and 180 degrees
        state["current_tilt"] = max(0, min(180, state["current_tilt"]))

        # If the zoom level has changed since the last frame
        if state["zoom_level"] != state["last_zoom_level"]:
            # Play the sound effect corresponding to the new zoom level
            hud_module.play_zoom_sound(state["zoom_level"])
            # Update the tracked last zoom level
            state["last_zoom_level"] = state["zoom_level"]

        if camera_ok:
            # Apply digital zoom to the raw camera frame
            zoomed_frame = hud_module.apply_zoom(frame, state["zoom_level"])
            # Pass the zoomed frame to the AI brain for object detection and tracking
            frame_ai, boxes = ai_module.scan_frame(zoomed_frame)
        else:
            # No camera: don't feed the black placeholder to the AI/tracker (it would only waste GPU time
            # and confuse the tracker's camera-motion compensation)
            frame_ai, boxes = frame, []
        
        # Update the HUD's firing logic (deducts ammo, plays sounds, calculates delays)
        # Returns True if a shot was fired EXACTLY on this frame update
        update_fire = hud_module.update_firing()
        
        # Create an integer flag representing if the laser/firing is active (1) or not (0)
        laser_flag = 1 if update_fire else 0
        # Create an integer flag representing if the AI detected at least one target (1) or none (0)
        # Dynamically get frame dimensions to calculate the true screen center
        frame_h, frame_w = frame.shape[:2]
        center_x_screen = frame_w / 2.0
        center_y_screen = frame_h / 2.0

        # Universal extraction & target selection (DJI ActiveTrack logic)
        target, all_targets = extract_and_select_target(
            boxes, center_x_screen, center_y_screen, locked_id=state["locked_target_id"]
        )

        # Flag representing if at least one target is detected
        drone_detected = 1 if target is not None else 0

        # If the turret is in autonomous mode but the camera is DISCONNECTED:
        # HOLD the current position. Sweeping blindly keeps moving the servos, which keeps flexing the
        # USB-C cable and can cause repeated disconnects while the camera is trying to reconnect.
        if state["autonomous_mode"] and not camera_ok:
            state["locked_target_id"] = None
            state["target_lost_count"] = 0

        # If the turret is in fully autonomous mode (and the camera is alive)
        elif state["autonomous_mode"]:
            if drone_detected:
                # Update locked target ID and reset lost counter
                if target.get("track_id") is not None:
                    state["locked_target_id"] = target["track_id"]
                state["target_lost_count"] = 0

                cx = target["cx"]
                cy = target["cy"]

                # Calculate error between target center and screen crosshair center
                error_x = cx - center_x_screen
                error_y = cy - center_y_screen

                # --- DJI-STYLE SMOOTH GIMBAL TRACKING WITH SOFT-DAMPING ---
                # 1. Limit the maximum servo speed per frame to prevent violent overshoot/jitter/motion blur
                MAX_STEP = 2.0 
                
                # 2. Direction Multipliers: If the turret moves AWAY from the drone instead of towards it, 
                # change 1.0 to -1.0 for the axis that moves in the wrong direction.
                PAN_DIR = 1.0
                TILT_DIR = 1.0

                # Soft quadratic damping: glides into the exact crosshairs without jerky stops
                deadzone = state.get("tracking_deadzone", 20)
                damp_x = ((abs(error_x) / deadzone) ** 2.0) if abs(error_x) < deadzone else 1.0
                damp_y = ((abs(error_y) / deadzone) ** 2.0) if abs(error_y) < deadzone else 1.0

                # Horizontal tracking (Pan): Move turret to center target with smooth inertia
                target_pan_adjust = error_x * state["pan_kp"] * PAN_DIR * damp_x
                target_pan_adjust = max(-MAX_STEP, min(MAX_STEP, target_pan_adjust))
                state["pan_step_smooth"] = 0.65 * state["pan_step_smooth"] + 0.35 * target_pan_adjust
                state["current_pan"] -= state["pan_step_smooth"]

                # Vertical tracking (Tilt): Move turret to center target with smooth inertia
                target_tilt_adjust = error_y * state["tilt_kp"] * TILT_DIR * damp_y
                target_tilt_adjust = max(-MAX_STEP, min(MAX_STEP, target_tilt_adjust))
                state["tilt_step_smooth"] = 0.65 * state["tilt_step_smooth"] + 0.35 * target_tilt_adjust
                state["current_tilt"] -= state["tilt_step_smooth"]

                # Visual Feedback: Mark primary locked target with tactical crosshair reticle
                tcx, tcy = int(target["cx"]), int(target["cy"])
                cv2.drawMarker(frame_ai, (tcx, tcy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                tid_label = f"LOCK [ID:{target['track_id']}]" if target.get("track_id") is not None else "LOCKED TARGET"
                cv2.putText(frame_ai, tid_label, (tcx + 12, tcy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            else:
                # Target was not detected in this frame
                state["target_lost_count"] += 1
                state["pan_step_smooth"] = 0.0
                state["tilt_step_smooth"] = 0.0

                # DJI ActiveTrack Coast / Hold Policy:
                # If the target was missed momentarily (e.g. slight blur or turn), HOLD POSITION!
                # Do NOT start sweeping away until target has been absent for the full threshold (30 frames ~ 1 sec).
                if state["target_lost_count"] > state["target_lost_threshold"]:
                    state["locked_target_id"] = None

                    # Sweep Mode: Slowly move the pan axis left and right to search for targets
                    state["current_pan"] += (state["search_speed"] * state["search_direction"])
                    state["current_tilt"] = 90.0 
                    
                    # Sweep boundary bounce
                    if state["current_pan"] >= 180.0:
                        state["current_pan"] = 180.0
                        state["search_direction"] = -1
                    elif state["current_pan"] <= 0.0:
                        state["current_pan"] = 0.0
                        state["search_direction"] = 1

        # Integer flag indicating whether the weapon is in BURST mode (1) or FULL_AUTO (0)
        fire_mode_flag = 1 if hud_module.fire_mode == "BURST" else 0

        # CRITICAL FIX: Ensure the final pan/tilt values are strictly clamped between 0 and 180 degrees 
        # AFTER all autonomous tracking and before sending out to the ESP32.
        # Sending negative or >180 values causes servo failure or lockups.
        state["current_pan"] = max(0.0, min(180.0, state["current_pan"]))
        state["current_tilt"] = max(0.0, min(180.0, state["current_tilt"]))

        # Cast the continuous float angles to integers required by the ESP32 servo controller
        p_int, t_int = int(state["current_pan"]), int(state["current_tilt"])
        
        # Get current time for rate limiting UDP packets
        current_time = time.time()
        # Get current ammo count
        current_ammo = hud_module.ammo
        
        # Determine if the turret needs to physically move based on new coordinate calculations
        is_moving_event = (p_int != last_sent_pan or t_int != last_sent_tilt)
        # Determine if critical state data (ammo count or target detection status) has changed
        is_state_changed = (current_ammo != last_sent_ammo) or (drone_detected != last_sent_drone)

        # Send a UDP packet to the ESP32 IF:
        # 1. A shot was fired (laser_flag == 1) OR
        # 2. Ammo or drone detection status changed OR
        # 3. Turret needs to move AND at least 0.05 seconds (50ms) have passed since the last packet (Rate Limiting)
        if laser_flag == 1 or is_state_changed or (is_moving_event and (current_time - last_udp_time > 0.05)):
            # Ensure we are sending the most up-to-date ammo count
            current_ammo = hud_module.ammo
            # Construct the comma-separated message payload string
            msg = f"{p_int},{t_int},{laser_flag},{drone_detected},{current_ammo},{fire_mode_flag}"
            
            try:
                # Encode the string to bytes and send it over UDP to the specified IP and Port
                sock.sendto(msg.encode(), (UDP_IP, UDP_PORT))
            except OSError:
                # Ignore OS errors (like WinError 10051) which happen when the computer is not connected to the ESP32 Wi-Fi network
                pass 
            
            # Update tracking variables with the data just sent
            last_sent_pan, last_sent_tilt = p_int, t_int
            last_sent_ammo = current_ammo
            last_sent_drone = drone_detected
            last_udp_time = current_time

        # If a shot was fired during this frame
        if update_fire:
            # Check if XInput detects a connected Xbox controller
            if XInput.get_connected()[0]:
                # Trigger maximum vibration on both left (low frequency) and right (high frequency) motors
                XInput.set_vibration(0, 65535, 65535)  
                # Set a timer to stop the vibration 0.08 seconds (80ms) from now
                rumble_end_time = time.time() + 0.08
                # Mark that the controller is currently vibrating
                is_rumbling = True
        
        # If the controller is vibrating AND the timer has expired
        if is_rumbling and time.time() > rumble_end_time:
            if XInput.get_connected()[0]:
                # Send 0 speed to both motors to stop vibration
                XInput.set_vibration(0, left_speed=0.0, right_speed=0.0)
            # Mark that vibration has stopped
            is_rumbling = False

        # Have the HUD module draw the UI overlays (crosshair, text) on top of the AI annotated frame
        final_frame = hud_module.draw_hud(frame_ai, state["zoom_level"])

        # If the camera is disconnected, draw a clear warning on top of the HUD
        if not camera_ok:
            fh, fw = final_frame.shape[:2]
            warn_1 = "CAMERA SIGNAL LOST"
            warn_2 = f"AUTO-RECONNECTING... [{cap.get_status()}]" if hasattr(cap, "get_status") else "AUTO-RECONNECTING..."
            cv2.putText(final_frame, warn_1, (int(fw * 0.5) - 210, int(fh * 0.30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(final_frame, warn_2, (int(fw * 0.5) - 260, int(fh * 0.30) + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        # Render real-time FPS and Camera Exposure overlay in the top-left corner
        exp_info = cap.get_exposure_info() if hasattr(cap, "get_exposure_info") else ""
        cv2.putText(final_frame, f"FPS: {current_fps:.1f} | {exp_info}", (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Render the final composed image to the OpenCV window
        cv2.imshow(window_name, final_frame)

        # Read the keyboard input. waitKey(1) pauses for 1ms and returns the ASCII value of the key pressed
        key = cv2.waitKey(1) & 0xFF
        # If 'q' is pressed, break the loop to exit the program
        if key == ord('q'):
            break
        # If '[' is pressed, decrease exposure (-1 EV) (darken sky)
        elif key == ord('['):
            if hasattr(cap, "step_exposure"):
                cap.step_exposure(-1.0)
        # If ']' is pressed, increase exposure (+1 EV) (brighten target)
        elif key == ord(']'):
            if hasattr(cap, "step_exposure"):
                cap.step_exposure(1.0)
        # If 'o' is pressed, toggle smart field Auto-Exposure (AEC on/off)
        elif key == ord('o'):
            if hasattr(cap, "toggle_auto_exposure"):
                cap.toggle_auto_exposure()
        # If '0' (zero) is pressed, reset exposure to default Auto & zero gain
        elif key == ord('0'):
            if hasattr(cap, "reset_auto_exposure"):
                cap.reset_auto_exposure()
        # If 'c' is pressed, open the camera hardware properties dialog (shutter, exposure, ISO, focus)
        elif key == ord('c'):
            if hasattr(cap, "open_settings"):
                print("[SISTEM] Membuka dialog tetapan kamera (Exposure / Shutter)...")
                cap.open_settings()
        # If 'r' is pressed, trigger the weapon reload function
        elif key == ord('r'):  
            hud_module.reload_weapon()
        # If 'f' is pressed, toggle fullscreen mode
        elif key == ord('f'): 
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                # Set OpenCV window property to fullscreen
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                # Set OpenCV window property back to normal resizable window
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        # If 't' is pressed, toggle autonomous tracking mode via keyboard
        elif key == ord('t'):
            state["autonomous_mode"] = not state["autonomous_mode"]
            mode_status = "AKTIF" if state["autonomous_mode"] else "NYAHAKTIF"
            print(f"[SISTEM] Mod Autonomi (Papan Kekunci 'T'): {mode_status}")
            if state["autonomous_mode"]:
                state["current_pan"] = 90.0
                state["current_tilt"] = 90.0
                state["locked_target_id"] = None
                state["target_lost_count"] = 0
        # If 'u' is pressed, dynamically reload the UI code without restarting the whole program
        elif key == ord('u'): 
            print("[SISTEM] Memuatkan semula (Reloading) fail UI...")
            try:
                # Use importlib to reload the hud_display module from disk
                importlib.reload(ui.hud_display)
                # Temporarily save current ammo state
                current_ammo = hud_module.ammo 
                # Re-instantiate the HeadUpDisplay object with the newly loaded code
                hud_module = ui.hud_display.HeadUpDisplay(max_ammo=1000)
                # Restore the ammo state
                hud_module.ammo = current_ammo
                print("[SISTEM] UI berjaya di-reload!")
            except Exception as e:
                # Log any syntax errors or issues in the reloaded file
                print(f"[RALAT] Gagal reload UI: {e}")
        # If 'p' is pressed, hard-restart the entire Python script
        elif key == ord('p'):
            # Reset servos to center before restarting
            try:
                msg = f"90,90,0,0,{hud_module.ammo},0"
                sock.sendto(msg.encode(), (UDP_IP, UDP_PORT))
            except OSError:
                pass
            # Stop the camera capture thread safely
            cap.stop()
            # Destroy the OpenCV window
            cv2.destroyAllWindows()
            # Shut down pygame components
            pygame.quit()
            # Import subprocess to launch a new process
            import subprocess
            # Launch a new instance of this exact script
            subprocess.Popen(
                [sys.executable, r"D:\MEDIA\RENTAP 1 - Workspace\main_system.py"], 
                creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NO_WINDOW
            )
            # Exit the current process
            sys.exit()

    # Before exiting the program entirely, reset the pan and tilt to center
    state["current_pan"] = 90
    state["current_tilt"] = 90
    try:
        # Hantar arahan reset terus ke ESP32
        msg = f"90,90,0,0,{hud_module.ammo},0"
        sock.sendto(msg.encode(), (UDP_IP, UDP_PORT))
    except OSError:
        pass
    
    # Stop the background camera thread
    cap.stop()
    # Close all OpenCV display windows
    cv2.destroyAllWindows()
    
# Standard Python boilerplate to ensure main() runs when the script is executed directly
if __name__ == "__main__":
    main()