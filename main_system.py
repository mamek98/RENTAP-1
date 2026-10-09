import cv2
import socket
import time
import os
import pygame
import XInput
import sys
from core.ai_vision import TurretVision
from ui.hud_display import HeadUpDisplay
import numpy as np
from core.camera_stream import ThreadedCamera

def main():

    UDP_IP = "192.168.4.1"
    UDP_PORT = 1234
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    pygame.init()
    pygame.joystick.init()
    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"[SISTEM] Joystick Dikesan: {joystick.get_name()}")
    else:
        print("[SISTEM] Tiada Joystick Dikesan. Kawalan mungkin terhad kepada keyboard/mouse.")

    xbox_speed = 10
    mouse_sensitivity=3     
    DEADZONE = 0.1
    last_sent_pan , last_sent_tilt = -1,-1

    last_sent_ammo = 1000
    last_sent_drone = 0
    
    last_udp_time = 0

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir,'models','best (2).pt')

    if not os.path.exists(model_path):
        print(f"[Ralat Kritikal] Otak AI tidak dijumpai di: {model_path}")
        return 

    ai_module = TurretVision(model_path)
    hud_module = HeadUpDisplay(max_ammo=1000)

    # Resolution set to 1080x720. Center is 540x360.
    cap = ThreadedCamera(src=1 , width=1080,height=720).start()
    time.sleep(1.0)

    window_name = "RENTAP-1 Prototype"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name,1080,720)

    # 1. NEW STATE VARIABLES ADDED FOR AUTONOMOUS TRACKING
    state = {
        "zoom_level" : 1.0,
        "target_zoom": 1.0,
        "zoom_steps" : [1.0,2.0,4.0,8.0,15.0],
        "step_idx"   : 0,
        "current_pan" : 90.0,
        "current_tilt" : 90.0,
        "last_mouse_x":480,
        "last_mouse_y":360,
        "last_zoom_level":1.0,
        "autonomous_mode": False,
        "search_direction": 1,
        "search_speed": 1.0,
        "pan_kp": 0.04,
        "tilt_kp": 0.04,
        "tracking_deadzone": 30
    }

    def mouse_controller(event, x, y, flags, param):
        nonlocal state
        if event == cv2.EVENT_MOUSEMOVE: 
            # 2. DISABLE MOUSE AIMING WHEN AUTONOMOUS MODE IS ON
            if not state["autonomous_mode"]:
                dx = x - state["last_mouse_x"]
                dy = y - state["last_mouse_y"]
                state["current_pan"] += -(dx * mouse_sensitivity )
                state["current_tilt"] -= dy * mouse_sensitivity
            
            state["last_mouse_x"] = x
            state["last_mouse_y"] = y
        
        if event == cv2.EVENT_LBUTTONDOWN:  
            hud_module.start_firing()
        elif event == cv2.EVENT_LBUTTONUP:
            hud_module.stop_firing()
        elif event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_RBUTTONDBLCLK:
            state["step_idx"] = (state["step_idx"] + 1) % len(state["zoom_steps"])
            state["zoom_level"] = state["zoom_steps"][state["step_idx"]]
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                state["zoom_level"] = min(10.0, state["zoom_level"] + 0.5)
            else:
                state["zoom_level"] = max(1.0, state["zoom_level"] - 0.5)

    cv2.setMouseCallback(window_name, mouse_controller)
    
    rumble_end_time = 0
    is_rumbling = False
    print("[SISTEM] RENTAP Dalam Talian. Tekan 'Q' untuk keluar, 'F' untuk tembak, 'R' untuk reload.")

    while True:
        success, frame = cap.read()
        if not success or frame is None:
            continue

        if joystick:
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    if event.button == 2:
                        hud_module.reload_weapon()
                    elif event.button == 3:
                        state["current_pan"] = 90
                        state["current_tilt"] = 90
                    elif event.button == 5:
                        state["step_idx"] = min(len(state["zoom_steps"]) - 1, state["step_idx"] + 1)
                        state["zoom_level"] = state["zoom_steps"][state["step_idx"]]
                    elif event.button == 4:  
                        state["step_idx"] = max(0, state["step_idx"] - 1)
                        state["zoom_level"] = state["zoom_steps"][state["step_idx"]]
                    elif event.button == 1:
                        state["current_pan"] = 90
                        state["current_tilt"] = 90
                        cap.stop()
                        cv2.destroyAllWindows() 

                elif event.type == pygame.JOYHATMOTION:
                    hat = joystick.get_hat(0)
                    if hat[0] == 1:
                        if hud_module.fire_mode == "FULL_AUTO":
                            hud_module.fire_mode = "BURST"
                            hud_module.mode_sound.play()  
                        else:
                            hud_module.fire_mode = "FULL_AUTO"
                            hud_module.mode_sound.play() 
                    
                    # 3. D-PAD DOWN TO TOGGLE AUTONOMOUS TRACKING
                    if hat[1] == -1: 
                        state["autonomous_mode"] = not state["autonomous_mode"]
                        mode_status = "AKTIF" if state["autonomous_mode"] else "NYAHAKTIF"
                        print(f"[SISTEM] Mod Autonomi: {mode_status}")
                        if state["autonomous_mode"]:
                            state["current_pan"] = 90.0
                            state["current_tilt"] = 90.0

            # 4. DISABLE MANUAL ANALOG STICK AIMING WHEN AUTONOMOUS MODE IS ON
            if not state["autonomous_mode"]:
                tilt_axis = joystick.get_axis(3)
                pan_axis = joystick.get_axis(2)

                if abs(pan_axis) > DEADZONE:
                    state["current_pan"] += (-pan_axis) * xbox_speed
                if abs(tilt_axis) > DEADZONE:
                    state["current_tilt"] += (-tilt_axis) * xbox_speed

            rt_axis = joystick.get_axis(5)
            if rt_axis > 0.5:
                hud_module.start_firing()
            elif rt_axis < 0.5 and not pygame.mouse.get_pressed()[0]: 
                hud_module.stop_firing()  
            
        state["current_pan"] = max(0,min(180, state["current_pan"]))
        state["current_tilt"] = max(0,min(180, state["current_tilt"]))

        if state["zoom_level"] != state["last_zoom_level"]:
            hud_module.play_zoom_sound(state["zoom_level"])
            state["last_zoom_level"] = state["zoom_level"]

        zoomed_frame = hud_module.apply_zoom(frame, state["zoom_level"])
        frame_ai, boxes = ai_module.scan_frame(zoomed_frame)
        
        update_fire = hud_module.update_firing()
        laser_flag = 1 if update_fire else 0
        drone_detected = 1 if len(boxes) > 0 else 0

        # 5. CORE AUTONOMOUS TRACKING & SEARCH LOGIC
        if state["autonomous_mode"]:
            if drone_detected:
                try:
                    # Extract the first detected bounding box
                    box = boxes[0] 
                    
                    # Force conversion to standard floats to prevent Tensor/NumPy TypeErrors
                    # Assuming TurretVision boxes return standard [x1, y1, x2, y2]
                    x1 = float(box[0])
                    y1 = float(box[1])
                    x2 = float(box[2])
                    y2 = float(box[3])
                    
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    
                    """
                    # NOTE: If your TurretVision actually outputs [x, y, width, height], 
                    # comment out the two lines above and uncomment the two lines below:
                    # cx = x1 + (x2 / 2.0)
                    # cy = y1 + (y2 / 2.0)
                    """
                    
                    # 540 and 360 represent the center of your 1080x720 frame
                    error_x = cx - 540.0  
                    error_y = cy - 360.0  
                    
                    if abs(error_x) > state["tracking_deadzone"]:
                        state["current_pan"] -= error_x * state["pan_kp"]
                    if abs(error_y) > state["tracking_deadzone"]:
                        state["current_tilt"] -= error_y * state["tilt_kp"]
                        
                except Exception as e:
                    # This will catch the error and print it to your terminal instead of hiding it
                    print(f"[RALAT TRACKING] Format box tidak sepadan! Error: {e} | Data: {boxes}")
                    
            else:
                # Sweep mode: Patrols left and right when no drone is detected
                state["current_pan"] += (state["search_speed"] * state["search_direction"])
                state["current_tilt"] = 90.0 # Maintain level horizon while searching
                
                if state["current_pan"] >= 180.0:
                    state["current_pan"] = 180.0
                    state["search_direction"] = -1
                elif state["current_pan"] <= 0.0:
                    state["current_pan"] = 0.0
                    state["search_direction"] = 1

        fire_mode_flag = 1 if hud_module.fire_mode == "BURST" else 0

        p_int , t_int = int(state["current_pan"]), int(state["current_tilt"])
        
        current_time = time.time()
        current_ammo = hud_module.ammo
        
        is_moving_event = (p_int != last_sent_pan or t_int != last_sent_tilt)
        is_state_changed =  (current_ammo !=last_sent_ammo) or (drone_detected != last_sent_drone)

        if laser_flag == 1 or is_state_changed or (is_moving_event and (current_time - last_udp_time > 0.05)):
            current_ammo = hud_module.ammo
            msg = f"{p_int},{t_int},{laser_flag},{drone_detected},{current_ammo},{fire_mode_flag}"
            print(f"[DEBUG UDP] Menghantar ke ESP32: Pan={p_int}, Tilt={t_int}, Peluru={current_ammo}")
            sock.sendto(msg.encode(), (UDP_IP, UDP_PORT))
            last_sent_pan, last_sent_tilt = p_int, t_int
            last_udp_time = current_time

        if update_fire:
            if XInput.get_connected()[0]:
                XInput.set_vibration(0, 65535, 65535)  
                rumble_end_time = time.time() + 0.08
                is_rumbling = True
        
        if is_rumbling and time.time() > rumble_end_time:
            if XInput.get_connected()[0]:
                XInput.set_vibration(0, left_speed=0.0, right_speed=0.0)
            is_rumbling = False

        final_frame = hud_module.draw_hud(frame_ai, state["zoom_level"])
        cv2.imshow(window_name, final_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):  
            hud_module.reload_weapon()
        elif key == ord('p'):
            cap.stop()
            cv2.destroyAllWindows()
            pygame.quit()
            import subprocess
            subprocess.Popen(
                [sys.executable, r"D:\RENTAP 1 - Workspace\main_system.py"], 
                creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NO_WINDOW
            )
            sys.exit()

    state["current_pan"] = 90
    state["current_tilt"] = 90
    cap.stop()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()