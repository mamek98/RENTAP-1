# Import OpenCV for image processing, color space conversions, and rendering
import cv2
# Import Pygame for audio playback (mixer)
import pygame
# Import time module to handle delays, timing for firing rate
import time
# Import NumPy for fast numerical operations, especially with image arrays
import numpy as np
# Import Pillow (PIL) for advanced text rendering with custom fonts over the OpenCV image
from PIL import Image, ImageDraw, ImageFont

class HeadUpDisplay:
    """
    Class to manage the Head-Up Display (HUD) overlay and sound effects.
    It handles drawing crosshairs, ammo count, system status, zoom levels,
    and also manages firing logic (burst/auto) and playing corresponding sounds.
    """
    def __init__(self, max_ammo=1000):
        # Current ammunition count
        self.ammo = max_ammo
        # Maximum possible ammunition
        self.max_ammo = max_ammo
        
        try:
            # Load the primary custom font from assets at size 30 for main HUD elements
            self.custom_font = ImageFont.truetype('assets/fonts/Purista.otf', 30)
            # Load a secondary custom font at size 25 for smaller HUD elements (like zoom text)
            self.custom_font2 = ImageFont.truetype('assets/fonts/Purista.otf', 25)
        except:
            # Fallback to default font if the custom font file is not found
            print("[AMARAN] Fail font Jura.ttf tidak dijumpai! Guna font default.")
            self.custom_font = ImageFont.load_default()

        try:
            # Load the crosshair image and convert it to RGBA to support transparency
            self.crosshair_img = Image.open('assets/images/Crosshair.png').convert("RGBA")
            
            # Resize the crosshair image to a fixed 600x600 resolution for the HUD
            self.crosshair_img = self.crosshair_img.resize((600, 600))
        except:
            # Handle the case where the crosshair image is missing
            print("[AMARAN] Fail gambar crosshair.png tidak dijumpai!")
            # Set crosshair to None to prevent errors during rendering
            self.crosshair_img = None
        
        # Define the current firing mode (can be "FULL_AUTO" or "BURST")
        self.fire_mode = "FULL_AUTO"
        # Counter to track how many shots have been fired in the current burst
        self.burst_shots_fired = 0

        # Rounds Per Minute (RPM) defining the fire rate of the weapon
        self.rpm = 1500
        # Calculate the delay between each shot in seconds based on RPM
        self.fire_delay = 60 / self.rpm
        # Timestamp of the last shot fired, used to enforce the fire delay
        self.last_fire_time = 0
        # Boolean flag indicating whether the trigger is currently held down
        self.is_firing = False

        # --- AUDIO OPTIMIZATION FOR STACKING MULTIPLE SOUNDS ---
        # Pre-initialize the mixer with standard CD-quality audio settings to reduce latency
        pygame.mixer.pre_init(44100, -16, 2, 512)
        # Initialize the pygame mixer module
        pygame.mixer.init()
        # Set the number of simultaneous playback channels to 32 to allow many overlapping sounds
        pygame.mixer.set_num_channels(32) 
        
        try:
            # Load the firing sound effect
            self.sound_fire = pygame.mixer.Sound('assets/sounds/fire.wav')
            # Reduce the volume of the fire sound to 50% to prevent distortion when multiple sounds overlap
            self.sound_fire.set_volume(0.5) 
            # Load the sound effect for switching firing modes
            self.mode_sound = pygame.mixer.Sound('assets/sounds/FireMode2.wav')
            # Load the sound effect for 2x zoom level
            self.zoom_2x = pygame.mixer.Sound('assets/sounds/Zoom2x .wav')
            # Load the sound effect for 4x zoom level
            self.zoom_4x = pygame.mixer.Sound('assets/sounds/Zoom4x .wav')
            # Load the sound effect for 8x zoom level
            self.zoom_8x = pygame.mixer.Sound('assets/sounds/Zoom8x .wav')
            # Load the sound effect for reloading the weapon
            self.reload_fire = pygame.mixer.Sound('assets/sounds/reload.wav')
        except:
            # Print a warning if any of the audio files are missing
            print("[AMARAN] Fail audio tidak dijumpai dalam folder assets/sounds/")

    def apply_zoom(self, frame, zoom_factor):
        """
        Applies a digital zoom to the camera frame.
        """
        # If zoom factor is 1.0 or less, no zoom is needed, return the original frame
        if zoom_factor <= 1.0 :
            return frame
        
        # Get the height and width of the current frame
        h, w = frame.shape[:2]
        # Calculate the center point coordinates of the frame
        center_x, center_y = w / 2 , h / 2

        # Calculate the new width and height based on the zoom factor
        new_w , new_h = w / zoom_factor , h / zoom_factor

        # Determine the bounding box for the zoomed region, ensuring it stays within frame boundaries
        x1 = max(0, int(center_x - new_w / 2))
        y1 = max(0, int(center_y - new_h / 2))
        x2 = min(w, int(center_x + new_w / 2))
        y2 = min(h, int(center_y + new_h / 2))

        # Crop the frame to the calculated bounding box
        cropped_frame = frame[y1:y2, x1:x2]

        # Resize the cropped region back to the original resolution to simulate zoom
        return cv2.resize(cropped_frame, (w, h))
            
    def draw_hud(self, frame, zoom_level=1.0):
        """
        Draws the HUD overlay (crosshair, ammo, mode, system status) on the frame.
        """
        # Get frame dimensions
        h, w, _ = frame.shape
        # Calculate center coordinates
        center_x, center_y = int(w/2), int(h/2)

        """
        # 1. Lukis Crosshair (KOD LAMA YANG DITUTUP)
        color = (0, 255, 0)
        thickness = 2
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), color, thickness)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), color, thickness)
        cv2.circle(frame, (center_x, center_y), 30, color, 1)
        """

        # Convert the OpenCV BGR frame to RGB format required by PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Create a PIL Image object from the RGB numpy array
        img_pil = Image.fromarray(frame_rgb)
        # Create a drawing context for the PIL image
        draw = ImageDraw.Draw(img_pil)

        # If the crosshair image was successfully loaded
        if self.crosshair_img is not None:
            # Get the width and height of the crosshair image
            cw, ch = self.crosshair_img.size
            # Calculate the top-left position to center the crosshair on the screen
            pos_x = center_x - int(cw/2)
            pos_y = center_y - int(ch/2)

            # Paste the crosshair onto the image, using itself as a transparency mask
            img_pil.paste(self.crosshair_img, (pos_x, pos_y), self.crosshair_img)

        # Define text color: green if ammo > 0, otherwise red
        text_color_pil = (0, 255, 0) if self.ammo > 0 else (255, 0, 0)

        # Prepare the ammo display text string (bottom-left)
        ammo_text = f"AMMO: {self.ammo}/{self.max_ammo}"
        draw.text((60, h - 90), ammo_text, font=self.custom_font, fill=text_color_pil)

        # Prepare the firing mode text (bottom-center)
        mode_text = f"MODE: {self.fire_mode}"
        draw.text((int(w * 0.42), h - 90), mode_text, font=self.custom_font, fill=text_color_pil)

        # Prepare the system status text (bottom-right)
        status_text = "SYSTEM: READY" if self.ammo > 0 else "SYSTEM: RELOAD REQUIRED"
        draw.text((w - 520, h - 90), status_text, font=self.custom_font, fill=text_color_pil)

        # Prepare the current zoom level text (center-right)
        zoom_text = f"{zoom_level:.1f}X"
        draw.text((w - 220, center_y - 16), zoom_text, font=self.custom_font2, fill=(0, 255, 0))

        # Convert the modified PIL image back to an OpenCV BGR numpy array
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Return the final composed frame
        return frame
    
    def start_firing(self):
        """
        Activates the firing mechanism if ammo is available.
        Handles both FULL_AUTO and BURST logic initialization.
        """
        # Only allow firing if there is ammunition left
        if self.ammo > 0:
            # If in burst mode, only start firing if not currently firing and burst count is 0
            if self.fire_mode == "BURST":
                if not self.is_firing and self.burst_shots_fired == 0:
                    self.is_firing = True
            # If in full auto mode, simply set firing to True
            else:
                self.is_firing = True

    def stop_firing(self):
        """
        Deactivates the firing mechanism. Natural sound decay is preserved.
        """
        # If in full auto mode, stop firing immediately
        if self.fire_mode == "FULL_AUTO":
            self.is_firing = False

        # If in burst mode and not actively firing, reset the burst counter to allow next burst
        if self.fire_mode == "BURST" and not self.is_firing:
            self.burst_shots_fired = 0
        
    def update_firing(self):
        """
        Called every frame to process firing logic, deduct ammo, and play sound effects.
        Returns True if a shot was fired this frame, False otherwise.
        """
        # Flag to indicate if a bullet was fired during this specific frame update
        shot_fired_this_frame = False

        # Check if the trigger is actively held/firing
        if self.is_firing:
            # Verify we have ammo to fire
            if self.ammo > 0:
                # Get the current system time in seconds
                current_time = time.time()
                
                # Check if enough time has passed since the last shot (based on RPM)
                if current_time - self.last_fire_time >= self.fire_delay:
                    # Deduct one bullet
                    self.ammo -= 1
                    # Update the last fire timestamp to the current time
                    self.last_fire_time = current_time

                    # Set the flag to indicate a shot was fired
                    shot_fired_this_frame = True
                    
                    # If the pygame audio mixer is successfully initialized
                    if pygame.mixer.get_init():
                        # Find an available audio channel. True parameter forces it to steal a channel if all are full
                        channel = pygame.mixer.find_channel(True)
                        # If a channel was acquired, play the firing sound
                        if channel:
                            channel.play(self.sound_fire)
                    
                    # Special logic for burst mode
                    if self.fire_mode == "BURST":
                        # Increment the burst shot counter
                        self.burst_shots_fired += 1
                        # If 5 shots have been fired in this burst, stop firing automatically
                        if self.burst_shots_fired >= 5: # Limit 5 das
                            self.is_firing = False
            # If we run out of ammo while trying to fire
            else:
                # Stop firing automatically
                self.is_firing = False 
                # Print a notification to the console
                print("KKLIK! Peluru Habis!")

        # Return the shot fired status to the main system
        return shot_fired_this_frame

    def reload_weapon(self):
        """
        Reloads the weapon to maximum capacity and plays the reload sound.
        """
        # If ammo is already full, do nothing
        if self.ammo == self.max_ammo:
            print("Peluru sudah penuh, tidak perlu reload!")
            return
        
        # Reset the current ammo back to the maximum capacity
        self.ammo = self.max_ammo
        # Stop firing forcefully during reload
        self.is_firing = False 
        
        # Play the reload sound effect if the mixer is initialized
        if pygame.mixer.get_init():
            self.reload_fire.play()
        # Print a status message to the console
        print("RELOADING... Peluru penuh!")
    
    def play_zoom_sound(self, current_zoom):
        """
        Plays the appropriate sound effect when the zoom level changes.
        """
        # If the mixer is not initialized, do nothing
        if not pygame.mixer.get_init():
            return
        
        # Forcefully find an available audio channel
        channel = pygame.mixer.find_channel(True)
        # Return if no channel is available
        if not channel:
            return
        
        # Check the current zoom level and play the corresponding sound effect if it exists
        if current_zoom == 2.0 and hasattr(self, 'zoom_2x'):
            channel.play(self.zoom_2x)
        elif current_zoom == 4.0 and hasattr(self, 'zoom_4x'):
            channel.play(self.zoom_4x)
        elif current_zoom == 8.0 and hasattr(self, 'zoom_8x'):
            channel.play(self.zoom_8x)
        elif current_zoom == 1.0:
            # Use zoom_2x sound for returning to 1x zoom as a fallback indicator
            channel.play(self.zoom_2x)