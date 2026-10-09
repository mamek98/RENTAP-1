import cv2
import pygame
import time
import numpy as np
from PIL import Image,ImageDraw,ImageFont

class HeadUpDisplay:
    def __init__(self, max_ammo=1000):
        self.ammo = max_ammo
        self.max_ammo = max_ammo
        
        try:
            self.custom_font = ImageFont.truetype('assets/fonts/Purista.otf',30)
            self.custom_font2 = ImageFont.truetype('assets/fonts/Purista.otf',25)
            
        except:
            print("[AMARAN] Fail font Jura.ttf tidak dijumpai! Guna font default.")
            self.custom_font = ImageFont.load_default()

        try:
            # Guna RGBA supaya ia sokong latar belakang lutsinar (transparent)
            self.crosshair_img = Image.open('assets/images/Crosshair.png').convert("RGBA")
            
            # Pilihan: Kalau gambar tu terlalu besar, awak boleh resize terus di sini.
            # Contohnya kecilkan jadi saiz 60x60 piksel:
            self.crosshair_img = self.crosshair_img.resize((600, 600))
        except:
            print("[AMARAN] Fail gambar crosshair.png tidak dijumpai!")
            self.crosshair_img = None
        
        self.fire_mode = "FULL_AUTO"
        self.burst_shots_fired = 0

        self.rpm = 1500
        self.fire_delay = 60 / self.rpm
        self.last_fire_time = 0
        self.is_firing = False

        # --- OPTIMISASI AUDIO UNTUK EFEK BERTINDIH (STACKING) ---
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(32) # Buka 32 saluran supaya bunyi boleh bertindih banyak!
        
        try:
            self.sound_fire = pygame.mixer.Sound('assets/sounds/fire.wav')
            # Perlahankan sikit volume setiap peluru supaya bila bertindih 16 kali, speaker tak pecah
            self.sound_fire.set_volume(0.5) 
            self.mode_sound = pygame.mixer.Sound('assets/sounds/FireMode2.wav')
            self.zoom_2x = pygame.mixer.Sound('assets/sounds/Zoom2x .wav')
            self.zoom_4x = pygame.mixer.Sound('assets/sounds/Zoom4x .wav')
            self.zoom_8x = pygame.mixer.Sound('assets/sounds/Zoom8x .wav')
            self.reload_fire = pygame.mixer.Sound('assets/sounds/reload.wav')
        except:
            print("[AMARAN] Fail audio tidak dijumpai dalam folder assets/sounds/")

    def apply_zoom(self,frame, zoom_factor):
        if zoom_factor <= 1.0 :
            return frame
        
        h, w = frame.shape[:2]
        center_x,center_y = w / 2 , h / 2

        new_w , new_h = w / zoom_factor , h / zoom_factor

        x1 = max(0, int(center_x - new_w / 2))
        y1 = max(0, int(center_y - new_h / 2))
        x2 = min(w, int(center_x + new_w / 2))
        y2 = min(h, int(center_y + new_h / 2))

        cropped_frame = frame[y1:y2, x1:x2]

        return cv2.resize(cropped_frame, (w, h))
            
    def draw_hud(self, frame, zoom_level=1.0):
        h, w, _ = frame.shape
        center_x, center_y = int(w/2), int(h/2)

        """
        # 1. Lukis Crosshair
        color = (0, 255, 0)
        thickness = 2
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), color, thickness)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), color, thickness)
        cv2.circle(frame, (center_x, center_y), 30, color, 1)"""



        frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        draw  = ImageDraw.Draw(img_pil)

        if self.crosshair_img is not None:
            cw,ch = self.crosshair_img.size
            pos_x = center_x - int(cw/2)
            pos_y = center_y - int(ch/2)

            img_pil.paste(self.crosshair_img, (pos_x, pos_y), self.crosshair_img)

        text_color_pil = (0, 255, 0) if self.ammo > 0 else (255, 0, 0)

        # 2. Status Peluru
        ammo_text = f"AMMO: {self.ammo}/{self.max_ammo}"
        #text_color = (0, 255, 0) if self.ammo > 0 else (0, 0, 255)
        #cv2.putText(frame, ammo_text, (w - 250, h - 30), self.font, 0.8, text_color, 2)
        draw.text((40 , h -80), ammo_text,font=self.custom_font,fill=text_color_pil)

        # 3. Status Sistem
        status_text = "SYSTEM: READY" if self.ammo > 0 else "SYSTEM: RELOAD REQUIRED"
        draw.text((1000, h -80), status_text,font=self.custom_font,fill=text_color_pil)
        #cv2.putText(frame, status_text, (30, h - 30), self.font, 0.8, text_color, 2)
        
        mode_text = f"MODE: {self.fire_mode}"
        draw.text((450, h -80), mode_text, font=self.custom_font, fill=text_color_pil)

        # 3. Status Zoom (Atas Kanan) - KOD BARU
        zoom_text = f"{zoom_level:.1f}X"
        draw.text((w - 300, center_y - 16), zoom_text, font=self.custom_font2, fill=(0, 255, 0))

        

        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        return frame
    
    def start_firing(self):
        """Hanya menukar status bendera kepada True"""
        if self.ammo > 0:
            if self.fire_mode == "BURST":
                if not self.is_firing and self.burst_shots_fired == 0:
                    self.is_firing = True
            else:
                self.is_firing = True

        

    def stop_firing(self):
        """Menukar status bendera kepada False. Bunyi gema akan dibiarkan habis secara natural."""
        if self.fire_mode =="FULL_AUTO":
            self.is_firing = False

        if self.fire_mode =="BURST" and not self.is_firing:
            self.burst_shots_fired =0
        

    def update_firing(self):
        """Setiap kali masa matang, tolak peluru dan MAINKAN BUNYI SECARA PAKSA"""

        shot_fired_this_frame = False

        if self.is_firing:
            if self.ammo > 0:
                current_time = time.time()
                
                if current_time - self.last_fire_time >= self.fire_delay:
                    self.ammo -= 1
                    self.last_fire_time = current_time

                    shot_fired_this_frame = True
                    
                    if pygame.mixer.get_init():
                        # KOD BAHARU: find_channel(True) bermaksud paksa ambil channel.
                        # Kalau semua 32 channel penuh, ia akan potong gema tembakan terawal!
                        channel = pygame.mixer.find_channel(True)
                        if channel:
                            channel.play(self.sound_fire)
                    
                    if self.fire_mode == "BURST":
                        self.burst_shots_fired += 1
                        if self.burst_shots_fired >= 5: # Limit 5 das
                            self.is_firing = False

                    # KOD BAHARU: Kita 'comment' (TUTUP) print ni! 
                    # Jangan suruh terminal print setiap peluru, nanti sistem lag teruk.
                    # print(f"BAM! Baki: {self.ammo}") 
            else:
                self.is_firing = False 
                print("KKLIK! Peluru Habis!")

        return shot_fired_this_frame

    def reload_weapon(self):

        if self.ammo == self.max_ammo:
            print("Peluru sudah penuh, tidak perlu reload!")
            return
        
        self.ammo = self.max_ammo
        self.is_firing = False # Hentikan tembakan masa reload
        if pygame.mixer.get_init():
            self.reload_fire.play()
        print("RELOADING... Peluru penuh!")
    
    def play_zoom_sound(self,current_zoom):
        if not pygame.mixer.get_init():
            return
        
        channel = pygame.mixer.find_channel(True)
        if not channel:
            return
        
        if current_zoom == 2.0 and hasattr(self, 'zoom_2x'):
            channel.play(self.zoom_2x)
        elif current_zoom == 4.0 and hasattr(self, 'zoom_4x'):
            channel.play(self.zoom_4x)
        elif current_zoom == 8.0 and hasattr(self, 'zoom_8x'):
            channel.play(self.zoom_8x)
        elif current_zoom == 1.0:
            channel.play(self.zoom_2x)