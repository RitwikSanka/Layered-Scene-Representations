import os
import cv2
import numpy as np

def generate_photorealistic_scene(idx):
    """Generates procedurally rendered realistic multi-plane scenes with depth cues."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    if idx == 1: # Landscape with Sky, Mountains, Lake, Trees, Foreground Rock
        # Sky gradient
        for y in range(120):
            r = int(120 + 60 * (y / 120))
            g = int(170 + 50 * (y / 120))
            b = int(230 + 25 * (y / 120))
            img[y, :] = [b, g, r]
        # Distant Mountain (far)
        pts_m1 = np.array([[0, 120], [60, 60], [140, 110], [210, 50], [256, 120]], np.int32)
        cv2.fillPoly(img, [pts_m1], (140, 120, 110))
        # Midground Hill & Forest (mid)
        pts_m2 = np.array([[0, 160], [80, 110], [180, 130], [256, 100], [256, 180], [0, 180]], np.int32)
        cv2.fillPoly(img, [pts_m2], (60, 110, 50))
        # Lake reflection (near-mid)
        for y in range(160, 210):
            img[y, :] = [180 - (y-160)*2, 140 - (y-160), 90]
        # Foreground Grass Bank & Boulder (near)
        pts_fg = np.array([[0, 200], [100, 205], [190, 195], [256, 210], [256, 256], [0, 256]], np.int32)
        cv2.fillPoly(img, [pts_fg], (35, 75, 30))
        cv2.ellipse(img, (180, 225), (45, 25), 0, 0, 360, (90, 95, 100), -1)
        
    elif idx == 2: # Architectural Street: Sky, Building, Road, Car, Pedestrian
        # Sky
        img[:100, :] = [220, 180, 135]
        # Building (far-mid)
        cv2.rectangle(img, (20, 30), (140, 180), (160, 150, 145), -1)
        for wy in range(45, 160, 30):
            for wx in range(35, 130, 25):
                cv2.rectangle(img, (wx, wy), (wx+15, wy+20), (60, 50, 45), -1)
        # Sidewalk & Road (mid-near)
        cv2.rectangle(img, (0, 180), (256, 200), (180, 180, 180), -1)
        cv2.rectangle(img, (0, 200), (256, 256), (70, 70, 70), -1)
        # Car (near)
        cv2.rectangle(img, (110, 190), (220, 235), (30, 40, 180), -1) # Red car
        cv2.rectangle(img, (135, 175), (195, 190), (50, 60, 190), -1) # Cabin
        cv2.circle(img, (135, 235), 14, (20, 20, 20), -1)
        cv2.circle(img, (195, 235), 14, (20, 20, 20), -1)
        
    elif idx == 3: # Indoor Living Room: Wall, Window, Floor, Sofa, Coffee Table
        # Wall (far)
        img[:140, :] = [210, 220, 225]
        # Window (far)
        cv2.rectangle(img, (140, 20), (230, 110), (240, 200, 170), -1)
        cv2.rectangle(img, (140, 20), (230, 110), (120, 100, 90), 3)
        # Wooden Floor (mid)
        for y in range(140, 256):
            img[y, :] = [50 + int((y-140)*0.4), 80 + int((y-140)*0.5), 130 + int((y-140)*0.5)]
        # Sofa (near-mid)
        cv2.rectangle(img, (20, 120), (160, 190), (140, 70, 60), -1) # Blue sofa
        cv2.rectangle(img, (15, 140), (165, 190), (120, 55, 45), -1)
        # Coffee Table (near)
        cv2.rectangle(img, (60, 185), (180, 220), (40, 70, 120), -1) # Wood table
        cv2.line(img, (75, 220), (75, 245), (30, 50, 90), 4)
        cv2.line(img, (165, 220), (165, 245), (30, 50, 90), 4)
        
    elif idx == 4: # Beach Sunset: Gradient Sky, Sea, Palm Tree, Beach Umbrella
        # Sunset Sky
        for y in range(130):
            img[y, :] = [int(80 + y*0.9), int(120 + y*0.6), int(240 - y*0.8)]
        # Sea
        for y in range(130, 175):
            img[y, :] = [180 - (y-130)*2, 110 - (y-130), 40]
        # Sand
        for y in range(175, 256):
            img[y, :] = [130 + int(y*0.3), 190 + int(y*0.2), 220]
        # Umbrella (near)
        pts_u = np.array([[90, 160], [170, 160], [130, 120]], np.int32)
        cv2.fillPoly(img, [pts_u], (40, 50, 220)) # Orange-red umbrella
        cv2.line(img, (130, 120), (130, 210), (80, 80, 80), 3)
        # Palm trunk (mid)
        cv2.ellipse(img, (220, 140), (60, 20), -40, 0, 360, (40, 140, 50), -1)
        
    else: # idx == 5: Forest Pathway with Trees, Trail, Boulder
        # Canopied Sky
        img[:90, :] = [190, 170, 140]
        # Distant foliage
        for cx in range(20, 250, 45):
            cv2.circle(img, (cx, 85), 35, (40, 90, 35), -1)
        # Ground
        for y in range(90, 256):
            img[y, :] = [45 + int(y*0.2), 85 + int(y*0.3), 60 + int(y*0.1)]
        # Trail
        pts_trail = np.array([[120, 90], [135, 90], [180, 256], [70, 256]], np.int32)
        cv2.fillPoly(img, [pts_trail], (90, 120, 145))
        # Foreground Trees (near)
        cv2.rectangle(img, (25, 40), (55, 256), (30, 50, 75), -1)
        cv2.rectangle(img, (200, 20), (235, 256), (25, 45, 70), -1)
        
    return img

def generate_anime_scene(idx):
    """Generates cel-shaded anime styled illustrations with sharp ink outlines."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    if idx == 1: # Anime City Skyline & Silhouette Character
        # Twilight cel gradient
        for y in range(160):
            img[y, :] = [int(190 - y*0.6), int(100 + y*0.3), int(120 + y*0.7)]
        # Anime clouds (soft flat)
        cv2.circle(img, (60, 60), 30, (230, 210, 220), -1)
        cv2.circle(img, (95, 55), 35, (230, 210, 220), -1)
        # Distant skyscrapers (purple silhouettes)
        cv2.rectangle(img, (30, 90), (70, 160), (120, 60, 75), -1)
        cv2.rectangle(img, (80, 70), (130, 160), (100, 50, 65), -1)
        cv2.rectangle(img, (140, 100), (180, 160), (115, 55, 70), -1)
        # Rooftop ledge (near-mid)
        cv2.rectangle(img, (0, 160), (256, 256), (45, 35, 40), -1)
        cv2.line(img, (0, 160), (256, 160), (10, 10, 10), 3) # Cel outline
        # Character silhouette (near)
        cv2.circle(img, (190, 145), 18, (15, 10, 20), -1) # Head
        cv2.rectangle(img, (175, 163), (205, 220), (20, 15, 25), -1) # Body
        
    elif idx == 2: # Anime Cherry Blossom Tree & Torii Gate
        # Soft blue-pink sky
        for y in range(180):
            img[y, :] = [int(240 - y*0.3), int(210 - y*0.1), 250]
        # Distant Mountain Fuji silhouette
        pts_fuji = np.array([[20, 180], [110, 90], [150, 90], [240, 180]], np.int32)
        cv2.fillPoly(img, [pts_fuji], (200, 170, 180))
        cv2.polylines(img, [pts_fuji], False, (140, 110, 120), 2)
        # Ground
        cv2.rectangle(img, (0, 180), (256, 256), (120, 160, 130), -1)
        cv2.line(img, (0, 180), (256, 180), (30, 30, 30), 2)
        # Red Torii Gate (mid)
        cv2.rectangle(img, (50, 110), (60, 190), (30, 40, 200), -1) # Pillar
        cv2.rectangle(img, (120, 110), (130, 190), (30, 40, 200), -1) # Pillar
        cv2.rectangle(img, (40, 110), (140, 120), (30, 40, 200), -1) # Crossbeam
        # Sakura blossom cluster (near-fg)
        cv2.circle(img, (200, 70), 45, (210, 180, 250), -1)
        cv2.circle(img, (225, 95), 35, (195, 165, 240), -1)
        
    elif idx == 3: # Anime Classroom: Window, Desks, Blackboard
        # Classroom wall
        img[:, :] = (215, 230, 235)
        # Window with bright sky (far)
        cv2.rectangle(img, (10, 20), (100, 140), (250, 230, 190), -1)
        cv2.rectangle(img, (10, 20), (100, 140), (40, 40, 40), 2)
        cv2.line(img, (55, 20), (55, 140), (40, 40, 40), 2)
        # Blackboard (mid)
        cv2.rectangle(img, (120, 30), (245, 120), (60, 100, 50), -1)
        cv2.rectangle(img, (120, 30), (245, 120), (40, 60, 100), 4)
        # Wooden Desk (near)
        cv2.rectangle(img, (60, 160), (190, 210), (60, 120, 180), -1)
        cv2.rectangle(img, (60, 160), (190, 210), (20, 20, 20), 2)
        cv2.line(img, (75, 210), (75, 250), (20, 20, 20), 3)
        cv2.line(img, (175, 210), (175, 250), (20, 20, 20), 3)
        
    elif idx == 4: # Anime Fantasy Crystal Cave
        # Dark mystical cave background
        for y in range(256):
            img[y, :] = [int(60 + y*0.2), 20, int(40 + y*0.3)]
        # Giant glowing blue crystal (mid)
        pts_cry = np.array([[128, 40], [165, 120], [150, 200], [106, 200], [91, 120]], np.int32)
        cv2.fillPoly(img, [pts_cry], (240, 200, 80)) # Cyan crystal
        cv2.polylines(img, [pts_cry], True, (255, 255, 255), 2)
        # Cave floor ledge (near)
        pts_cave = np.array([[0, 190], [90, 180], [170, 185], [256, 175], [256, 256], [0, 256]], np.int32)
        cv2.fillPoly(img, [pts_cave], (30, 20, 30))
        cv2.polylines(img, [pts_cave], False, (10, 10, 10), 3)
        
    else: # idx == 5: Anime Meadow with Windmill
        # Blue clear sky
        img[:150, :] = [245, 215, 160]
        # Rolling green hills (far)
        cv2.circle(img, (70, 220), 120, (100, 190, 120), -1)
        cv2.circle(img, (200, 230), 130, (80, 170, 100), -1)
        # Windmill tower (mid)
        pts_wm = np.array([[160, 100], [180, 100], [190, 190], [150, 190]], np.int32)
        cv2.fillPoly(img, [pts_wm], (230, 230, 235))
        cv2.polylines(img, [pts_wm], True, (30, 30, 30), 2)
        # Blades
        cv2.line(img, (170, 100), (130, 70), (40, 40, 40), 3)
        cv2.line(img, (170, 100), (210, 130), (40, 40, 40), 3)
        # Foreground flowers (near)
        cv2.rectangle(img, (0, 220), (256, 256), (50, 140, 60), -1)
        for fx in range(25, 240, 35):
            cv2.circle(img, (fx, 235), 6, (60, 80, 240), -1)
            
    return img

def generate_vector_scene(idx):
    """Generates crisp SVG/vector-style flat graphic scenes with clean geometric layers."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    if idx == 1: # Modern App Analytics Dashboard Card
        # Flat UI background canvas
        img[:, :] = (245, 246, 248)
        # Card Container (mid)
        cv2.rectangle(img, (28, 30), (228, 226), (255, 255, 255), -1)
        cv2.rectangle(img, (28, 30), (228, 226), (220, 224, 230), 2)
        # Bar Chart elements (near-mid)
        bars = [(50, 140, 30, 60, (235, 120, 60)), (95, 100, 30, 100, (80, 180, 100)),
                (140, 70, 30, 130, (70, 130, 240)), (185, 120, 30, 80, (200, 80, 170))]
        for bx, by, bw, bh, col in bars:
            cv2.rectangle(img, (bx, by), (bx+bw, by+bh), col, -1)
        # Floating Notification Badge (near/foreground)
        cv2.circle(img, (205, 50), 16, (60, 70, 230), -1) # Red badge
        cv2.circle(img, (205, 50), 6, (255, 255, 255), -1)
        
    elif idx == 2: # Flat Design Rocket Launch Infographic
        # Deep space dark indigo background
        img[:, :] = (50, 25, 20)
        # Background moon (far)
        cv2.circle(img, (200, 60), 32, (180, 220, 240), -1)
        # Clouds layer (mid)
        cv2.circle(img, (60, 210), 45, (120, 70, 60), -1)
        cv2.circle(img, (130, 220), 55, (120, 70, 60), -1)
        cv2.circle(img, (200, 215), 45, (120, 70, 60), -1)
        # Rocket body (near)
        pts_rkt = np.array([[128, 45], [150, 120], [106, 120]], np.int32)
        cv2.fillPoly(img, [pts_rkt], (240, 240, 245))
        # Rocket fins
        pts_fin1 = np.array([[106, 105], [90, 130], [106, 125]], np.int32)
        pts_fin2 = np.array([[150, 105], [166, 130], [150, 125]], np.int32)
        cv2.fillPoly(img, [pts_fin1], (50, 60, 220))
        cv2.fillPoly(img, [pts_fin2], (50, 60, 220))
        # Porthole
        cv2.circle(img, (128, 85), 10, (230, 160, 50), -1)
        
    elif idx == 3: # Flat Mountain Sunrise Icon/Illustration
        # Pale yellow sky
        img[:, :] = (200, 240, 255)
        # Golden Sun (far)
        cv2.circle(img, (128, 120), 42, (60, 180, 250), -1)
        # Left Mountain (mid)
        pts_lm = np.array([[10, 220], [95, 110], [180, 220]], np.int32)
        cv2.fillPoly(img, [pts_lm], (180, 110, 60)) # Teal mountain
        # Right Mountain (near)
        pts_rm = np.array([[90, 220], [175, 80], [250, 220]], np.int32)
        cv2.fillPoly(img, [pts_rm], (140, 70, 30)) # Deep teal mountain
        cv2.rectangle(img, (0, 220), (256, 256), (80, 160, 90), -1) # Forest base
        
    elif idx == 4: # Flat Geometric Compass & Navigation Icon
        # Clean white card background
        img[:, :] = (250, 250, 250)
        # Outer Ring (far)
        cv2.circle(img, (128, 128), 90, (225, 230, 235), -1)
        cv2.circle(img, (128, 128), 75, (255, 255, 255), -1)
        # Dial markings (mid)
        cv2.line(img, (128, 42), (128, 60), (140, 140, 140), 3)
        cv2.line(img, (128, 196), (128, 214), (140, 140, 140), 3)
        cv2.line(img, (42, 128), (60, 128), (140, 140, 140), 3)
        cv2.line(img, (196, 128), (214, 128), (140, 140, 140), 3)
        # Compass Needle (near/foreground)
        pts_n = np.array([[128, 55], [142, 128], [114, 128]], np.int32)
        pts_s = np.array([[128, 201], [142, 128], [114, 128]], np.int32)
        cv2.fillPoly(img, [pts_n], (50, 60, 230)) # Red north needle
        cv2.fillPoly(img, [pts_s], (160, 160, 165)) # Gray south needle
        cv2.circle(img, (128, 128), 8, (40, 40, 45), -1) # Center brass pivot
        
    else: # idx == 5: Isometric Tech Server Stack
        # Light pastel slate background
        img[:, :] = (235, 230, 225)
        # Base platform (far-mid)
        pts_base = np.array([[128, 180], [210, 215], [128, 250], [46, 215]], np.int32)
        cv2.fillPoly(img, [pts_base], (170, 155, 145))
        # Server Unit 1 (lower, mid)
        cv2.rectangle(img, (60, 145), (196, 185), (80, 80, 90), -1)
        cv2.circle(img, (80, 165), 5, (70, 220, 80), -1) # Green LED
        cv2.circle(img, (100, 165), 5, (60, 180, 240), -1) # Blue LED
        # Server Unit 2 (upper, near)
        cv2.rectangle(img, (60, 95), (196, 135), (95, 95, 110), -1)
        cv2.circle(img, (80, 115), 5, (70, 220, 80), -1) # Green LED
        cv2.circle(img, (100, 115), 5, (50, 60, 220), -1) # Red LED
        # Floating Hologram Cube (near-fg)
        cv2.rectangle(img, (110, 40), (146, 76), (230, 140, 40), -1)
        cv2.rectangle(img, (110, 40), (146, 76), (255, 255, 255), 2)
        
    return img

def generate_dataset(base_dir="/home/doom/layered_reps/data"):
    categories = {
        'photorealistic': generate_photorealistic_scene,
        'anime': generate_anime_scene,
        'vector': generate_vector_scene
    }
    
    for cat_name, gen_fn in categories.items():
        cat_dir = os.path.join(base_dir, cat_name)
        os.makedirs(cat_dir, exist_ok=True)
        for i in range(1, 6):
            img_bgr = gen_fn(i)
            file_path = os.path.join(cat_dir, f"{cat_name}_{i}.png")
            cv2.imwrite(file_path, img_bgr)
            print(f"Generated [{cat_name}] sample: {file_path}")

if __name__ == "__main__":
    generate_dataset()
