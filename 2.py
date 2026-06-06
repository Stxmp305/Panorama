import sys
import cv2
import numpy as np

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def nothing(x): pass

def ultimate_panorama_stitch(video_top_path, video_bottom_path):
    cap_top = cv2.VideoCapture(video_top_path)
    cap_bot = cv2.VideoCapture(video_bottom_path)

    if not cap_top.isOpened() or not cap_bot.isOpened():
        print("Error: ไม่สามารถเปิดไฟล์วิดีโอได้")
        return

    width = int(cap_top.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_top.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap_top.get(cv2.CAP_PROP_FPS))

    # --- สเต็ป 1: Setup UI เพื่อสร้าง "Panorama Background" ---
    cap_top.set(cv2.CAP_PROP_POS_FRAMES, 60)
    cap_bot.set(cv2.CAP_PROP_POS_FRAMES, 60)
    _, ref_top = cap_top.read()
    _, ref_bot = cap_bot.read()

    win_setup = "Setup Panorama (Press ENTER to Start)"
    cv2.namedWindow(win_setup, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO)
    cv2.createTrackbar('Y Offset', win_setup, int(height * 0.7), height, nothing)
    cv2.createTrackbar('Cut Line %', win_setup, 50, 100, nothing)
    cv2.createTrackbar('Feather (Blend)', win_setup, 30, 150, nothing)

    print("\n--- 🛠️ ตั้งค่าฉากหลัง Panorama ---")
    print("1. ปรับ Y Offset ให้ถังดับเพลิงและพื้นตรงกันที่สุด")
    print("2. ปรับ Cut Line และ Feather ให้รอยต่อเนียนตา")
    print("3. กด ENTER เพื่อเริ่มรันระบบไดคัทมือ (Motion Overlay)")

    dy, cut_percent, feather = 0, 50, 30
    while True:
        dy = cv2.getTrackbarPos('Y Offset', win_setup)
        cp = cv2.getTrackbarPos('Cut Line %', win_setup)
        feather = cv2.getTrackbarPos('Feather (Blend)', win_setup)

        canvas_h = height + dy
        canvas = np.zeros((canvas_h, width, 3), dtype=np.uint8)
        
        # วางภาพล่าง
        canvas[dy:canvas_h, :] = ref_bot
        
        # วางภาพบนแบบชั่วคราวให้เห็นรอยต่อ
        added = cv2.addWeighted(canvas[0:height, :], 0.5, ref_top, 0.5, 0)
        canvas[0:height, :] = ref_top
        canvas[dy:height, :] = added[dy:height, :]

        # วาดเส้นจุดตัด
        cut_y = int(dy + ((height - dy) * (cp / 100.0)))
        cv2.line(canvas, (0, cut_y), (width, cut_y), (0, 0, 255), 2)

        cv2.imshow(win_setup, cv2.resize(canvas, (0,0), fx=0.4, fy=0.4))
        if cv2.waitKey(30) & 0xFF in [13, 32]: # กด Enter
            cut_percent = cp
            break
    cv2.destroyAllWindows()

    # --- สเต็ป 2: เรนเดอร์ (Panorama Background + Motion Hand) ---
    cap_top.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_bot.set(cv2.CAP_PROP_POS_FRAMES, 0)

    final_h = height + dy
    cut_y = int(dy + ((height - dy) * (cut_percent / 100.0)))
    
    # สร้าง Background Blend Mask (ทำให้ฉากหลังเนียนเป็น Panorama)
    bg_mask = np.zeros((final_h, width, 1), dtype=np.float32)
    bg_mask[0:cut_y, :] = 1.0 
    if feather > 0:
        ksize = (feather * 2) + 1
        bg_mask = cv2.GaussianBlur(bg_mask, (ksize, ksize), 0).reshape(final_h, width, 1)

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    win_main = 'True Panorama + Motion Protect'
    cv2.namedWindow(win_main, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO)

    print("\n🎬 กำลังเรนเดอร์ภาพ... ฉากหลังจะเนียนเป็นชิ้นเดียว และมือจะไม่ขาด")

    while True:
        ret_t, frame_top = cap_top.read()
        ret_b, frame_bot = cap_bot.read()
        if not ret_t or not ret_b: break

        # 1. สร้าง Panorama Background ให้เนียนที่สุด
        top_canvas = np.zeros((final_h, width, 3), dtype=np.uint8)
        top_canvas[0:height, :] = frame_top
        
        bot_canvas = np.zeros((final_h, width, 3), dtype=np.uint8)
        bot_canvas[dy:final_h, :] = frame_bot[0:height, :]

        # ผสมฉากหลังด้วย bg_mask (นี่คือจุดที่ทำให้มันเป็น Panorama)
        panorama_bg = (top_canvas.astype(np.float32) * bg_mask) + (bot_canvas.astype(np.float32) * (1.0 - bg_mask))

        # 2. จับการเคลื่อนไหวของมือ/กล่อง จากกล้องบน
        fg_mask = bg_subtractor.apply(frame_top)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel, iterations=3)
        fg_mask_blur = cv2.GaussianBlur(fg_mask, (21, 21), 0).astype(np.float32) / 255.0
        
        # ขยาย mask ให้เต็มผืนผ้าใบใหญ่
        full_fg_mask = np.zeros((final_h, width, 1), dtype=np.float32)
        full_fg_mask[0:height, :, 0] = fg_mask_blur
        full_fg_mask_3d = np.repeat(full_fg_mask, 3, axis=2)

        # 3. 💥 ทับซ้อน (Composite): วางมือที่ขยับ ลงบนฉากหลัง Panorama
        final_frame = (top_canvas.astype(np.float32) * full_fg_mask_3d) + (panorama_bg * (1.0 - full_fg_mask_3d))

        cv2.imshow(win_main, cv2.resize(final_frame.astype(np.uint8), (0,0), fx=0.4, fy=0.4))
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap_top.release()
    cap_bot.release()
    cv2.destroyAllWindows()
    print("\n✅ เสร็จสมบูรณ์!")

# ทดสอบรัน
ultimate_panorama_stitch(
    r"C:\Users\Purip\OneDrive\เอกสาร\802330642.782205.mp4", 
    r"C:\Users\Purip\OneDrive\เอกสาร\802330642.712218.mp4"
)