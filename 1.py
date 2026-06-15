
import sys
import cv2
import numpy as np
import argparse
import time
from collections import deque
from pathlib import Path

# ป้องกันปัญหา UnicodeEncodeError บน Windows terminal เมื่อแสดงผลภาษาไทย/สัญลักษณ์พิเศษ
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# ─────────────────────────────────────────────
#  CONFIG — ค่าเริ่มต้น (แก้ได้ที่นี่ หรือผ่าน UI)
# ─────────────────────────────────────────────
DEFAULT_TOP  = r"C:\Users\Purip\OneDrive\เอกสาร\802330642.782205.mp4"
DEFAULT_BOT  = r"C:\Users\Purip\OneDrive\เอกสาร\802330642.712218.mp4"
OUTPUT_PATH  = "panorama_output.mp4"

# ─────────────────────────────────────────────
# 🎨 ฟังก์ชันโคลนนิ่งสีอัตโนมัติ (Color Transfer)
# ─────────────────────────────────────────────
def match_color(src, target):
    """
    ปรับสีและแสงของภาพ src ให้เหมือนกับภาพ target 
    โดยใช้เทคนิค LAB Color Space (ประมวลผลเร็วและเนียนที่สุดสำหรับวิดีโอ)
    """
    # แปลงเป็น LAB เพื่อแยก ความสว่าง (L) ออกจาก สี (A, B)
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    tar_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    # หาค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐาน
    src_mean, src_std = cv2.meanStdDev(src_lab)
    tar_mean, tar_std = cv2.meanStdDev(tar_lab)

    # ป้องกันค่าหารด้วย 0
    src_std = np.clip(src_std, 0.001, None)

    # สมการปรับสี: (src - src_mean) * (tar_std / src_std) + tar_mean
    out_lab = ((src_lab - src_mean.flatten()) * (tar_std.flatten() / src_std.flatten())) + tar_mean.flatten()
    
    # จำกัดขอบเขตไม่ให้สีล้น แล้วแปลงกลับเป็น BGR
    out_lab = np.clip(out_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────
#  BLEND FUNCTIONS
# ─────────────────────────────────────────────

def blend_hard(top_canvas, bot_warped, cut_y, feather_px, final_h, width):
    """Hard cut — ไม่มี feather เลย เร็วที่สุด"""
    out = bot_warped.copy()
    out[:cut_y] = top_canvas[:cut_y]
    return out


def blend_feather(top_canvas, bot_warped, cut_y, feather_px, final_h, width):
    """Linear feather — เกลี่ยขอบเป็นเส้นตรง"""
    if feather_px <= 0:
        return blend_hard(top_canvas, bot_warped, cut_y, 0, final_h, width)

    half = feather_px
    y0 = max(0, cut_y - half)
    y1 = min(final_h, cut_y + half)

    mask = np.zeros((final_h, 1), dtype=np.float32)
    mask[:y0] = 1.0
    mask[y1:] = 0.0
    if y1 > y0:
        ramp = np.linspace(1.0, 0.0, y1 - y0, dtype=np.float32)
        mask[y0:y1, 0] = ramp

    mask3 = mask.reshape(final_h, 1, 1)
    blended = (top_canvas.astype(np.float32) * mask3 +
               bot_warped.astype(np.float32) * (1.0 - mask3))
    return np.clip(blended, 0, 255).astype(np.uint8)


def blend_laplacian(top_canvas, bot_warped, cut_y, feather_px, final_h, width):
    """
    Cosine (Laplacian-style) feather — เนียนที่สุด
    ใช้ cosine curve แทน linear ทำให้ transition นุ่มนวลกว่า
    """
    if feather_px <= 0:
        return blend_hard(top_canvas, bot_warped, cut_y, 0, final_h, width)

    half = feather_px
    y0 = max(0, cut_y - half)
    y1 = min(final_h, cut_y + half)

    mask = np.zeros((final_h, 1), dtype=np.float32)
    mask[:y0] = 1.0
    mask[y1:] = 0.0
    if y1 > y0:
        t = np.linspace(0.0, np.pi, y1 - y0, dtype=np.float32)
        ramp = 0.5 * (1.0 + np.cos(t))
        mask[y0:y1, 0] = ramp

    mask3 = mask.reshape(final_h, 1, 1)
    blended = (top_canvas.astype(np.float32) * mask3 +
               bot_warped.astype(np.float32) * (1.0 - mask3))
    return np.clip(blended, 0, 255).astype(np.uint8)


BLEND_FUNCS = {
    "hard":       blend_hard,
    "feather":    blend_feather,
    "laplacian":  blend_laplacian,
}


# ─────────────────────────────────────────────
#  SETUP UI
# ─────────────────────────────────────────────

def run_setup_ui(ref_top, ref_bot, height, width):
    """
    แสดง Setup Window ให้ผู้ใช้ตั้งค่าก่อนเรนเดอร์
    กด ENTER หรือ SPACE เพื่อยืนยัน
    """
    win = "StitchLab Setup  [ENTER = Confirm | ESC = Quit]"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def noop(_): pass

    cv2.createTrackbar("Y Overlap Start  ", win, int(height * 0.70), height, noop)
    cv2.createTrackbar("Cut Line %       ", win, 50, 100, noop)
    cv2.createTrackbar("Feather px       ", win, 30, 120, noop)
    cv2.createTrackbar("Max Shake px     ", win, 3, 20, noop)
    # Blend: 0=Hard, 1=Feather, 2=Laplacian
    cv2.createTrackbar("Blend (0H 1F 2L) ", win, 2, 2, noop)
    cv2.createTrackbar("Color Match (0/1)", win, 1, 1, noop)
    cv2.createTrackbar("Motion Overlay (0/1)", win, 1, 1, noop)

    print("\n╔══════════════════════════════════════════╗")
    print("║         StitchLab — Setup Mode           ║")
    print("╠══════════════════════════════════════════╣")
    print("║  Trackbars:                              ║")
    print("║   Y Overlap  — เลื่อนภาพล่างขึ้น/ลง      ║")
    print("║   Cut Line % — เลือกจุดตัด (0=บน 100=ล่าง)║")
    print("║   Feather    — ความกว้างรอยเบลอ (px)     ║")
    print("║   Max Shake  — จำกัดการดิ้นของกล้องล่าง  ║")
    print("║   Blend      — 0=Hard 1=Feather 2=Laplace║")
    print("║   Color Match— 0=Off 1=On                ║")
    print("║   Motion Overlay — 0=Off 1=On (กันมือแหว่ง)║")
    print("╠══════════════════════════════════════════╣")
    print("║  ENTER / SPACE = ยืนยัน | ESC = ออก      ║")
    print("╚══════════════════════════════════════════╝\n")

    blend_names = ["HARD", "FEATHER", "LAPLACIAN"]

    while True:
        # Check if window was closed via 'X' button
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                raise SystemExit("Setup window closed.")
        except cv2.error:
            cv2.destroyAllWindows()
            raise SystemExit("Setup window closed.")

        cy    = cv2.getTrackbarPos("Y Overlap Start  ", win)
        cp    = cv2.getTrackbarPos("Cut Line %       ", win)
        fth   = cv2.getTrackbarPos("Feather px       ", win)
        shake = cv2.getTrackbarPos("Max Shake px     ", win)
        bmode = cv2.getTrackbarPos("Blend (0H 1F 2L) ", win)
        color_match = cv2.getTrackbarPos("Color Match (0/1)", win)
        motion_overlay = cv2.getTrackbarPos("Motion Overlay (0/1)", win)

        canvas_h = height + cy
        canvas = np.zeros((canvas_h, width, 3), dtype=np.uint8)

        # วางภาพบน
        canvas[0:height, :] = ref_top

        # Blend zone ของกล้องล่าง
        bot_placed = np.zeros((canvas_h, width, 3), dtype=np.uint8)
        ref_bot_matched = match_color(ref_bot, ref_top) if color_match == 1 else ref_bot
        bot_placed[cy:cy + height, :] = ref_bot_matched

        # แสดง overlap zone ด้วย alpha
        overlap_top = cy
        overlap_bot = min(height, canvas_h)
        if overlap_bot > overlap_top:
            canvas[overlap_top:overlap_bot, :] = cv2.addWeighted(
                canvas[overlap_top:overlap_bot, :], 0.55,
                bot_placed[overlap_top:overlap_bot, :], 0.45, 0
            )

        # วาดเส้น Cut Line
        cut_y = int(cy + (height - cy) * (cp / 100.0))
        cut_y = max(0, min(canvas_h - 1, cut_y))
        cv2.line(canvas, (0, cut_y), (width, cut_y), (0, 220, 80), 2)

        # วาด Feather zone
        if fth > 0:
            cv2.line(canvas, (0, max(0, cut_y - fth)),
                     (width, max(0, cut_y - fth)), (0, 160, 255), 1)
            cv2.line(canvas, (0, min(canvas_h - 1, cut_y + fth)),
                     (width, min(canvas_h - 1, cut_y + fth)), (0, 160, 255), 1)

        # HUD
        hud_lines = [
            f"Y Overlap: {cy}px  |  Final H: {canvas_h}px",
            f"Cut Line: {cp}%  ({cut_y}px)",
            f"Feather: {fth}px  |  Max Shake: {shake}px",
            f"Blend Mode: {blend_names[bmode]}  |  Color Match: {'ON' if color_match == 1 else 'OFF'}",
            f"Motion Overlay: {'ON' if motion_overlay == 1 else 'OFF'}",
            "",
            "[ ENTER / SPACE = Confirm  |  ESC = Quit ]",
        ]
        colors = [(255, 220, 80)] * 5 + [(0,0,0)] + [(100, 220, 255)]
        for i, (line, col) in enumerate(zip(hud_lines, colors)):
            cv2.putText(canvas, line, (12, 28 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(canvas, line, (12, 28 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1)

        scale = min(1.0, 700 / canvas_h, 700 / width)
        preview = cv2.resize(canvas, (0, 0), fx=scale, fy=scale)
        cv2.imshow(win, preview)

        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32):   # ENTER or SPACE
            cv2.destroyAllWindows()
            blend_key = ["hard", "feather", "laplacian"][bmode]
            print(f"✅ Config: Y={cy}, Cut={cp}%, Feather={fth}px, "
                  f"Shake={shake}px, Blend={blend_key}, Color Match={'ON' if color_match == 1 else 'OFF'}, "
                  f"Motion Overlay={'ON' if motion_overlay == 1 else 'OFF'}")
            return cy, cp, fth, shake, blend_key, color_match, motion_overlay
        if key == 27:          # ESC
            cv2.destroyAllWindows()
            raise SystemExit("Setup cancelled.")


# ─────────────────────────────────────────────
#  ECC BASELINE CALIBRATION
# ─────────────────────────────────────────────

def calibrate_ecc(ref_top, ref_bot, height, initial_dy):
    """
    หา baseline dx ที่แม่นยำด้วย ECC บน Overlap Zone
    ทำแค่ครั้งเดียวตอน startup
    """
    half = int(height * 0.4)
    overlap_top = ref_top[height - half:, :]
    overlap_bot = ref_bot[:half, :]

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-9)
    try:
        g_top = cv2.cvtColor(overlap_top, cv2.COLOR_BGR2GRAY)
        g_bot = cv2.cvtColor(overlap_bot, cv2.COLOR_BGR2GRAY)
        _, warp = cv2.findTransformECC(
            g_top, g_bot, warp, cv2.MOTION_TRANSLATION, criteria
        )
        dx = float(warp[0, 2])
        print(f"   ECC baseline dx = {dx:+.3f}px  (dy locked to {initial_dy}px)")
        return dx
    except cv2.error as e:
        print(f"   ⚠️  ECC failed ({e}), using dx=0")
        return 0.0


# ─────────────────────────────────────────────
#  MAIN STITCHER
# ─────────────────────────────────────────────

def stitch(
    top_src, bot_src,
    output_path=OUTPUT_PATH,
    live_mode=False,
):
    """
    top_src / bot_src : path string หรือ camera index (int)
    live_mode         : ถ้า True จะไม่เขียนไฟล์ แค่แสดงผล
    """

    cap_top = cv2.VideoCapture(top_src)
    cap_bot = cv2.VideoCapture(bot_src)

    if not cap_top.isOpened():
        raise IOError(f"ไม่สามารถเปิด source: {top_src}")
    if not cap_bot.isOpened():
        raise IOError(f"ไม่สามารถเปิด source: {bot_src}")

    fps    = cap_top.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap_top.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_top.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ────── ดึง Reference Frame ──────
    ref_frame_idx = 60 if not live_mode else 0
    cap_top.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_idx)
    cap_bot.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_idx)
    ok_t, ref_top = cap_top.read()
    ok_b, ref_bot = cap_bot.read()
    if not ok_t or not ok_b:
        # fallback: ลองเฟรมแรก
        cap_top.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap_bot.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _, ref_top = cap_top.read()
        _, ref_bot = cap_bot.read()

    # ────── Setup UI ──────
    initial_dy, cut_percent, feather_px, max_shake, blend_mode, color_match, motion_overlay = \
        run_setup_ui(ref_top, ref_bot, height, width)

    blend_fn = BLEND_FUNCS[blend_mode]
    final_h  = height + initial_dy

    # ────── ECC Baseline ──────
    print("\n🔬 Calibrating ECC baseline...")
    ref_bot_calib = match_color(ref_bot, ref_top) if color_match == 1 else ref_bot
    baseline_dx = calibrate_ecc(ref_top, ref_bot_calib, height, initial_dy)

    # ────── Calculate Crop Margins to remove black borders ──────
    crop_margin = int(np.ceil(abs(baseline_dx) + max_shake))
    # Make sure we don't crop more than 1/4 of the width
    crop_margin = min(crop_margin, width // 4)
    crop_w = width - crop_margin
    crop_w = (crop_w // 2) * 2  # Ensure even width for video codecs
    crop_margin = width - crop_w
    print(f"   Auto-cropping horizontal borders: width reduced from {width}px to {crop_w}px (crop margin = {crop_margin}px)")

    # ────── VideoWriter ──────
    out_vid = None
    if not live_mode:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_vid = cv2.VideoWriter(output_path, fourcc, fps, (crop_w, final_h))
        cap_top.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap_bot.set(cv2.CAP_PROP_POS_FRAMES, 0)
        total_frames = int(cap_top.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        total_frames = 0  # ไม่รู้จำนวนเฟรมใน live

    # ────── LK Parameters ──────
    lk_params = dict(
        winSize  = (21, 21),
        maxLevel = 3,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    feat_params = dict(
        maxCorners   = 200,
        qualityLevel = 0.01,
        minDistance  = 15,
        blockSize    = 7,
    )

    # BG Mask: ติดตามเฉพาะขอบภาพ (หลีกเลี่ยงมือ/กล่อง)
    bg_mask = np.zeros((height, width), dtype=np.uint8)
    bg_mask[0:int(height * 0.15), :]     = 255   # แถบบน
    bg_mask[:, 0:int(width * 0.08)]      = 255   # แถบซ้าย
    bg_mask[:, int(width * 0.92):]       = 255   # แถบขวา

    # ────── Background Subtractor for Motion Overlay ──────
    bg_subtractor = None
    morph_kernel = None
    if motion_overlay == 1:
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=False)
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # ────── State ──────
    prev_gray_top  = None
    prev_gray_bot  = None
    pts_top        = None
    pts_bot        = None
    REDETECT_EVERY = 30

    smooth_dx = float(baseline_dx)
    smooth_dy = float(initial_dy)
    ALPHA     = 0.08           # EMA alpha — เพิ่ม = ตอบสนองเร็วขึ้น, ลด = นิ่งขึ้น

    dx_buf = deque([baseline_dx] * 5, maxlen=5)
    dy_buf = deque([float(initial_dy)] * 5, maxlen=5)

    frame_idx  = 0
    t_start    = time.time()

    print(f"\n🎬 Rendering  |  {width}×{final_h}  |  {fps:.0f}fps  |  blend={blend_mode}")
    if not live_mode:
        print(f"   Total frames: {total_frames}")
    print("   Press Q to stop\n")

    cv2.namedWindow("StitchLab Output", cv2.WINDOW_AUTOSIZE)

    while True:
        ret_t, frame_top = cap_top.read()
        ret_b, frame_bot = cap_bot.read()
        if not ret_t or not ret_b:
            break

        # ── Color Matching ──
        if color_match == 1:
            frame_bot = match_color(frame_bot, frame_top)

        gray_top = cv2.cvtColor(frame_top, cv2.COLOR_BGR2GRAY)
        gray_bot = cv2.cvtColor(frame_bot, cv2.COLOR_BGR2GRAY)

        # ── LK Optical Flow ──
        raw_dx = baseline_dx
        raw_dy = float(initial_dy)

        if prev_gray_top is not None and pts_top is not None and len(pts_top) >= 10:
            new_top, st_t, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray_top, gray_top, pts_top, None, **lk_params)
            new_bot, st_b, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray_bot, gray_bot, pts_bot, None, **lk_params)

            good = (st_t.ravel() == 1) & (st_b.ravel() == 1)
            if good.sum() >= 8:
                rel_dx = np.median(new_top[good, 0, 0] - new_bot[good, 0, 0])
                # Y ล็อคไว้ ไม่ให้ขยับ (อ้างอิงจาก Setup)
                raw_dx = float(np.clip(rel_dx,
                                       baseline_dx - max_shake,
                                       baseline_dx + max_shake))
                raw_dy = float(np.clip(initial_dy,
                                       initial_dy - max_shake,
                                       initial_dy + max_shake))
                pts_top = new_top[good].reshape(-1, 1, 2)
                pts_bot = new_bot[good].reshape(-1, 1, 2)

        # ── Re-detect Points ──
        if frame_idx % REDETECT_EVERY == 0 or pts_top is None or len(pts_top) < 15:
            corners = cv2.goodFeaturesToTrack(gray_top, mask=bg_mask, **feat_params)
            if corners is not None:
                pts_top = corners
                pts_bot = pts_top.copy()
                pts_bot[:, :, 1] -= initial_dy

        prev_gray_top = gray_top.copy()
        prev_gray_bot = gray_bot.copy()

        # ── EMA + Temporal Median ──
        dx_buf.append(raw_dx)
        dy_buf.append(raw_dy)
        buf_dx = float(np.median(list(dx_buf)))
        buf_dy = float(np.median(list(dy_buf)))

        smooth_dx = smooth_dx * (1 - ALPHA) + buf_dx * ALPHA
        smooth_dy = smooth_dy * (1 - ALPHA) + buf_dy * ALPHA

        apply_dx = round(smooth_dx)
        apply_dy = round(smooth_dy)

        # ── Warp Bottom ──
        top_canvas = np.zeros((final_h, width, 3), dtype=np.uint8)
        top_canvas[0:height, :] = frame_top

        M = np.float32([[1, 0, apply_dx], [0, 1, apply_dy]])
        warped_bot = cv2.warpAffine(frame_bot, M, (width, final_h), borderMode=cv2.BORDER_REPLICATE)

        # ── Blend ──
        cut_y = int(apply_dy + (height - apply_dy) * cut_percent / 100)
        cut_y = max(feather_px, min(final_h - feather_px, cut_y))

        final_frame = blend_fn(top_canvas, warped_bot, cut_y, feather_px, final_h, width)

        # ── Motion Overlay (Hand Protection) ──
        if motion_overlay == 1 and bg_subtractor is not None:
            # Detect motion in frame_top
            fg_mask = bg_subtractor.apply(frame_top)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, morph_kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, morph_kernel, iterations=3)
            fg_mask_blur = cv2.GaussianBlur(fg_mask, (21, 21), 0).astype(np.float32) / 255.0

            # Expand mask to canvas size
            full_fg_mask = np.zeros((final_h, width, 1), dtype=np.float32)
            full_fg_mask[0:height, :, 0] = fg_mask_blur
            full_fg_mask_3d = np.repeat(full_fg_mask, 3, axis=2)

            # Composite top frame over the blended frame in moving regions
            final_frame = (top_canvas.astype(np.float32) * full_fg_mask_3d) + (final_frame.astype(np.float32) * (1.0 - full_fg_mask_3d))
            final_frame = np.clip(final_frame, 0, 255).astype(np.uint8)

        # ── Crop to remove black borders ──
        if baseline_dx < 0:
            cropped_frame = final_frame[:, 0:crop_w]
        else:
            cropped_frame = final_frame[:, crop_margin:]

        # ── Write ──
        if out_vid is not None:
            out_vid.write(cropped_frame)

        # ── Preview (ทุก 2 เฟรม เพื่อประหยัด CPU) ──
        if frame_idx % 2 == 0:
            elapsed = time.time() - t_start
            current_fps = frame_idx / elapsed if elapsed > 0 else 0

            preview = cv2.resize(cropped_frame, (0, 0), fx=0.4, fy=0.4)

            # HUD บน preview
            if not live_mode and total_frames > 0:
                pct = frame_idx / total_frames * 100
                eta = (total_frames - frame_idx) / current_fps if current_fps > 0 else 0
                hud = (f"Frame {frame_idx}/{total_frames} ({pct:.0f}%)  "
                       f"| {current_fps:.1f}fps  "
                       f"| ETA {int(eta//60):02d}:{int(eta%60):02d}  "
                       f"| dx={apply_dx} dy={apply_dy}")
            else:
                hud = (f"LIVE  Frame {frame_idx}  "
                       f"| {current_fps:.1f}fps  "
                       f"| dx={apply_dx} dy={apply_dy}")

            cv2.putText(preview, hud, (8, preview.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2)
            cv2.putText(preview, hud, (8, preview.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 220, 80), 1)

            cv2.imshow("StitchLab Output", preview)

        # Check if preview window was closed via 'X' button
        try:
            if frame_idx > 0 and cv2.getWindowProperty("StitchLab Output", cv2.WND_PROP_VISIBLE) < 1:
                print("\n⏹  Stopped by user (window closed).")
                break
        except cv2.error:
            pass

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n⏹  Stopped by user.")
            break

        frame_idx += 1

    # ── Cleanup ──
    cap_top.release()
    cap_bot.release()
    if out_vid is not None:
        out_vid.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\n✅ Done!  {frame_idx} frames in {elapsed:.1f}s  "
          f"({frame_idx/elapsed:.1f} fps avg)")
    if not live_mode:
        print(f"   Output: {Path(output_path).resolve()}")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="StitchLab — Vertical Panorama Stitching"
    )
    parser.add_argument("--top",    default=DEFAULT_TOP,
                        help="Path to top camera video (or camera index for live)")
    parser.add_argument("--bot",    default=DEFAULT_BOT,
                        help="Path to bottom camera video (or camera index for live)")
    parser.add_argument("--output", default=OUTPUT_PATH,
                        help="Output video path")
    parser.add_argument("--live",   action="store_true",
                        help="Live mode: use webcams instead of files")
    parser.add_argument("--top-id", type=int, default=0,
                        help="Camera index for top (live mode)")
    parser.add_argument("--bot-id", type=int, default=1,
                        help="Camera index for bottom (live mode)")
    args = parser.parse_args()

    if args.live:
        print(f"🔴 LIVE MODE  |  Top cam: {args.top_id}  |  Bot cam: {args.bot_id}")
        stitch(args.top_id, args.bot_id, live_mode=True)
    else:
        stitch(args.top, args.bot, output_path=args.output)


if __name__ == "__main__":
    main()
