"""
Stitch3_Pro.py  —  Packing Station Edition
"""

import cv2
import numpy as np
import time
import argparse
import threading
import json
import os

CONFIG_FILE = "panorama_config.json"

C_BLUE_MID   = (211, 133, 33)
C_BLUE_LIGHT = (245, 181, 100)
C_WHITE      = (255, 255, 255)
C_TEXT_PRI   = (245, 240, 235)
C_TEXT_SEC   = (180, 160, 130)
C_CARD       = (75,  55,  28)
C_RED        = (60,  60,  220)

# ==============================================================================
# CAMERA STREAM
# ==============================================================================

class CameraStream:
    def __init__(self, idx, req_w=1920, req_h=1080):
        self.idx = str(idx)
        if self.idx.isdigit(): self.idx = int(self.idx)
        self.cap = cv2.VideoCapture(self.idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.w == 0 or self.h == 0:
            self.w, self.h = req_w, req_h
        self.frame    = None
        self.frame_id = 0
        self.lock     = threading.Lock()
        self.running  = True
        self.thread   = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ok, frm = self.cap.read()
            if ok and frm is not None:
                with self.lock:
                    self.frame    = frm.copy()
                    self.frame_id += 1
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is None: return False, None, 0
            return True, self.frame.copy(), self.frame_id

    def stop(self):
        self.running = False
        if self.thread.is_alive(): self.thread.join(timeout=1.0)
        self.cap.release()


# ==============================================================================
# COLOR MATCH  (cached, overlap-zone-only)
# ==============================================================================

_COLOR_CACHE = {'id': None, 'ts': None}

def color_match_lab_cached(src, target, frame_id, cmatch_on, ov):
    if not cmatch_on:
        return src
    if _COLOR_CACHE['id'] is None or (frame_id - _COLOR_CACHE['id']) > 15:
        ov_safe   = max(10, min(ov, src.shape[1]-10, target.shape[1]-10))
        s_slice   = cv2.cvtColor(src[:, :ov_safe],    cv2.COLOR_BGR2LAB).astype(np.float32)
        t_slice   = cv2.cvtColor(target[:, -ov_safe:], cv2.COLOR_BGR2LAB).astype(np.float32)
        sm, ss    = cv2.meanStdDev(s_slice)
        tm, ts    = cv2.meanStdDev(t_slice)
        ss        = np.clip(ss, 0.001, None)
        _COLOR_CACHE['ts'] = (sm.flatten(), ss.flatten(), tm.flatten(), ts.flatten())
        _COLOR_CACHE['id'] = frame_id
    sm, ss, tm, ts = _COLOR_CACHE['ts']
    s   = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = ((s - sm) * (ts / ss)) + tm
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

def reset_color_cache():
    _COLOR_CACHE['id'] = None
    _COLOR_CACHE['ts'] = None


# ==============================================================================
# AUTO-SEAM  — หาตำแหน่ง seam ที่ "ขัดตาน้อยที่สุด" ในโซน overlap อัตโนมัติ
# ==============================================================================

_SEAM_CACHE = {'seam': None, 'tick': 0}

def find_best_seam(canvas_l, canvas_r, ov_start, ov_end, frame_id):
    """
    คำนวณทุก 30 เฟรม — หาคอลัมน์ในโซน overlap ที่ผลต่างสีระหว่างสองภาพน้อยที่สุด
    เพื่อให้รอยต่อซ่อนตัวอยู่ในบริเวณที่สองภาพ "เหมือนกันมากที่สุด"
    """
    tick = _SEAM_CACHE['tick']
    if _SEAM_CACHE['seam'] is not None and (frame_id - tick) < 30:
        return _SEAM_CACHE['seam']

    # โฟกัส AI เฉพาะช่วงกลางจอ (30% ถึง 70% ของความสูง) ที่มือคุณจะอยู่
    # และข้ามทีละ 4 แถวเพื่อประหยัด CPU หลีกเลี่ยงขอบเลนส์กว้างที่บิดเบี้ยว
    h = canvas_l.shape[0]
    y0, y1 = int(h * 0.3), int(h * 0.7)
    zone_l = canvas_l[y0:y1:4, ov_start:ov_end].astype(np.float32)
    zone_r = canvas_r[y0:y1:4, ov_start:ov_end].astype(np.float32)
    diff   = np.abs(zone_l - zone_r).mean(axis=(0, 2))   # shape: (ov_width,)

    # Smooth ก่อน หา minimum เพื่อกันจุด noise
    kernel = min(15, len(diff) // 4 * 2 + 1)
    if kernel > 3:
        diff = cv2.GaussianBlur(diff.reshape(1, -1),
                                (kernel, 1), 0).flatten()

    best_col = int(np.argmin(diff))
    seam     = ov_start + best_col
    _SEAM_CACHE['seam'] = seam
    _SEAM_CACHE['tick'] = frame_id
    return seam


# ==============================================================================
# STITCH
# ==============================================================================

_ALPHA_CACHE = {}

def stitch_images(img_l, img_r, overlap_px, feather_px,
                  dy_l, dy_r, seam_pct, auto_seam, frame_id,
                  ghost_mode=False):
    h       = img_l.shape[0]
    wl      = img_l.shape[1]
    wr      = img_r.shape[1]
    ov      = max(2, min(overlap_px, wl-2, wr-2))
    final_w = wl + wr - ov

    if dy_l != 0:
        M     = np.float32([[1,0,0],[0,1,dy_l]])
        img_l = cv2.warpAffine(img_l, M, (wl, h), borderMode=cv2.BORDER_REPLICATE)
    if dy_r != 0:
        M     = np.float32([[1,0,0],[0,1,dy_r]])
        img_r = cv2.warpAffine(img_r, M, (wr, h), borderMode=cv2.BORDER_REPLICATE)

    canvas_l = np.zeros((h, final_w, 3), dtype=np.uint8)
    canvas_r = np.zeros((h, final_w, 3), dtype=np.uint8)
    canvas_l[:, :wl] = img_l

    r_start = wl - ov
    r_end   = min(r_start + wr, final_w)
    canvas_r[:, r_start:r_end] = img_r[:, :r_end-r_start]

    if auto_seam:
        seam_x = find_best_seam(canvas_l, canvas_r, r_start, wl, frame_id)
    else:
        seam_x = int(r_start + ov * (seam_pct / 100.0))
        seam_x = max(r_start+1, min(seam_x, wl-1))

    if ghost_mode:
        out = canvas_l.copy()
        out[:, r_start:wl] = cv2.addWeighted(
            canvas_l[:, r_start:wl], 0.5,
            canvas_r[:, r_start:wl], 0.5, 0)
        out[:, wl:] = canvas_r[:, wl:]
        return out, seam_x, r_start, wl

    out = canvas_l.copy()
    out[:, seam_x:] = canvas_r[:, seam_x:]

    fp = max(0, min(feather_px, seam_x - r_start, wl - seam_x - 1))
    if fp > 1:
        x0     = seam_x - fp
        x1     = seam_x + fp
        w_mask = x1 - x0
        if w_mask not in _ALPHA_CACHE:
            t = np.linspace(0.0, np.pi, w_mask, dtype=np.float32)
            _ALPHA_CACHE[w_mask] = (0.5 * (1.0 + np.cos(t))).reshape(1, -1, 1)
        alpha = _ALPHA_CACHE[w_mask]
        sl = canvas_l[:, x0:x1].astype(np.float32)
        sr = canvas_r[:, x0:x1].astype(np.float32)
        out[:, x0:x1] = np.clip(sl*alpha + sr*(1-alpha), 0, 255).astype(np.uint8)

    return out, seam_x, r_start, wl


# ==============================================================================
# GEOMETRY
# ==============================================================================

def apply_perspective(img, pts):
    if pts is None: return img
    h, w = img.shape[:2]
    src  = np.float32(pts)
    dst  = np.float32([[0,0],[w,0],[w,h],[0,h]])
    M    = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h))

def process_geometry(img, fh, fv, rot, zoom, pts, angle=0):
    res = apply_perspective(img, pts)
    if zoom > 1.0:
        h, w   = res.shape[:2]
        nh, nw = int(h/zoom), int(w/zoom)
        y0, x0 = (h-nh)//2, (w-nw)//2
        res    = cv2.resize(res[y0:y0+nh, x0:x0+nw], (w,h),
                            interpolation=cv2.INTER_LINEAR)
    if fh:    res = cv2.flip(res, 1)
    if fv:    res = cv2.flip(res, 0)
    if rot==1: res = cv2.rotate(res, cv2.ROTATE_90_CLOCKWISE)
    elif rot==2: res = cv2.rotate(res, cv2.ROTATE_180)
    elif rot==3: res = cv2.rotate(res, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle != 0:
        h, w = res.shape[:2]
        M    = cv2.getRotationMatrix2D((w/2.0, h/2.0), angle, 1.0)
        res  = cv2.warpAffine(res, M, (w,h), borderMode=cv2.BORDER_REPLICATE)
    return res


# ==============================================================================
# UI HELPERS
# ==============================================================================

def draw_rounded_rect(img, x1, y1, x2, y2, r, color, thickness=-1):
    if thickness == -1:
        cv2.rectangle(img, (x1+r,y1), (x2-r,y2), color, -1)
        cv2.rectangle(img, (x1,y1+r), (x2,y2-r), color, -1)
        for cx,cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
            cv2.circle(img, (cx,cy), r, color, -1)
    else:
        cv2.rectangle(img, (x1+r,y1), (x2-r,y1), color, thickness)
        cv2.rectangle(img, (x1+r,y2), (x2-r,y2), color, thickness)
        cv2.rectangle(img, (x1,y1+r), (x1,y2-r), color, thickness)
        cv2.rectangle(img, (x2,y1+r), (x2,y2-r), color, thickness)
        for cx,cy,a1,a2 in [(x1+r,y1+r,180,270),(x2-r,y1+r,270,360),
                             (x2-r,y2-r,0,90),(x1+r,y2-r,90,180)]:
            cv2.ellipse(img, (cx,cy), (r,r), 0, a1, a2, color, thickness)

def draw_glow_line(img, x1, y1, x2, y2, color, alpha=0.4):
    ov = img.copy()
    cv2.line(ov, (x1,y1), (x2,y2), color, 3)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)
    cv2.line(img, (x1,y1), (x2,y2), color, 1)

def put_text_shadow(img, text, pos, font, scale, color, thickness=1):
    x, y = pos
    cv2.putText(img, text, (x+1,y+1), font, scale, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, pos,       font, scale, color,   thickness,   cv2.LINE_AA)

def section_header(panel, text, y, pw, badge=None):
    F = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel, text, (14,y), F, 0.42, C_BLUE_LIGHT, 1, cv2.LINE_AA)
    if badge:
        (tw,_),_ = cv2.getTextSize(text, F, 0.42, 1)
        bx = 14 + tw + 8
        draw_rounded_rect(panel, bx, y-12, bx+len(badge)*7+10, y+2, 3, C_BLUE_MID)
        cv2.putText(panel, badge, (bx+5,y-1), F, 0.32, C_WHITE, 1, cv2.LINE_AA)
    draw_glow_line(panel, 12, y+5, pw-12, y+5, C_BLUE_MID, 0.5)
    return y + 16


# ==============================================================================
# MAIN APP
# ==============================================================================

class StitchApp:
    PANEL_W = 340

    def __init__(self, left_id, right_id):
        print("Starting Stitch3 Pro...")
        self.cam_l = CameraStream(left_id)
        self.cam_r = CameraStream(right_id)
        self.W     = self.cam_l.w
        self.H     = self.cam_l.h

        self.swap      = False
        self.cmatch    = True
        self.overlap   = int(self.W * 0.1)
        self.feather   = 30
        self.dy_l      = 0
        self.dy_r      = 0
        self.seam_pct  = 50
        self.auto_seam = True        # ใหม่: Auto-Seam เปิดเป็น default
        self.ghost     = True
        self.show_ov   = True

        def_pts = [[0,0],[self.W,0],[self.W,self.H],[0,self.H]]
        self.cams = [
            {'fh':False,'fv':False,'rot':0,'zoom':1.0,'pts':def_pts.copy(),'angle':0},
            {'fh':False,'fv':False,'rot':0,'zoom':1.0,'pts':def_pts.copy(),'angle':0},
        ]

        self.load_config()

        self.edit_cam      = 0
        self.mode          = "SETUP"
        self.seam_x        = 0
        self.ov_start      = 0      # ขอบซ้ายของโซน overlap
        self.ov_end        = 0      # ขอบขวาของโซน overlap
        self.btn_rects     = {}
        self.click_queue   = []
        self.preview_w     = 800
        self.preview_scale = 1.0
        self.recording     = False
        self.out_vid       = None
        self.out_path      = "live_panorama.mp4"
        self.rec_start_t   = 0.0
        self.rec_frames    = 0

        self.last_fid_l    = -1
        self.last_fid_r    = -1
        self._last_frame_t = 0.0      # สำหรับ time-based 5fps throttle
        self.drag_idx      = -1
        self._fps          = 0.0
        self._fps_count    = 0
        self._fps_t        = time.time()

    # ── CONFIG ────────────────────────────────────────────────────────────────

    def save_config(self):
        data = {
            'swap':self.swap, 'cmatch':self.cmatch, 'overlap':self.overlap,
            'feather':self.feather, 'dy_l':self.dy_l, 'dy_r':self.dy_r,
            'seam_pct':self.seam_pct, 'auto_seam':self.auto_seam, 'cams':self.cams,
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(data, f)
        print(f"Config saved -> {CONFIG_FILE}")

    def load_config(self):
        if not os.path.exists(CONFIG_FILE): return
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
            self.swap      = data.get('swap',      self.swap)
            self.cmatch    = data.get('cmatch',    self.cmatch)
            self.overlap   = data.get('overlap',   self.overlap)
            self.feather   = data.get('feather',   self.feather)
            self.dy_l      = data.get('dy_l',      0)
            self.dy_r      = data.get('dy_r',      data.get('dy', 0))
            self.seam_pct  = data.get('seam_pct',  self.seam_pct)
            self.auto_seam = data.get('auto_seam', self.auto_seam)
            self.cams      = data.get('cams',      self.cams)
            for cam in self.cams:
                if 'angle' not in cam: cam['angle'] = 0
            print(f"Config loaded <- {CONFIG_FILE}")
        except Exception as e:
            print(f"[!] Config load failed: {e}")

    # ── FRAME PIPELINE ────────────────────────────────────────────────────────

    def _get_frames(self):
        """
        ปลดล็อค 5fps: ประมวลผลเต็มสปีด (30fps) ทันทีที่กล้องตัวใดตัวหนึ่งมีเฟรมใหม่
        """
        ok_l, fl, fid_l = self.cam_l.read()
        ok_r, fr, fid_r = self.cam_r.read()
        if not ok_l or not ok_r: return None, None
        
        # ป้องกันการคำนวณเฟรมเดิมซ้ำ (ประหยัด CPU 100%)
        if fid_l == self.last_fid_l and fid_r == self.last_fid_r:
            return None, None
        
        self.last_fid_l = fid_l
        self.last_fid_r = fid_r

        gl   = process_geometry(fl, **self.cams[0])
        gr   = process_geometry(fr, **self.cams[1])
        left, right = (gr, gl) if self.swap else (gl, gr)

        mh = min(left.shape[0], right.shape[0])
        mw = min(left.shape[1], right.shape[1])
        left, right = left[:mh,:mw], right[:mh,:mw]

        ov    = max(10, min(self.overlap, left.shape[1]-10, right.shape[1]-10))
        right = color_match_lab_cached(right, left, fid_r, self.cmatch, ov)
        return left, right

    def _build_pano(self, draw_guides=False):
        left, right = self._get_frames()
        if left is None: return None

        ov       = max(1, min(self.overlap, left.shape[1]-10, right.shape[1]-10))
        is_ghost = self.ghost and (self.mode == "SETUP")
        fid      = self.last_fid_r

        pano, seam, ov_start, ov_end = stitch_images(
            left, right, ov, self.feather,
            self.dy_l, self.dy_r, self.seam_pct,
            self.auto_seam, fid,
            ghost_mode=is_ghost)

        self.seam_x   = seam
        self.ov_start = ov_start
        self.ov_end   = ov_end

        if draw_guides:
            h = pano.shape[0]

            if self.show_ov:
                # โซน overlap (สีน้ำเงินโปร่งแสง) ลดความเข้มข้นลง
                overlay = pano.copy()
                cv2.rectangle(overlay, (ov_start,0), (ov_end,h), (200,100,30), -1)
                cv2.addWeighted(overlay, 0.06, pano, 0.94, 0, pano)
                cv2.line(pano, (ov_start,0), (ov_start,h), (160,60,10), 1)
                cv2.line(pano, (ov_end,  0), (ov_end,  h), (160,60,10), 1)

            # Seam line (สีฟ้าสว่าง)
            ov2 = pano.copy()
            cv2.line(ov2, (seam,0), (seam,h), (255,200,60), 4)
            cv2.addWeighted(ov2, 0.45, pano, 0.55, 0, pano)
            cv2.line(pano, (seam,0), (seam,h), (255,220,80), 1)

            # Feather boundary lines
            fp = self.feather
            if fp > 0:
                for ex in [max(0,seam-fp), min(pano.shape[1]-1,seam+fp)]:
                    cv2.line(pano, (ex,0),(ex,h),(180,140,50),1,cv2.LINE_AA)

            if self.show_ov:
                # Label ขนาด overlap บนภาพ
                ov_px  = ov_end - ov_start
                ov_pct = ov_px / pano.shape[1] * 100
                lbl    = f"OV={ov_px}px ({ov_pct:.0f}%)"
                lx     = ov_start + 4
                cv2.putText(pano, lbl, (lx, h-8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.40, (220,200,80), 1, cv2.LINE_AA)
                            
            # Label auto/manual seam
            mode_lbl = "AUTO-SEAM" if (self.auto_seam and not is_ghost) else "MANUAL SEAM"
            cv2.putText(pano, mode_lbl, (seam+4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,220,80), 1, cv2.LINE_AA)

        return pano

    # ── MOUSE ─────────────────────────────────────────────────────────────────

    def _on_mouse(self, event, x, y, flags, param):
        if self.mode == "SETUP":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.click_queue.append((x, y))
        elif self.mode == "CALIBRATE":
            pts    = self.cams[self.edit_cam]['pts']
            sc     = self.preview_scale
            ox, oy = int(x/sc), int(y/sc)
            if event == cv2.EVENT_LBUTTONDOWN:
                min_d = 9999
                for i, p in enumerate(pts):
                    d = (p[0]-ox)**2 + (p[1]-oy)**2
                    if d < 4000 and d < min_d:
                        min_d = d; self.drag_idx = i
            elif event == cv2.EVENT_MOUSEMOVE and self.drag_idx != -1:
                pts[self.drag_idx] = [max(0,min(self.W,ox)), max(0,min(self.H,oy))]
            elif event == cv2.EVENT_LBUTTONUP:
                self.drag_idx = -1

    def _check_clicks(self):
        if not self.click_queue: return
        cx, cy = self.click_queue.pop(0)
        for name, (bx,by,bw,bh) in self.btn_rects.items():
            if bx <= cx <= bx+bw and by <= cy <= by+bh:
                self._handle(name); return

    def _handle(self, name):
        cam = self.cams[self.edit_cam]
        if   name == 'cam_l':     self.edit_cam  = 0
        elif name == 'cam_r':     self.edit_cam  = 1
        elif name == 'swap':      self.swap       = not self.swap
        elif name == 'flip_h':    cam['fh']       = not cam['fh']
        elif name == 'flip_v':    cam['fv']       = not cam['fv']
        elif name == 'rot_cw':    cam['rot']      = (cam['rot']+1)%4
        elif name == 'rot_ccw':   cam['rot']      = (cam['rot']-1)%4
        elif name == 'zoom_in':   cam['zoom']     = min(4.0, round(cam['zoom']+0.1,1))
        elif name == 'zoom_out':  cam['zoom']     = max(1.0, round(cam['zoom']-0.1,1))
        elif name == 'color':
            self.cmatch = not self.cmatch
            reset_color_cache()
        elif name == 'ghost':     self.ghost      = not self.ghost
        elif name == 'auto_seam':
            self.auto_seam = not self.auto_seam
            _SEAM_CACHE['seam'] = None
        elif name == 'show_ov':   self.show_ov = not self.show_ov
        elif name == 'calibrate': self.mode       = "CALIBRATE"
        elif name == 'go_live':
            self.save_config(); self.mode = "LIVE"

    # ── PANEL ─────────────────────────────────────────────────────────────────

    def _draw_panel(self, panel_h):
        PW = self.PANEL_W
        F  = cv2.FONT_HERSHEY_SIMPLEX

        panel = np.zeros((panel_h, PW, 3), dtype=np.uint8)
        for row in range(panel_h):
            t = row / panel_h
            panel[row, :] = (int(38+t*8), int(22+t*4), int(10+t*3))

        self.btn_rects.clear()

        def btn(name, text, x, y, bw, bh, active=False, accent=None):
            bg = accent if (active and accent) else (
                 (C_BLUE_MID[0]//3, C_BLUE_MID[1]//3, C_BLUE_MID[2]//3) if active
                 else (55, 42, 22))
            draw_rounded_rect(panel, x, y, x+bw, y+bh, 5, bg)
            border = accent if (accent and active) else (
                     C_BLUE_LIGHT if active else (90, 72, 48))
            draw_rounded_rect(panel, x, y, x+bw, y+bh, 5, border, 1)
            (tw,th),_ = cv2.getTextSize(text, F, 0.43, 1)
            tx, ty    = x+(bw-tw)//2, y+(bh+th)//2
            cv2.putText(panel, text, (tx+1,ty+1), F, 0.43, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(panel, text, (tx,ty),     F, 0.43,
                        C_WHITE if active else C_TEXT_SEC, 1, cv2.LINE_AA)
            if active:
                iw = bw//3
                cv2.line(panel, (x+(bw-iw)//2, y+bh-2),
                         (x+(bw+iw)//2, y+bh-2), C_BLUE_LIGHT, 2)
            self.btn_rects[name] = (self.preview_w+x, y, bw, bh)

        def stat_badge(label, value, x, y, w, h, color=None):
            draw_rounded_rect(panel, x, y, x+w, y+h, 4, color or C_CARD)
            cv2.putText(panel, label, (x+6, y+12), F, 0.33, C_TEXT_SEC, 1, cv2.LINE_AA)
            (vw,_),_ = cv2.getTextSize(value, F, 0.52, 1)
            cv2.putText(panel, value, (x+(w-vw)//2, y+h-6),
                        F, 0.52, C_TEXT_PRI, 1, cv2.LINE_AA)

        PAD = 12
        y   = 14

        # Title
        draw_rounded_rect(panel, PAD, y, PW-PAD, y+32, 6,
                          (C_BLUE_MID[0]//2, C_BLUE_MID[1]//2, C_BLUE_MID[2]//2))
        put_text_shadow(panel, "STITCH3  PRO", (PAD+10, y+22), F, 0.58, C_WHITE, 1)
        put_text_shadow(panel, "SETUP",        (PW-62,  y+22), F, 0.42, C_BLUE_LIGHT, 1)
        y += 42

        # Section 1: Camera
        y = section_header(panel, "CAMERA SELECT", y, PW); y += 6
        half = (PW - 2*PAD - 8) // 2
        btn('cam_l', '<< LEFT',  PAD,        y, half, 34, self.edit_cam==0, C_BLUE_MID)
        btn('cam_r', 'RIGHT >>', PAD+half+8, y, half, 34, self.edit_cam==1, C_BLUE_MID)
        y += 44

        cname = "LEFT CAMERA" if self.edit_cam==0 else "RIGHT CAMERA"
        cv2.putText(panel, f"Editing: {cname}", (PAD, y),
                    F, 0.40, C_BLUE_LIGHT, 1, cv2.LINE_AA)
        y += 18

        cam = self.cams[self.edit_cam]
        btn('flip_h', f"H-Flip {'ON' if cam['fh'] else 'OFF'}", PAD,        y, half, 32, cam['fh'])
        btn('flip_v', f"V-Flip {'ON' if cam['fv'] else 'OFF'}", PAD+half+8, y, half, 32, cam['fv'])
        y += 40

        btn('rot_ccw', '< CCW', PAD,       y, 72, 32)
        rot_lbl = f"{cam['rot']*90} deg"
        (rw,_),_ = cv2.getTextSize(rot_lbl, F, 0.5, 1)
        cv2.putText(panel, rot_lbl, (PAD+72+(PW-2*PAD-144-rw)//2, y+20),
                    F, 0.50, C_TEXT_PRI, 1, cv2.LINE_AA)
        btn('rot_cw',  'CW >', PW-PAD-72, y, 72, 32)
        y += 40

        btn('zoom_out', ' - ', PAD,       y, 44, 32)
        zoom_lbl = f"ZOOM  {cam['zoom']:.1f}x"
        (zw,_),_ = cv2.getTextSize(zoom_lbl, F, 0.44, 1)
        cv2.putText(panel, zoom_lbl, (PAD+44+(PW-2*PAD-88-zw)//2, y+20),
                    F, 0.44, C_TEXT_PRI, 1, cv2.LINE_AA)
        btn('zoom_in',  ' + ', PW-PAD-44, y, 44, 32)
        y += 40

        draw_rounded_rect(panel, PAD, y, PW-PAD, y+34, 6, (20,80,180))
        draw_rounded_rect(panel, PAD, y, PW-PAD, y+34, 6, (40,120,220), 1)
        cal_lbl = "[+] PERSPECTIVE CALIBRATE"
        (lw,lh),_ = cv2.getTextSize(cal_lbl, F, 0.43, 1)
        cv2.putText(panel, cal_lbl, (PAD+(PW-2*PAD-lw)//2, y+(34+lh)//2),
                    F, 0.43, C_WHITE, 1, cv2.LINE_AA)
        self.btn_rects['calibrate'] = (self.preview_w+PAD, y, PW-2*PAD, 34)
        y += 46

        # Section 2: Global
        y = section_header(panel, "GLOBAL SETTINGS", y, PW); y += 6
        btn('swap',      f"SWAP L<>R  {'ON' if self.swap else 'OFF'}",
            PAD, y, PW-2*PAD, 32, self.swap,      (160,60,10));  y += 40
        btn('color',     f"Color Match  {'ON' if self.cmatch else 'OFF'}",
            PAD, y, PW-2*PAD, 32, self.cmatch,    C_BLUE_MID);   y += 40
        btn('ghost',     f"Ghost Overlay  {'ON' if self.ghost else 'OFF'}",
            PAD, y, PW-2*PAD, 32, self.ghost,     (60,120,160)); y += 40

        # Auto-Seam toggle
        as_color = (30, 140, 80) if self.auto_seam else None
        btn('auto_seam', f"Auto-Seam  {'ON' if self.auto_seam else 'OFF (Manual)'}",
            PAD, y, PW-2*PAD, 32, self.auto_seam, as_color);    y += 40
            
        btn('show_ov', f"Show OV Zone  {'ON' if self.show_ov else 'OFF'}",
            PAD, y, PW-2*PAD, 32, self.show_ov, (180,80,10)); y += 44

        # Section 3: Stats
        y = section_header(panel, "LIVE STATS", y, PW, badge="1080p 30FPS"); y += 8
        sw = (PW - 2*PAD - 8) // 3
        stat_badge("FPS",     f"{self._fps:.1f}",   PAD,          y, sw, 38)
        stat_badge("OVERLAP", str(self.overlap),     PAD+sw+4,     y, sw, 38, (50,40,22))
        stat_badge("FEATHER", str(self.feather),     PAD+2*(sw+4), y, sw, 38, (50,40,22))
        y += 50

        # Tips strip
        draw_rounded_rect(panel, PAD, y, PW-PAD, y+28, 4, (35,28,14))
        tips = [
            "W/S = vertical align  |  A/D = overlap",
            "Auto-Seam: finds best blend point every 30fr",
            "Color Match resets when toggled",
        ]
        tip = tips[int(time.time()*0.5) % len(tips)]
        cv2.putText(panel, tip, (PAD+6, y+18), F, 0.35, C_TEXT_SEC, 1, cv2.LINE_AA)
        y += 36

        # Launch button
        btn_y = panel_h - 66
        draw_rounded_rect(panel, PAD, btn_y, PW-PAD, btn_y+52, 8,
                          (C_BLUE_MID[0], C_BLUE_MID[1]//2+20, 15))
        draw_rounded_rect(panel, PAD+2, btn_y+2, PW-PAD-2, btn_y+18, 6,
                          (min(255,C_BLUE_MID[0]+40), C_BLUE_MID[1]//2+40, 30))
        lv_lbl = ">>  START LIVE STREAM"
        (lw,lh),_ = cv2.getTextSize(lv_lbl, F, 0.60, 2)
        cv2.putText(panel, lv_lbl, (PAD+(PW-2*PAD-lw)//2+1, btn_y+33),
                    F, 0.60, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(panel, lv_lbl, (PAD+(PW-2*PAD-lw)//2,   btn_y+32),
                    F, 0.60, C_WHITE, 2, cv2.LINE_AA)
        self.btn_rects['go_live'] = (self.preview_w+PAD, btn_y, PW-2*PAD, 52)

        return panel

    # ── CALIBRATE MODE ────────────────────────────────────────────────────────

    def _run_calibrate(self, WIN):
        print("Calibrate — drag corners, press Q to return")
        while self.mode == "CALIBRATE":
            src    = self.cam_l if self.edit_cam==0 else self.cam_r
            ok, frm, _ = src.read()
            if not ok or frm is None:
                time.sleep(0.05); continue

            h, w = frm.shape[:2]
            scale = min(1.0, 1000/w, 800/h)
            self.preview_scale = scale
            pts   = self.cams[self.edit_cam]['pts']

            poly = np.array(pts, dtype=np.int32)
            mask = np.zeros((h,w), np.uint8)
            cv2.fillPoly(mask, [poly], 255)
            frm[mask==0] = (frm[mask==0] * 0.45).astype(np.uint8)

            for i in range(4):
                cv2.line(frm, tuple(pts[i]), tuple(pts[(i+1)%4]), (255,200,50), 2, cv2.LINE_AA)
            for i, p in enumerate(pts):
                cv2.circle(frm, tuple(p), 18, (0,0,0), 3)
                cv2.circle(frm, tuple(p), 18, (255,200,50), 2)
                cv2.circle(frm, tuple(p), 6,  (255,255,255), -1)
                cv2.putText(frm, ["TL","TR","BR","BL"][i], (p[0]+22,p[1]+6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,220,80), 2, cv2.LINE_AA)

            bar_h = 52
            cv2.rectangle(frm, (0,0), (w,bar_h),
                          (C_BLUE_MID[0]//2, C_BLUE_MID[1]//2, 10), -1)
            cname = "LEFT" if self.edit_cam==0 else "RIGHT"
            cv2.putText(frm, f"PERSPECTIVE  |  {cname} CAMERA",
                        (18,32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, C_WHITE, 2, cv2.LINE_AA)
            cv2.putText(frm, "Drag corners to correct keystone    [Q] Save & Return",
                        (18,46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_BLUE_LIGHT, 1, cv2.LINE_AA)

            cv2.imshow(WIN, cv2.resize(frm, (int(w*scale), int(h*scale))))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q'), 13):
                self.save_config(); self.mode = "SETUP"; break
            try:
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    self.mode = "QUIT"; break
            except cv2.error:
                self.mode = "QUIT"; break

    # ── SETUP MODE ────────────────────────────────────────────────────────────

    def _run_setup(self):
        WIN = "Stitch3 Pro - Setup"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN, self._on_mouse)

        cv2.createTrackbar("Overlap",     WIN, self.overlap,           self.W, lambda v: None)
        cv2.createTrackbar("Seam %",      WIN, self.seam_pct,          100,    lambda v: None)
        cv2.createTrackbar("DY Left",     WIN, self.dy_l+50,           100,    lambda v: None)
        cv2.createTrackbar("DY Right",    WIN, self.dy_r+50,           100,    lambda v: None)
        cv2.createTrackbar("Angle Left",  WIN, self.cams[0]['angle']+45, 90,   lambda v: None)
        cv2.createTrackbar("Angle Right", WIN, self.cams[1]['angle']+45, 90,   lambda v: None)
        cv2.createTrackbar("Feather",     WIN, self.feather,           150,    lambda v: None)

        print("\nSetup ready. Click START LIVE STREAM when done.")
        pano = None

        while self.mode == "SETUP":
            self.overlap            = max(1, cv2.getTrackbarPos("Overlap",     WIN))
            self.seam_pct           =        cv2.getTrackbarPos("Seam %",      WIN)
            self.dy_l               =        cv2.getTrackbarPos("DY Left",     WIN) - 50
            self.dy_r               =        cv2.getTrackbarPos("DY Right",    WIN) - 50
            self.cams[0]['angle']   =        cv2.getTrackbarPos("Angle Left",  WIN) - 45
            self.cams[1]['angle']   =        cv2.getTrackbarPos("Angle Right", WIN) - 45
            self.feather            =        cv2.getTrackbarPos("Feather",     WIN)

            new_pano = self._build_pano(draw_guides=True)
            if new_pano is not None:
                pano = new_pano
                self._fps_count += 1
                elapsed = time.time() - self._fps_t
                if elapsed >= 1.0:
                    self._fps       = self._fps_count / elapsed
                    self._fps_count = 0
                    self._fps_t     = time.time()

            if pano is None:
                time.sleep(0.02); continue

            ph, pw = pano.shape[:2]
            scale  = min(1.0, (1280-self.PANEL_W)/pw, 750/ph)
            prev_w = int(pw*scale)
            prev_h = int(ph*scale)
            pano_r = cv2.resize(pano, (prev_w, prev_h))

            self.preview_w    = prev_w
            self.preview_scale = scale

            panel_h = max(prev_h, 700)
            panel   = self._draw_panel(panel_h)
            disp_h  = max(prev_h, panel_h)
            disp    = np.full((disp_h, prev_w+self.PANEL_W, 3), (38,22,10), dtype=np.uint8)
            disp[:prev_h, :prev_w]        = pano_r
            disp[:panel.shape[0], prev_w:] = panel

            for i in range(2):
                cv2.line(disp, (prev_w+i,0), (prev_w+i,disp_h),
                         (C_BLUE_MID[0]//2, C_BLUE_MID[1]//2, 15), 1)

            bar_y = disp_h - 22
            cv2.rectangle(disp, (0,bar_y), (prev_w,disp_h), (25,18,8), -1)
            seam_mode = "AUTO" if self.auto_seam else f"MANUAL {self.seam_pct}%"
            ghost_txt = "GHOST " if self.ghost else ""
            status = (f"  OV:{self.overlap}px  Seam:{seam_mode}  "
                      f"Feather:{self.feather}  FPS:{self._fps:.1f}  {ghost_txt}")
            cv2.putText(disp, status, (8,bar_y+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_BLUE_LIGHT, 1, cv2.LINE_AA)

            cv2.imshow(WIN, disp)
            self._check_clicks()

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                self.save_config(); self.mode = "QUIT"; break
            elif key == ord('a'):
                self.overlap = max(1, self.overlap-1)
                cv2.setTrackbarPos("Overlap", WIN, self.overlap)
            elif key == ord('d'):
                self.overlap = min(self.W, self.overlap+1)
                cv2.setTrackbarPos("Overlap", WIN, self.overlap)
            elif key == ord('w'):
                if self.edit_cam==0:
                    self.dy_l -= 1; cv2.setTrackbarPos("DY Left",  WIN, self.dy_l+50)
                else:
                    self.dy_r -= 1; cv2.setTrackbarPos("DY Right", WIN, self.dy_r+50)
            elif key == ord('s'):
                if self.edit_cam==0:
                    self.dy_l += 1; cv2.setTrackbarPos("DY Left",  WIN, self.dy_l+50)
                else:
                    self.dy_r += 1; cv2.setTrackbarPos("DY Right", WIN, self.dy_r+50)
            elif key == ord('c'):
                reset_color_cache()
                print("Color cache reset")
            try:
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    self.save_config(); self.mode = "QUIT"; break
            except cv2.error:
                self.save_config(); self.mode = "QUIT"; break

    # ── LIVE MODE ─────────────────────────────────────────────────────────────

    def _run_live(self):
        WIN = "Stitch3 Pro 1080p - LIVE"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        print("\n" + "-"*50)
        print("  LIVE MODE  (UNLOCKED 30fps)")
        print("  Q/ESC = stop  |  R = record  |  C = reset color")
        print("-"*50)

        t0, cnt = time.time(), 0

        while self.mode == "LIVE":
            pano = self._build_pano(draw_guides=False)
            if pano is None:
                time.sleep(0.01); continue

            cnt    += 1
            elapsed = time.time() - t0
            fps_now = cnt / elapsed if elapsed > 0 else 0

            if self.recording:
                if self.out_vid is None:
                    h, w = pano.shape[:2]
                    self.out_vid    = cv2.VideoWriter(
                        self.out_path,
                        cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w,h))
                    self.rec_start_t = time.time()
                    self.rec_frames  = 0
                    print(f"REC -> {self.out_path}")
                self.out_vid.write(pano)
                self.rec_frames += 1

            # HUD
            hud_h = 44
            cv2.rectangle(pano, (0,0), (pano.shape[1],hud_h),
                          (C_BLUE_MID[0]//3, C_BLUE_MID[1]//3, 8), -1)
            cv2.putText(pano, "LIVE", (14,28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, C_BLUE_LIGHT, 2, cv2.LINE_AA)
            cv2.putText(pano, f"{fps_now:.1f}fps", (80,28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_TEXT_PRI, 1, cv2.LINE_AA)

            if self.recording:
                rec_elapsed = time.time() - self.rec_start_t
                m, s  = divmod(int(rec_elapsed), 60)
                rec_x = pano.shape[1] - 160
                cv2.circle(pano, (rec_x,22), 7, C_RED, -1)
                cv2.putText(pano, f"REC {m:02d}:{s:02d}", (rec_x+14,28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100,100,255), 1, cv2.LINE_AA)
            else:
                cv2.putText(pano, "[R] Record", (pano.shape[1]-130,28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_TEXT_SEC, 1, cv2.LINE_AA)

            cv2.imshow(WIN, pano)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')): break
            elif key == ord('r'):
                self.recording = not self.recording
                if not self.recording and self.out_vid:
                    self.out_vid.release(); self.out_vid = None
                    print(f"[OK] Saved: {self.out_path}  ({self.rec_frames} frames)")
            elif key == ord('c'):
                reset_color_cache()
                print("Color cache reset")
            try:
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1: break
            except cv2.error: break

    # ── RUN ───────────────────────────────────────────────────────────────────

    def run(self):
        try:
            while self.mode in ("SETUP", "CALIBRATE"):
                if self.mode == "SETUP":
                    self._run_setup()
                elif self.mode == "CALIBRATE":
                    self._run_calibrate("Stitch3 Pro - Setup")
            if self.mode == "LIVE":
                cv2.destroyAllWindows()
                self._run_live()
        finally:
            self.cam_l.stop()
            self.cam_r.stop()
            if self.out_vid:
                self.out_vid.release()
                print(f"[OK] Saved: {self.out_path}")
            cv2.destroyAllWindows()
            print("Done.")


# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--left",   type=str, default="0")
    parser.add_argument("--right",  type=str, default="1")
    parser.add_argument("--output", default="live_panorama.mp4")
    args = parser.parse_args()

    app = StitchApp(args.left, args.right)
    app.out_path = args.output
    app.run()