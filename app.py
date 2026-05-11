import os
import urllib.request
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk

import cv2
import numpy as np
import mediapipe as mp
import time
from PIL import Image, ImageTk


GREEN_BG = (0, 255, 0)
FACE_LANDMARKER_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int


@dataclass
class RegionState:
    base_box: Box | None = None
    offset_x: int = 0
    offset_y: int = 0
    scale: float = 1.0
    current_box: Box | None = None


class EyeMouthMaskerApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title('Eye & Mouth Masker')
        self.root.configure(bg='black')
        self.root.update_idletasks()
        try:
            self.root.state('zoomed')
        except tk.TclError:
            self.root.geometry(f'{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0')
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

        self.capture = None
        self.running = False
        self.after_id = None
        self.photo = None
        self.current_frame_size = (640, 480)
        self.preview_display_size = (640, 480)
        self.detection_max_dim = 480
        self.frame_count = 0
        self.detection_interval = 1  # run detection every N frames (1 = every frame)
        self.landmark_smooth_alpha = 0.65  # EMA weight for smoothing detected boxes
        self.last_feature_boxes = None
        self.active_drag_region = None
        self.drag_pointer_offset = (0, 0)
        self.blur_strength = tk.IntVar(value=15)
        self.status_text = tk.StringVar(value='Ready. Click Start Webcam.')

        self.feature_regions = {
            'leftEye': RegionState(),
            'rightEye': RegionState(),
            'mouth': RegionState(),
        }

        self.face_landmarker = self._create_face_landmarker()

        self.face_cascade = self._load_cascade('haarcascade_frontalface_default.xml')
        self.eye_cascade = self._load_cascade('haarcascade_eye_tree_eyeglasses.xml')
        self.mouth_cascade = self._load_cascade('haarcascade_smile.xml')

        self._build_ui()

    def _load_cascade(self, filename: str) -> cv2.CascadeClassifier:
        path = os.path.join(cv2.data.haarcascades, filename)  # type: ignore[attr-defined]
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            raise RuntimeError(f'Failed to load cascade: {filename}')
        return cascade

    def _model_path(self) -> str:
        assets_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(assets_dir, exist_ok=True)
        return os.path.join(assets_dir, 'face_landmarker.task')

    def _ensure_face_landmarker_model(self) -> str:
        model_path = self._model_path()
        if not os.path.exists(model_path):
            self.set_status('Downloading Face Mesh model...')
            urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, model_path)
        return model_path

    def _create_face_landmarker(self):
        try:
            model_path = self._ensure_face_landmarker_model()
            base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            return mp.tasks.vision.FaceLandmarker.create_from_options(options)
        except Exception as error:
            self.set_status(f'Face Mesh unavailable, using fallback detection: {error}')
            return None

    def _build_ui(self) -> None:
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        frame_shell = tk.Frame(self.root, bg='black', bd=0, highlightthickness=0)
        frame_shell.grid(row=0, column=0, sticky='nsew')

        self.preview = tk.Label(frame_shell, bg='black')
        self.preview.pack(fill='both', expand=True)
        self.preview.bind('<ButtonPress-1>', self._on_preview_mouse_down)
        self.preview.bind('<B1-Motion>', self._on_preview_mouse_move)
        self.preview.bind('<ButtonRelease-1>', self._on_preview_mouse_up)
        self.preview.bind('<Motion>', self._on_preview_hover)
        self.preview.bind('<MouseWheel>', self._on_preview_mouse_wheel)
        self.preview.bind('<Button-4>', self._on_preview_mouse_wheel)
        self.preview.bind('<Button-5>', self._on_preview_mouse_wheel)

        controls = tk.Frame(self.root, bg='#0b1220', bd=0, highlightthickness=1, highlightbackground='#243244')
        controls.grid(row=1, column=0, sticky='ew')

        controls_inner = tk.Frame(controls, bg='#0b1220')
        controls_inner.pack(fill='x', padx=18, pady=14)

        self.start_button = ttk.Button(controls_inner, text='Start Webcam', command=self.start_camera)
        self.start_button.grid(row=0, column=0, padx=(0, 10), sticky='w')

        self.stop_button = ttk.Button(controls_inner, text='Stop Webcam', command=self.stop_camera, state='disabled')
        self.stop_button.grid(row=0, column=1, padx=(0, 20), sticky='w')

        slider_frame = tk.Frame(controls_inner, bg='#0b1220')
        slider_frame.grid(row=0, column=2, sticky='ew')
        controls_inner.columnconfigure(2, weight=1)

        slider_label_row = tk.Frame(slider_frame, bg='#0b1220')
        slider_label_row.pack(fill='x')

        slider_label = tk.Label(
            slider_label_row,
            text='Mask Feather',
            fg='white',
            bg='#0b1220',
            font=('Segoe UI', 10, 'bold'),
        )
        slider_label.pack(side='left')

        self.slider_value_label = tk.Label(
            slider_label_row,
            text='15',
            fg='#94a3b8',
            bg='#0b1220',
            font=('Segoe UI', 10),
        )
        self.slider_value_label.pack(side='right')

        slider = tk.Scale(
            slider_frame,
            from_=0,
            to=50,
            orient='horizontal',
            variable=self.blur_strength,
            command=self._on_slider_change,
            bg='#0b1220',
            fg='white',
            troughcolor='#334155',
            highlightthickness=0,
            relief='flat',
        )
        slider.pack(fill='x', pady=(6, 0))

    def _on_slider_change(self, _value: str) -> None:
        self.slider_value_label.config(text=str(self.blur_strength.get()))

    def _clamp_scale(self, value: float) -> float:
        return max(0.35, min(3.0, value))

    def set_status(self, message: str) -> None:
        self.status_text.set(message)

    def start_camera(self) -> None:
        if self.running:
            return

        self.set_status('Requesting camera access...')

        backend = cv2.CAP_DSHOW if os.name == 'nt' else 0
        capture = cv2.VideoCapture(0, backend)
        # Try a set of preferred resolutions (high -> low). Many cameras will report
        # the closest supported resolution rather than exactly what we request.
        # Prefer moderate resolutions for a balance of quality and speed.
        preferred_res = [(1280, 720), (640, 480)]
        chosen = None
        for (w, h) in preferred_res:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            capture.set(cv2.CAP_PROP_FPS, 30)
            # read back what the capture accepted
            actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if actual_w > 0 and actual_h > 0:
                chosen = (actual_w, actual_h)
                break

        if chosen is None:
            actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
            chosen = (actual_w, actual_h)

        # store the actual frame size we'll be working with
        self.current_frame_size = chosen

        if not capture.isOpened():
            capture.release()
            messagebox.showerror('Camera error', 'Could not open the webcam.')
            self.set_status('Camera error: could not open the webcam.')
            return

        self.capture = capture
        self.running = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.set_status('Camera running - tracking eyes and mouth.')
        self._process_frame()

    def stop_camera(self) -> None:
        self.running = False

        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        self.active_drag_region = None
        self.preview.config(cursor='arrow')

        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.set_status('Camera stopped.')
        self._show_frame(self._make_blank_frame(640, 480))

    def on_close(self) -> None:
        self.stop_camera()
        if self.face_landmarker is not None:
            self.face_landmarker.close()
        self.root.destroy()

    def _process_frame(self) -> None:
        if not self.running or self.capture is None:
            return

        ok, frame = self.capture.read()
        if not ok:
            self.set_status('Camera error: failed to read a frame.')
            self.after_id = self.root.after(15, self._process_frame)
            return
        # increment frame counter for detection skipping
        self.frame_count += 1

        frame = cv2.flip(frame, 1)
        masked = self._build_masked_frame(frame)
        self._show_frame(masked)
        self.after_id = self.root.after(15, self._process_frame)

    def _make_blank_frame(self, width: int, height: int):
        frame = np.zeros((height, width, 3), dtype='uint8')
        frame[:, :] = GREEN_BG
        return frame

    def _show_frame(self, frame) -> None:
        self.current_frame_size = (frame.shape[1], frame.shape[0])
        widget_width = max(1, self.preview.winfo_width())
        widget_height = max(1, self.preview.winfo_height())

        scale = min(widget_width / frame.shape[1], widget_height / frame.shape[0])
        display_width = max(1, int(frame.shape[1] * scale))
        display_height = max(1, int(frame.shape[0] * scale))
        self.preview_display_size = (display_width, display_height)

        display_frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((widget_height, widget_width, 3), dtype='uint8')
        canvas[:, :] = GREEN_BG
        offset_x = max(0, (widget_width - display_width) // 2)
        offset_y = max(0, (widget_height - display_height) // 2)
        canvas[offset_y:offset_y + display_height, offset_x:offset_x + display_width] = display_frame

        display = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(display)
        self.photo = photo
        self.preview.config(image=photo)

    def _preview_image_offset(self):
        widget_width = self.preview.winfo_width() or self.preview_display_size[0]
        widget_height = self.preview.winfo_height() or self.preview_display_size[1]
        image_width, image_height = self.preview_display_size
        offset_x = max(0, (widget_width - image_width) // 2)
        offset_y = max(0, (widget_height - image_height) // 2)
        return offset_x, offset_y

    def _get_preview_pointer(self, event):
        offset_x, offset_y = self._preview_image_offset()
        display_width, display_height = self.preview_display_size
        frame_width, frame_height = self.current_frame_size

        if display_width <= 0 or display_height <= 0:
            return {'x': event.x - offset_x, 'y': event.y - offset_y}

        image_x = (event.x - offset_x) * frame_width / display_width
        image_y = (event.y - offset_y) * frame_height / display_height
        return {'x': image_x, 'y': image_y}

    def _build_masked_frame(self, frame):
        height, width = frame.shape[:2]
        background = self._make_blank_frame(width, height)
        source_frame = frame.copy()

        feature_boxes = []
        if self.face_landmarker is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # For speed, downscale the frame sent to MediaPipe detection while
            # keeping landmark normalization so coordinates map back to the
            # original frame correctly. This generally improves FPS while
            # preserving landmark accuracy.
            max_dim = getattr(self, 'detection_max_dim', 480)
            h, w = rgb_frame.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                small_w = max(1, int(w * scale))
                small_h = max(1, int(h * scale))
                small_rgb = cv2.resize(rgb_frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
            else:
                small_rgb = rgb_frame

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small_rgb)

            do_detect = (self.frame_count % self.detection_interval) == 0
            if do_detect:
                # run detection on the small image and scale boxes back to original size
                if hasattr(self.face_landmarker, 'detect_for_video'):
                    mesh_results = self.face_landmarker.detect_for_video(mp_image, int(time.time() * 1000))
                else:
                    mesh_results = self.face_landmarker.detect(mp_image)

                if mesh_results and getattr(mesh_results, 'face_landmarks', None):
                    # feature boxes are relative to the small image; scale them up
                    small_h, small_w = small_rgb.shape[:2]
                    # use a uniform scale to avoid stretching boxes
                    uniform_scale = (width / small_w) if small_w > 0 else 1.0
                    boxes_small = self._feature_boxes_from_landmarks(mesh_results.face_landmarks[0], small_w, small_h)
                    scaled = {}
                    for k, b in boxes_small.items():
                        scaled[k] = Box(int(b.x * uniform_scale), int(b.y * uniform_scale), max(1, int(b.w * uniform_scale)), max(1, int(b.h * uniform_scale)))
                    feature_boxes = scaled
                    self.last_feature_boxes = feature_boxes
            else:
                # reuse most recent detection if available
                if self.last_feature_boxes is not None:
                    feature_boxes = self.last_feature_boxes
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            face_boxes = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(120, 120),
            )

            if len(face_boxes) == 0:
                return background

            face = self._largest_box(face_boxes)
            eyes = self._detect_eye_boxes(gray, face)
            mouth = self._detect_mouth_box(gray, face)

            feature_boxes = {
                'leftEye': eyes[0],
                'rightEye': eyes[1],
            }
            if mouth is not None:
                feature_boxes['mouth'] = mouth

        if not feature_boxes:
            return background

        self._sync_feature_regions(feature_boxes, width, height)

        output = background.copy()
        for key in ('mouth', 'leftEye', 'rightEye'):
            region = self.feature_regions[key]
            if region.base_box is None:
                continue
            self._draw_region_sprite(output, source_frame, region)

        return output

    def _feature_boxes_from_landmarks(self, landmarks, width: int, height: int):
        left_eye_idx = [33, 160, 158, 133, 153, 144, 159, 145]
        right_eye_idx = [362, 385, 387, 263, 373, 380, 386, 374]
        mouth_idx = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

        landmark_points = landmarks.landmark if hasattr(landmarks, 'landmark') else landmarks

        left_eye = [self._landmark_to_point(landmark_points[index], width, height) for index in left_eye_idx]
        right_eye = [self._landmark_to_point(landmark_points[index], width, height) for index in right_eye_idx]
        mouth = [self._landmark_to_point(landmark_points[index], width, height) for index in mouth_idx]

        # Use smaller paddings (in pixels) so masks don't accidentally include nearby features
        return {
            'leftEye': self._clamp_box(self._box_from_points(left_eye, 10), width, height),
            'rightEye': self._clamp_box(self._box_from_points(right_eye, 10), width, height),
            'mouth': self._clamp_box(self._box_from_points(mouth, 12), width, height),
        }

    def _largest_box(self, boxes) -> Box:
        x, y, w, h = max(boxes, key=lambda item: item[2] * item[3])
        return Box(int(x), int(y), int(w), int(h))

    def _landmark_to_point(self, landmark, width: int, height: int):
        return {
            'x': int(landmark.x * width),
            'y': int(landmark.y * height),
        }

    def _sync_feature_regions(self, feature_boxes, width: int, height: int) -> None:
        for key, base_box in feature_boxes.items():
            region = self.feature_regions[key]
            # Smooth base_box updates to reduce choppy motion when detections
            # arrive at lower frequency. Use an exponential moving average.
            if region.base_box is None:
                region.base_box = base_box
            else:
                a = getattr(self, 'landmark_smooth_alpha', 0.65)
                sm_x = int(region.base_box.x * (1.0 - a) + base_box.x * a)
                sm_y = int(region.base_box.y * (1.0 - a) + base_box.y * a)
                sm_w = int(region.base_box.w * (1.0 - a) + base_box.w * a)
                sm_h = int(region.base_box.h * (1.0 - a) + base_box.h * a)
                region.base_box = Box(sm_x, sm_y, max(1, sm_w), max(1, sm_h))

            current_box = self._current_region_box(region)
            region.current_box = self._clamp_box(current_box, width, height)

    def _current_region_box(self, region: RegionState) -> Box:
        if region.base_box is None:
            return Box(0, 0, 1, 1)

        base_center_x = region.base_box.x + region.base_box.w / 2
        base_center_y = region.base_box.y + region.base_box.h / 2
        current_w = max(1, int(region.base_box.w * region.scale))
        current_h = max(1, int(region.base_box.h * region.scale))
        current_center_x = base_center_x + region.offset_x
        current_center_y = base_center_y + region.offset_y
        return Box(
            int(current_center_x - current_w / 2),
            int(current_center_y - current_h / 2),
            current_w,
            current_h,
        )

    def _draw_region_sprite(self, output, source_frame, region: RegionState) -> None:
        if region.base_box is None or region.current_box is None:
            return

        source_box = self._clamp_box(region.base_box, source_frame.shape[1], source_frame.shape[0])
        source_crop = source_frame[source_box.y:source_box.y + source_box.h, source_box.x:source_box.x + source_box.w]
        if source_crop.size == 0:
            return

        dest_box = self._clamp_box(region.current_box, output.shape[1], output.shape[0])
        if dest_box.w <= 0 or dest_box.h <= 0:
            return

        sprite = cv2.resize(source_crop, (dest_box.w, dest_box.h), interpolation=cv2.INTER_LINEAR)
        alpha = self._sprite_alpha_mask(dest_box.w, dest_box.h)
        self._blend_sprite(output, sprite, alpha, dest_box)

    def _sprite_alpha_mask(self, width: int, height: int):
        mask = np.zeros((height, width), dtype='uint8')
        cv2.ellipse(mask, (width // 2, height // 2), (max(1, width // 2), max(1, height // 2)), 0, 0, 360, 255, -1)
        feather = int(self.blur_strength.get())
        if feather > 0:
            kernel = feather * 2 + 1
            mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
        return mask

    def _blend_sprite(self, output, sprite, alpha_mask, dest_box: Box) -> None:
        x1 = max(0, dest_box.x)
        y1 = max(0, dest_box.y)
        x2 = min(output.shape[1], dest_box.x + dest_box.w)
        y2 = min(output.shape[0], dest_box.y + dest_box.h)

        if x2 <= x1 or y2 <= y1:
            return

        src_x1 = x1 - dest_box.x
        src_y1 = y1 - dest_box.y
        src_x2 = src_x1 + (x2 - x1)
        src_y2 = src_y1 + (y2 - y1)

        roi = output[y1:y2, x1:x2]
        sprite_roi = sprite[src_y1:src_y2, src_x1:src_x2]
        alpha_roi = alpha_mask[src_y1:src_y2, src_x1:src_x2].astype('float32') / 255.0
        alpha_roi = cv2.merge([alpha_roi, alpha_roi, alpha_roi])
        blended = (sprite_roi.astype('float32') * alpha_roi) + (roi.astype('float32') * (1.0 - alpha_roi))
        output[y1:y2, x1:x2] = blended.astype('uint8')

    def _box_from_points(self, points, padding: int) -> Box:
        min_x = min(point['x'] for point in points) - padding
        max_x = max(point['x'] for point in points) + padding
        min_y = min(point['y'] for point in points) - padding
        max_y = max(point['y'] for point in points) + padding
        return Box(min_x, min_y, max(1, max_x - min_x), max(1, max_y - min_y))

    def _detect_eye_boxes(self, gray, face: Box):
        feature_boxes = []

        left_zone = self._clamp_box(
            Box(
                face.x + int(face.w * 0.08),
                face.y + int(face.h * 0.14),
                int(face.w * 0.38),
                int(face.h * 0.34),
            ),
            gray.shape[1],
            gray.shape[0],
        )
        right_zone = self._clamp_box(
            Box(
                face.x + int(face.w * 0.54),
                face.y + int(face.h * 0.14),
                int(face.w * 0.38),
                int(face.h * 0.34),
            ),
            gray.shape[1],
            gray.shape[0],
        )

        left_eye = self._detect_best_eye(gray, left_zone)
        right_eye = self._detect_best_eye(gray, right_zone)

        feature_boxes.append(left_eye or self._fallback_eye_box(left_zone))
        feature_boxes.append(right_eye or self._fallback_eye_box(right_zone))
        return feature_boxes

    def _detect_best_eye(self, gray, zone: Box):
        roi = gray[zone.y:zone.y + zone.h, zone.x:zone.x + zone.w]
        if roi.size == 0:
            return None

        detections = self.eye_cascade.detectMultiScale(
            roi,
            scaleFactor=1.05,
            minNeighbors=6,
            minSize=(20, 20),
        )

        if len(detections) == 0:
            return None

        x, y, w, h = max(detections, key=lambda item: item[2] * item[3])
        return self._expand_and_clamp(
            Box(zone.x + int(x), zone.y + int(y), int(w), int(h)),
            gray.shape[1],
            gray.shape[0],
            0.9,
        )

    def _fallback_eye_box(self, zone: Box) -> Box:
        fallback = Box(
            zone.x + int(zone.w * 0.15),
            zone.y + int(zone.h * 0.25),
            int(zone.w * 0.7),
            int(zone.h * 0.5),
        )
        return self._clamp_box(fallback, zone.x + zone.w, zone.y + zone.h)

    def _detect_mouth_box(self, gray, face: Box):
        mouth_zone = self._clamp_box(
            Box(
                face.x + int(face.w * 0.18),
                face.y + int(face.h * 0.52),
                int(face.w * 0.64),
                int(face.h * 0.34),
            ),
            gray.shape[1],
            gray.shape[0],
        )

        roi = gray[mouth_zone.y:mouth_zone.y + mouth_zone.h, mouth_zone.x:mouth_zone.x + mouth_zone.w]
        if roi.size == 0:
            return self._fallback_mouth_box(face)

        detections = self.mouth_cascade.detectMultiScale(
            roi,
            scaleFactor=1.15,
            minNeighbors=18,
            minSize=(30, 20),
        )

        if len(detections) == 0:
            return self._fallback_mouth_box(face)

        x, y, w, h = max(detections, key=lambda item: item[2] * item[3])
        return self._expand_and_clamp(
            Box(mouth_zone.x + int(x), mouth_zone.y + int(y), int(w), int(h)),
            gray.shape[1],
            gray.shape[0],
            0.35,
        )

    def _fallback_mouth_box(self, face: Box) -> Box:
        fallback = Box(
            face.x + int(face.w * 0.25),
            face.y + int(face.h * 0.62),
            int(face.w * 0.5),
            int(face.h * 0.22),
        )
        return self._clamp_box(fallback, face.x + face.w, face.y + face.h)

    def _clamp_box(self, box: Box, max_width: int, max_height: int) -> Box:
        x = max(0, min(box.x, max_width - 1))
        y = max(0, min(box.y, max_height - 1))
        w = max(1, min(box.w, max_width - x))
        h = max(1, min(box.h, max_height - y))
        return Box(x, y, w, h)

    def _expand_and_clamp(self, box: Box, max_width: int, max_height: int, padding_ratio: float) -> Box:
        padding_x = int(box.w * padding_ratio)
        padding_y = int(box.h * padding_ratio)
        expanded = Box(box.x - padding_x, box.y - padding_y, box.w + padding_x * 2, box.h + padding_y * 2)
        return self._clamp_box(expanded, max_width, max_height)

    def _point_in_box(self, point, box: Box) -> bool:
        return box.x <= point['x'] <= box.x + box.w and box.y <= point['y'] <= box.y + box.h

    def _find_region_at_point(self, point):
        for key in ('mouth', 'rightEye', 'leftEye'):
            region = self.feature_regions[key]
            if region.current_box is not None and self._point_in_box(point, region.current_box):
                return key
        return None

    def _on_preview_mouse_down(self, event) -> None:
        if not self.running:
            return

        pointer = self._get_preview_pointer(event)
        region_key = self._find_region_at_point(pointer)
        if region_key is None:
            return

        region = self.feature_regions[region_key]
        if region.current_box is None or region.base_box is None:
            return

        self.active_drag_region = region_key
        current_center_x = region.current_box.x + region.current_box.w / 2
        current_center_y = region.current_box.y + region.current_box.h / 2
        self.drag_pointer_offset = (int(pointer['x'] - current_center_x), int(pointer['y'] - current_center_y))
        self.preview.config(cursor='fleur')

    def _on_preview_mouse_move(self, event) -> None:
        if self.active_drag_region is None:
            self._on_preview_hover(event)
            return

        region = self.feature_regions[self.active_drag_region]
        if region.base_box is None:
            return

        pointer = self._get_preview_pointer(event)
        desired_center_x = pointer['x'] - self.drag_pointer_offset[0]
        desired_center_y = pointer['y'] - self.drag_pointer_offset[1]
        base_center_x = region.base_box.x + region.base_box.w / 2
        base_center_y = region.base_box.y + region.base_box.h / 2
        region.offset_x = int(desired_center_x - base_center_x)
        region.offset_y = int(desired_center_y - base_center_y)
        region.current_box = self._current_region_box(region)
        self.preview.config(cursor='fleur')

    def _on_preview_mouse_up(self, _event) -> None:
        self.active_drag_region = None
        self.preview.config(cursor='arrow')

    def _on_preview_mouse_wheel(self, event) -> None:
        pointer = self._get_preview_pointer(event)
        region_key = self.active_drag_region or self._find_region_at_point(pointer)
        if region_key is None:
            return

        region = self.feature_regions[region_key]
        if region.base_box is None:
            return

        if hasattr(event, 'num') and event.num in (4, 5):
            direction = 1 if event.num == 4 else -1
        else:
            direction = 1 if event.delta > 0 else -1

        step = 0.03 if getattr(event, 'state', 0) & 0x0001 else 0.08
        region.scale = self._clamp_scale(region.scale + (direction * step))
        region.current_box = self._current_region_box(region)

    def _on_preview_hover(self, event) -> None:
        if self.active_drag_region is not None:
            return

        pointer = self._get_preview_pointer(event)
        region_key = self._find_region_at_point(pointer)
        self.preview.config(cursor='hand2' if region_key is not None else 'arrow')

    def run(self) -> None:
        self._show_frame(self._make_blank_frame(640, 480))
        self.root.mainloop()


def main() -> None:
    app = EyeMouthMaskerApp()
    app.run()


if __name__ == '__main__':
    main()
