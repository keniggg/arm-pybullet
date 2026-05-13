"""Eye-in-hand RGB camera window with frame rate control."""
from __future__ import annotations

import numpy as np
import mujoco

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class RGBCameraWindow:
    """Wrist-mounted RGB camera display with configurable frame decimation."""

    def __init__(
        self,
        model,
        camera_id: int,
        width: int = 480,
        height: int = 360,
        render_every_n: int = 8,
        window_name: str = "Wrist RGB Camera",
    ):
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required for RGBCameraWindow.")
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.render_every_n = render_every_n
        self.window_name = window_name
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.enabled = True
        self.frame_count = 0

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)

    def should_update(self, sim_step: int) -> bool:
        return self.enabled and (sim_step % self.render_every_n == 0)

    def update(self, data, overlay_text: str | None = None) -> None:
        if not self.enabled:
            return
        try:
            self.renderer.update_scene(data, camera=self.camera_id)
            rgb = self.renderer.render()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if overlay_text:
                cv2.putText(
                    bgr, overlay_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
                )

            cv2.imshow(self.window_name, bgr)
            self.frame_count += 1
        except Exception as exc:
            if self.enabled:
                print(f"RGB camera error: {exc}")
                self.enabled = False

    def close(self) -> None:
        self.enabled = False
        if self.renderer is not None:
            self.renderer.close()
        if CV2_AVAILABLE:
            cv2.destroyWindow(self.window_name)
