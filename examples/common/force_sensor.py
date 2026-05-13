"""Six-axis force/torque sensor reading and real-time display."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class FTReading:
    force: np.ndarray
    torque: np.ndarray
    timestamp: float


class ForceTorqueSensor:
    """Reads 6-axis F/T from MuJoCo sensor data."""

    def __init__(self, model, force_name: str = "wrist_force", torque_name: str = "wrist_torque"):
        self.force_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, force_name)
        self.torque_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, torque_name)
        if self.force_id < 0 or self.torque_id < 0:
            raise RuntimeError(f"Sensors '{force_name}'/'{torque_name}' not found in model.")
        self.force_adr = int(model.sensor_adr[self.force_id])
        self.torque_adr = int(model.sensor_adr[self.torque_id])

    def read(self, data) -> FTReading:
        force = data.sensordata[self.force_adr:self.force_adr + 3].copy()
        torque = data.sensordata[self.torque_adr:self.torque_adr + 3].copy()
        return FTReading(force=force, torque=torque, timestamp=float(data.time))

class FTDisplay:
    """Real-time scrolling 6-axis force/torque plot using OpenCV."""

    COLORS_FORCE = [(0, 0, 255), (0, 200, 0), (255, 0, 0)]  # Fx=red, Fy=green, Fz=blue (BGR)
    COLORS_TORQUE = [(255, 255, 0), (255, 0, 255), (0, 255, 255)]  # Mx=cyan, My=magenta, Mz=yellow
    LABELS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

    def __init__(self, width: int = 480, height: int = 320, history_len: int = 200):
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required for FTDisplay.")
        self.width = width
        self.height = height
        self.history_len = history_len
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.force_history = np.zeros((history_len, 3), dtype=np.float64)
        self.torque_history = np.zeros((history_len, 3), dtype=np.float64)
        self.write_idx = 0
        self.filled = False
        self.force_scale = 5.0
        self.torque_scale = 1.0

        cv2.namedWindow("Force/Torque Sensor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Force/Torque Sensor", width, height)

    def update(self, reading: FTReading) -> None:
        self.force_history[self.write_idx] = reading.force
        self.torque_history[self.write_idx] = reading.torque
        self.write_idx = (self.write_idx + 1) % self.history_len
        if self.write_idx == 0:
            self.filled = True

        f_max = max(np.max(np.abs(self.force_history)) + 0.1, 1.0)
        t_max = max(np.max(np.abs(self.torque_history)) + 0.01, 0.1)
        self.force_scale = f_max
        self.torque_scale = t_max

    def show(self) -> None:
        self.canvas[:] = 20
        h, w = self.height, self.width
        text_h = 50
        plot_h = (h - text_h) // 2
        n = self.history_len if self.filled else self.write_idx
        if n < 2:
            cv2.imshow("Force/Torque Sensor", self.canvas)
            return

        indices = np.arange(n)
        ordered = (indices + (self.write_idx if self.filled else 0)) % (self.history_len if self.filled else n)

        force_data = self.force_history[ordered] if self.filled else self.force_history[:n]
        torque_data = self.torque_history[ordered] if self.filled else self.torque_history[:n]

        self._draw_text(force_data[-1], torque_data[-1])
        self._draw_plot(force_data, self.force_scale, text_h, plot_h, self.COLORS_FORCE, "Force (N)")
        self._draw_plot(torque_data, self.torque_scale, text_h + plot_h, plot_h, self.COLORS_TORQUE, "Torque (Nm)")

        cv2.imshow("Force/Torque Sensor", self.canvas)

    def _draw_text(self, force: np.ndarray, torque: np.ndarray) -> None:
        labels = [f"Fx:{force[0]:+.2f}", f"Fy:{force[1]:+.2f}", f"Fz:{force[2]:+.2f}",
                  f"Mx:{torque[0]:+.3f}", f"My:{torque[1]:+.3f}", f"Mz:{torque[2]:+.3f}"]
        colors = self.COLORS_FORCE + self.COLORS_TORQUE
        for i, (lbl, clr) in enumerate(zip(labels, colors)):
            x = 10 + (i % 3) * 155
            y = 18 if i < 3 else 38
            cv2.putText(self.canvas, lbl, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, clr, 1, cv2.LINE_AA)

    def _draw_plot(self, data: np.ndarray, scale: float, y_offset: int, plot_h: int,
                   colors: list, title: str) -> None:
        w = self.width
        n = len(data)
        mid_y = y_offset + plot_h // 2

        cv2.line(self.canvas, (0, mid_y), (w, mid_y), (60, 60, 60), 1)
        cv2.putText(self.canvas, title, (5, y_offset + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        cv2.putText(self.canvas, f"+{scale:.1f}", (w - 50, y_offset + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)

        for ch in range(3):
            pts = []
            for i in range(n):
                x = int(i * (w - 1) / max(n - 1, 1))
                val = data[i, ch]
                y = int(mid_y - (val / scale) * (plot_h // 2 - 5))
                y = max(y_offset + 2, min(y_offset + plot_h - 2, y))
                pts.append((x, y))
            if len(pts) > 1:
                cv2.polylines(self.canvas, [np.array(pts, dtype=np.int32)], False, colors[ch], 1, cv2.LINE_AA)

    def close(self) -> None:
        if CV2_AVAILABLE:
            cv2.destroyWindow("Force/Torque Sensor")
