"""
interactive extrinsic calibration tool
"""

import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import font
import threading


class ExtrinsicCalibrator:
    def __init__(self, extrinsic=np.eye(4)):
        assert extrinsic.shape == (4, 4), f"{extrinsic.shape} != (4,4)"

        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        assert np.isclose(np.linalg.det(R), 1), f"Invalid rotation matrix: {R}"

        self.extrinsic = extrinsic
        self.tx, self.ty, self.tz = t
        self.rx, self.ry, self.rz = cv2.Rodrigues(R)[0].flatten()
        self.is_updated = True
        self.lock = threading.Lock()

        # Start the Tkinter GUI in a separate thread
        self.gui_thread = threading.Thread(target=self.create_gui)
        self.gui_thread.daemon = True
        self.gui_thread.start()

    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("Extrinsic Calibrator")

        # Set font size
        self.font = font.Font(size=20)

        # Create entry widgets for translation
        self.create_entry("tx", self.tx)
        self.create_entry("ty", self.ty)
        self.create_entry("tz", self.tz)

        # Create entry widgets for rotation
        self.create_entry("Rx", np.rad2deg(self.rx))
        self.create_entry("Ry", np.rad2deg(self.ry))
        self.create_entry("Rz", np.rad2deg(self.rz))

        # Add a label to display the current matrix
        self.matrix_label = tk.Label(self.root, text="", justify="left", font=self.font)
        self.matrix_label.pack(padx=10, pady=10)

        self.update_matrix()

        # Handle the window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.root.mainloop()

    def create_entry(self, label, initial_value):
        frame = ttk.Frame(self.root)
        frame.pack(fill="x", padx=5, pady=5)

        label_widget = ttk.Label(frame, text=label, font=self.font)
        label_widget.pack(side="left")

        entry = ttk.Entry(frame, font=self.font)
        entry.insert(0, str(initial_value))
        entry.pack(fill="x", expand=True)
        entry.bind("<Return>", self.update_matrix)

        setattr(self, f"{label}_entry", entry)

    def update_matrix(self, event=None):
        try:
            self.lock.acquire()
            self.tx = float(self.tx_entry.get())
            self.ty = float(self.ty_entry.get())
            self.tz = float(self.tz_entry.get())
            self.rx = np.deg2rad(float(self.Rx_entry.get()))
            self.ry = np.deg2rad(float(self.Ry_entry.get()))
            self.rz = np.deg2rad(float(self.Rz_entry.get()))

            self.extrinsic[:3, 3] = np.array([self.tx, self.ty, self.tz])
            self.extrinsic[:3, :3] = cv2.Rodrigues(
                np.array([self.rx, self.ry, self.rz])
            )[0]

            self.is_updated = True
            self.display_matrix()
        except ValueError:
            print("Invalid input. Please enter numerical values.")
        finally:
            self.lock.release()

    def display_matrix(self):
        text = f"tx: {self.tx:.3f}, ty: {self.ty:.3f}, tz: {self.tz:.3f}\n"
        text += f"Rx: {np.rad2deg(self.rx):.2f}, Ry: {np.rad2deg(self.ry):.2f}, Rz: {np.rad2deg(self.rz):.2f}\n"
        text += "Extrinsic Matrix:\n"
        text += np.array2string(self.extrinsic, precision=3, separator=", ")

        self.matrix_label.config(text=text)

    def on_closing(self):
        self.root.quit()
        self.root.destroy()

    def get_extrinsic(self):
        self.is_updated = False
        return self.extrinsic


if __name__ == "__main__":
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = np.array([0.1, 0.2, 0.3])
    extrinsic[:3, :3] = cv2.Rodrigues(np.array([0.1, 0.2, 0.3]))[0]

    extrinsic_calibrator = ExtrinsicCalibrator(extrinsic)

    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        extrinsic = extrinsic_calibrator.get_extrinsic()

        if not extrinsic_calibrator.is_updated:
            break

    extrinsic = extrinsic_calibrator.get_extrinsic()
    print(extrinsic)
