import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np

def get_disparity_map(img_left, img_right, min_disparity, num_disparities,
                      block_size,p1,p2, uniqueness_ratio, speckle_window_size, speckle_range,
                      pre_filter_cap, lmbda, sigma, sgbm_mode):
    # SGBM Parameters
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=p1 * block_size**2,
        P2=p2 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        preFilterCap=pre_filter_cap,
        mode=sgbm_mode
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    disp_left = left_matcher.compute(img_left, img_right)
    disp_right = right_matcher.compute(img_right, img_left)

    # WLSFilter Parameters
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    filtered_left = wls_filter.filter(disp_left, img_left, None, disp_right)
    
    return filtered_left

class StereoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stereo Vision")
        self.root.geometry("800x600")
        
        self.img_left = None
        self.img_right = None

        # Create the frame for parameter tuning on the left side
        self.params_frame = tk.Frame(root)
        self.params_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Create the frame for image display on the right side
        self.image_frame = tk.Label(root)
        self.image_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        self.create_widgets()
        
    def create_widgets(self):
        self.min_disparity = self.create_param("Min Disparity", 0)
        self.num_disparities = self.create_param("Num Disparities", 16)
        self.p1 = self.create_param("P1", 4)
        self.p2 = self.create_param("P2", 64)
        self.block_size = self.create_param("Block Size", 5)
        self.uniqueness_ratio = self.create_param("Uniqueness Ratio", 2)
        self.speckle_window_size = self.create_param("Speckle Window Size", 50)
        self.speckle_range = self.create_param("Speckle Range", 2)
        self.pre_filter_cap = self.create_param("Pre Filter Cap", 63)
        self.lmbda = self.create_param("Lambda", 80000)
        self.sigma = self.create_param("Sigma", 1.2)
        self.sbgm_mode = self.create_param("SGBM Mode", cv2.StereoSGBM_MODE_SGBM_3WAY)

        tk.Button(self.params_frame, text="Load Images", command=self.load_images).pack()
        tk.Button(self.params_frame, text="Compute Disparity", command=self.compute_disparity).pack()
        
    def create_param(self, label, default):
        tk.Label(self.params_frame, text=label).pack()
        var = tk.DoubleVar(value=default)
        entry = tk.Entry(self.params_frame, textvariable=var)
        entry.pack()
        entry.bind('<KeyRelease>', self.on_parameter_change)
        return var
        
    def load_images(self):
        file1 = filedialog.askopenfilename(title="Select Left Image")
        file2 = filedialog.askopenfilename(title="Select Right Image")
        self.img_left = cv2.imread(file1, cv2.IMREAD_GRAYSCALE)
        self.img_right = cv2.imread(file2, cv2.IMREAD_GRAYSCALE)
        self.compute_disparity()
        
    def on_parameter_change(self, event):
        self.compute_disparity()
        
    def compute_disparity(self):
        if self.img_left is None or self.img_right is None:
            print("Please load images first!")
            return
        
        disparity = get_disparity_map(
            self.img_left, self.img_right, 
            int(self.min_disparity.get()),
            int(self.num_disparities.get()),
            int(self.p1.get()),
            int(self.p2.get()),
            int(self.block_size.get()),
            int(self.uniqueness_ratio.get()),
            int(self.speckle_window_size.get()),
            int(self.speckle_range.get()),
            int(self.pre_filter_cap.get()),
            int(self.lmbda.get()),
            float(self.sigma.get()),
            int(self.sbgm_mode.get())
        )

        # Normalize the disparity map
        disparity[~np.isfinite(disparity)] = 0

        print("disparity min: ", disparity.min())
        print("disparity max: ", disparity.max())
        if (disparity.max() - disparity.min()) > 0:
            disparity = (disparity - disparity.min()) / (disparity.max() - disparity.min()) * 255
        else:
            disparity = np.zeros_like(disparity)

        disparity_img = Image.fromarray(disparity.astype(np.uint8))
        disparity_tk = ImageTk.PhotoImage(disparity_img)
        self.image_frame.configure(image=disparity_tk)
        self.image_frame.image = disparity_tk
        
root = tk.Tk()
app = StereoGUI(root)
root.mainloop()