import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def draw_epipolar_lines(img0, img1, pts0, pts1, outfile=None):
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # Create a new image
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img0, cmap="gray")
    ax[1].imshow(img1, cmap="gray")
    plt.suptitle("Epipolar Lines")

    # Keep only epipolar lines
    F, mask = cv2.findFundamentalMat(pts0, pts1, cv2.FM_LMEDS)
    pts0 = pts0[mask.ravel() == 1]
    pts1 = pts1[mask.ravel() == 1]

    ep1 = cv2.computeCorrespondEpilines(pts0, 2, F)
    ep1 = ep1.reshape(-1, 3)
    ep0 = cv2.computeCorrespondEpilines(pts1, 1, F.T)
    ep0 = ep0.reshape(-1, 3)

    # draw points
    ax[0].scatter(pts0[:, 0], pts0[:, 1], c=pt_colors, s=5)
    ax[1].scatter(pts1[:, 0], pts1[:, 1], c=pt_colors, s=5)

    pt_colors = cm.rainbow(np.linspace(0, 1, len(pts0)))
    np.random.shuffle(pt_colors)

    # draw lines
    def draw_line(ax, ep, pts):
        for i, (x, y, z), pt in enumerate(zip(ep, pts)):
            color = pt_colors[i]
            x0, y0 = map(int, [0, -z / y])
            x1, y1 = map(int, [img1.shape[1], -(z + x * img1.shape[1]) / y])
            ax.plot([x0, x1], [y0, y1], color=color)
            ax.scatter(pt[0], pt[1], c=color, s=5)

    draw_line(ax[0], ep0, pts0)
    draw_line(ax[1], ep1, pts1)

    # Save or display the image
    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()
        plt.close()


if __name__ == "__main__":
    pass
