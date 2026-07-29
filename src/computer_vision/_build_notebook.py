#!/usr/bin/env python3
"""Generate the Classical CV notebook as valid JSON."""
import json

cells = []

def md(text):
    return {"cell_type": "markdown", "id": f"md-{len(cells)}", "metadata": {}, "source": text}

def code(text, output=False):
    c = {"cell_type": "code", "id": f"code-{len(cells)}", "metadata": {}, "source": text, "outputs": []}
    if output:
        c["outputs"] = output
    return c

# ============================================================
# Title
# ============================================================
cells.append(md([
    "# Computer Vision: From Pixels to Intelligence\n",
    "\n",
    "**A hands-on journey through the foundations of computer vision (1960s–2000s)**\n",
    "\n",
    "Before deep learning, computer vision was built by hand — mathematicians, geometricians, and signal processors who taught machines to \"see\" using calculus, linear algebra, and a deep understanding of how light interacts with the world.\n",
    "\n",
    "In this notebook, we'll walk through the classical CV pipeline step by step. You'll write the code, run the operations, and see how each building block adds a layer of understanding.\n",
    "\n",
    "> **The story:** We start with the simplest question — *what is an image to a computer?* — and build up to feature matching and image stitching. By the end, you'll understand the foundations that every modern CV system (including neural networks) still rests on."
]))

# ============================================================
# Setup
# ============================================================
cells.append(md(["## Setup"]))
cells.append(code([
    "import numpy as np\n",
    "import cv2\n",
    "import matplotlib.pyplot as plt\n",
    "import torch  # Needed by timm/diffusers\n",
    "\n",
    "plt.rcParams['figure.figsize'] = (10, 6)\n",
    "plt.rcParams['figure.dpi'] = 100\n",
    "plt.rcParams['font.size'] = 14\n"
]))

# ============================================================
# Synthetic images
# ============================================================
cells.append(md([
    "## Generating Synthetic Test Images\n",
    "\n",
    "We create test images using NumPy — self-contained, no external files needed.\n",
    "\n",
    "> **Why synthetic?** In the 1960s–1980s, researchers worked with carefully controlled lab images — shaded boxes, geometric patterns, line drawings. Algorithms had to be clever, not data-hungry."
]))

cells.append(code([
    "def create_colorful_scene(size=256):\n",
    "    img = np.zeros((size, size, 3), dtype=np.uint8)\n",
    "    img[:, :] = (100, 150, 255)  # BGR background\n",
    "    cv2.rectangle(img, (30, 30), (120, 100), (0, 0, 255), -1)  # Red rect\n",
    "    cv2.circle(img, (180, 80), 40, (0, 255, 0), -1)  # Green circle\n",
    "    pts = np.array([[size//2, 20], [size//2-50, size//2+20],\n",
    "                    [size//2+50, size//2+20]], np.int32).reshape((-1,1,2))\n",
    "    cv2.fillPoly(img, [pts], (0, 255, 255))  # Yellow triangle\n",
    "    cv2.rectangle(img, (160, 140), (230, 220), (255,255,255), 3)  # White square\n",
    "    cv2.rectangle(img, (170, 150), (220, 210), (0,0,0), -1)  # Black inner\n",
    "    return img\n",
    "\n",
    "def create_edge_scene(size=256):\n",
    "    img = np.zeros((size, size), dtype=np.uint8)\n",
    "    img[60:80, :] = 200\n",
    "    img[140:160, :] = 200\n",
    "    img[:, 60:80] = 200\n",
    "    img[:, 160:180] = 200\n",
    "    noise = np.random.randint(0, 30, (size, size), dtype=np.uint8)\n",
    "    img = cv2.bitwise_or(img, noise)\n",
    "    return img\n",
    "\n",
    "def create_corner_scene(size=256):\n",
    "    img = np.zeros((size, size), dtype=np.uint8)\n",
    "    for i in range(0, size, 64):\n",
    "        for j in range(0, size, 64):\n",
    "            cv2.rectangle(img, (j+5, i+5), (j+55, i+55), 200, 2)\n",
    "    cv2.line(img, (100, 100), (160, 100), 255, 3)\n",
    "    cv2.line(img, (160, 100), (160, 160), 255, 3)\n",
    "    return img\n",
    "\n",
    "color_scene = create_colorful_scene()\n",
    "edge_scene = create_edge_scene()\n",
    "corner_scene = create_corner_scene()\n",
    "\n",
    "print(f\"Color scene: {color_scene.shape} {color_scene.dtype}\")\n",
    "print(f\"Edge scene:  {edge_scene.shape} {edge_scene.dtype}\")\n",
    "print(f\"Corner scene: {corner_scene.shape} {corner_scene.dtype}\")\n"
]))

cells.append(code([
    "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
    "axes[0].imshow(color_scene)\n",
    "axes[0].set_title(\"Colorful Scene (BGR)\")\n",
    "axes[0].axis('off')\n",
    "axes[1].imshow(edge_scene, cmap='gray')\n",
    "axes[1].set_title(\"Edge Scene (Grayscale)\")\n",
    "axes[1].axis('off')\n",
    "axes[2].imshow(corner_scene, cmap='gray')\n",
    "axes[2].set_title(\"Corner Scene (Grayscale)\")\n",
    "axes[2].axis('off')\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]))

# ============================================================
# Section 1: What is an image?
# ============================================================
cells.append(md([
    "---\n",
    "\n",
    "## 1. What Is an Image? — The NumPy Array\n",
    "\n",
    "To a computer, an image is just a **grid of numbers**.\n",
    "\n",
    "| Type | Structure | Meaning |\n",
    "|------|-----------|---------|\n",
    "| Grayscale | 2D array (H x W) | Each number = brightness (0=black, 255=white) |\n",
    "| Color (BGR) | 3D array (H x W x 3) | Three channels: Blue, Green, Red |\n",
    "\n",
    "> **Historical note:** The first digital photo (1957, Russell Kirsch at NIST) was a 176x176 pixel grid. The fundamental representation hasn't changed."
]))

cells.append(code([
    "print(\"=== Grayscale ===\")\n",
    "print(f\"Shape: {edge_scene.shape}  # (height, width)\")\n",
    "print(f\"Range: [{edge_scene.min()}, {edge_scene.max()}]\")\n",
    "print(f\"\\nFirst 5x5 pixel block:\")\n",
    "print(edge_scene[:5, :5])\n",
    "\n",
    "print(\"\\n=== Color ===\")\n",
    "print(f\"Shape: {color_scene.shape}  # (height, width, channels)\")\n",
    "print(f\"Pixel at (0,0): BGR = {color_scene[0, 0]}\")\n"
]))

cells.append(md([
    "### Image Operations Are Just Matrix Operations\n",
    "\n",
    "- **Brightness** -> add a constant\n",
    "- **Contrast** -> multiply by a scalar\n",
    "- **Cropping** -> array slicing\n",
    "- **Blurring** -> convolution with a kernel\n",
    "\n",
    "This is why image operations are **embarrassingly parallel** — every pixel is independent."
]))

cells.append(code([
    "# Brightness\n",
    "brighter = np.clip(edge_scene.astype(float) * 1.5, 0, 255).astype(np.uint8)\n",
    "# Crop\n",
    "cropped = color_scene[40:200, 40:200]\n",
    "\n",
    "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
    "axes[0].imshow(edge_scene, cmap='gray')\n",
    "axes[0].set_title(\"Original\")\n",
    "axes[0].axis('off')\n",
    "axes[1].imshow(brighter, cmap='gray')\n",
    "axes[1].set_title(\"Brightness x 1.5\")\n",
    "axes[1].axis('off')\n",
    "axes[2].imshow(cropped)\n",
    "axes[2].set_title(\"Cropped [40:200, 40:200]\")\n",
    "axes[2].axis('off')\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]))

# ============================================================
# Section 2: Color Spaces
# ============================================================
cells.append(md([
    "---\n",
    "\n",
    "## 2. Color Spaces — Seeing in Different Dimensions\n",
    "\n",
    "RGB mixes brightness and color together. If lighting changes, all three channels change.\n",
    "\n",
    "> **Historical note:** In the 1970s–1980s, CV researchers realized you need to separate *illumination* from *reflectance*. This led to color spaces like HSV and CIELAB that separate intensity from chroma."
]))

cells.append(code([
    "bgr = color_scene\n",
    "rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)\n",
    "gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)\n",
    "hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)\n",
    "lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)\n",
    "\n",
    "h, s, v = cv2.split(hsv)\n",
    "l, a, b_ch = cv2.split(lab)\n",
    "\n",
    "fig, axes = plt.subplots(2, 3, figsize=(15, 10))\n",
    "\n",
    "axes[0, 0].imshow(bgr)\n",
    "axes[0, 0].set_title(\"BGR (OpenCV default)\")\n",
    "axes[0, 0].axis('off')\n",
    "axes[0, 1].imshow(rgb)\n",
    "axes[0, 1].set_title(\"RGB\")\n",
    "axes[0, 1].axis('off')\n",
    "axes[0, 2].imshow(gray, cmap='gray')\n",
    "axes[0, 2].set_title(\"Grayscale (luminance only)\")\n",
    "axes[0, 2].axis('off')\n",
    "\n",
    "axes[1, 0].imshow(h, cmap='viridis')\n",
    "axes[1, 0].set_title(\"Hue (color type)\")\n",
    "axes[1, 0].axis('off')\n",
    "axes[1, 1].imshow(s, cmap='viridis')\n",
    "axes[1, 1].set_title(\"Saturation (color intensity)\")\n",
    "axes[1, 1].axis('off')\n",
    "axes[1, 2].imshow(l, cmap='viridis')\n",
    "axes[1, 2].set_title(\"Lightness (CIELAB)\")\n",
    "axes[1, 2].axis('off')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]))

cells.append(md([
    "### Key Insight\n",
    "\n",
    "The **Saturation** channel tells you where colored objects are, regardless of brightness. The **Hue** channel tells you *what color* something is, independent of lighting.\n",
    "\n",
    "> **Hands-on:** Threshold the saturation channel to segment colored shapes:"
]))

cells.append(code([
    "# Hands-on: Try different threshold values\n",
    "threshold = 50  # <-- Try changing this!\n",
    "mask = (s > threshold).astype(np.uint8) * 255\n",
    "\n",
    "fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n",
    "axes[0].imshow(s, cmap='viridis')\n",
    "axes[0].set_title(f\"Saturation (threshold={threshold})\")\n",
    "axes[0].axis('off')\n",
    "axes[1].imshow(mask, cmap='gray')\n",
    "axes[1].set_title(\"Binary Mask\")\n",
    "axes[1].axis('off')\n",
    "axes[2].imshow(bgr)\n",
    "axes[2].set_title(\"Original\")\n",
    "axes[2].axis('off')\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]))

# ============================================================
# Section 3: Edge Detection
# ============================================================
cells.append(md([
    "---\n",
    "\n",
    "## 3. Edge Detection — Finding Boundaries\n",
    "\n",
    "Edges are where pixel intensity changes sharply — object boundaries, surface discontinuities, shadows.\n",
    "\n",
    "> **Historical timeline:**\n",
    "> - **1960s:** Edwin Land's RETINEX showed edges matter more than absolute brightness.\n",
    "> - **1970s:** David Marr's theory: edge detection is the first step in visual understanding.\n",
    "> - **1986:** John Canny published the optimal edge detector with three criteria: good detection, good localization, minimal response.\n",
    "\n",
    "### The Math: Sobel Kernels\n",
    "\n",
    "An edge is a **gradient**. We compute it with convolution:\n",
    "\n",
    "$$G_x = \\begin{bmatrix} -1 & 0 & 1 \\\\ -2 & 0 & 2 \\\\ -1 & 0 & 1 \\end{bmatrix}, \\quad\n",
    "G_y = \\begin{bmatrix} -1 & -2 & -1 \\\\ 0 & 0 & 0 \\\\ 1 & 2 & 1 \\end{bmatrix}$$\n",
    "\n",
    "$$\\text{Strength} = \\sqrt{G_x^2 + G_y^2}, \\quad \\text{Direction} = \\arctan2(G_y, G_x)$$"
]))

cells.append(code([
    "# Build Sobel kernels from scratch\n",
    "sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)\n",
    "sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)\n",
    "\n",
    "# Convolution via filter2D\n",
    "grad_x = cv2.filter2D(edge_scene.astype(np.float32), -1, sobel_x)\n",
    "grad_y = cv2.filter2D(edge_scene.astype(np.float32), -1, sobel_y)\n",
    "magnitude = np.sqrt(grad_x**2 + grad_y**2)\n",
    "direction = np.arctan2(grad_y, grad_x)\n",
    "\n",
    "mag_display = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)\n",
    "\n",
    "fig, axes = plt.subplots(1, 4, figsize=(16, 4))\n",
    "axes[0].imshow(edge_scene, cmap='gray')\n",
    "axes[0].set_title(\"Original\")\n",
    "axes[0].axis('off')\n",
    "axes[1].imshow(grad_x, cmap='coolwarm')\n",
    "axes[1].set_title(\"Gx (vertical edges)\")\n",
    "axes[1].axis('off')\n",
    "axes[2].imshow(grad_y, cmap='coolwarm')\n",
    "axes[2].set_title(\"Gy (horizontal edges)\")\n",
    "axes[2].axis('off')\n",
    "axes[3].imshow(mag_display, cmap='gray')\n",
    "axes[3].set_title(\"Magnitude sqrt(Gx^2 + Gy^2)\")\n",
    "axes[3].axis('off')\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]))

cells.append(md([
    "### The Canny Edge Detector (1986)\n",
    "\n",
    "Canny solved Sobel's problems with a 5-step pipeline:\n",
    "\n",
    "1. **Gaussian blur** — reduce noise\n",
    "2. **Gradient computation** — Sobel\n",
    "3. **Non-maximum suppression** — thin edges to 1 pixel\n",
    "4. **Double threshold** — strong vs. weak edges\n",
    "5. **Hysteresis** — connect weak edges to strong ones"
]))

cells.append(code([
    "# Full Canny pipeline\n",
    "blurred = cv2.GaussianBlur(edge_scene, (5, 5), 1.5)\n",
    "canny = cv2.Canny(edge_scene, 50, 150)\n",
    "\n",
    "fig, axes = plt.subplots(1, 4, figsize=(16, 4))\n",
    "axes[0].imshow(edge_scene, cmap='gray')\n",
    "axes[0].set_title(\"Original\")\n",
    "axes[0].axis('off')\n",
    "axes[1].imshow(blurred, cmap='gray')\n",
    "axes[1].set_title(\"1. Gaussian Blur\")\n",
    "axes[1].axis('off')\n",
    "gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0)\n",
    "gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1)\n",
    "mag = np.sqrt(gx**2 + gy**2)\n",
    "axes[2].imshow(cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cmap='gray')\n",
    "axes[2].set_title(\"2. Gradient (thick edges)\")\n",
    "axes[2].axis('off')\n",
    "axes[3].imshow(canny, cmap='gray')\n",
    "axes[3].set_title(\"3. Canny (thin, clean)\")\n",
    "axes[3].axis('off')\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]))

cells.append(md([
    "### Hysteresis: Two Thresholds\n",
    "\n",
    "- **High threshold (150):** Strong edges only\n",
    "- **Low threshold (50):** Weak edges\n",
    "\n",
    "A weak edge survives **only if connected to a strong edge**. Like following breadcrumbs.\n",
    "\n",
    "> **Hands-on:** Experiment with thresholds:"
]))

cells.append(code([
    "# Hands-on: Try different threshold pairs\n",
    "low, high = 30, 100  # <-- Experiment!\n",
    "edges = cv2.Canny(edge_scene, low, high)\n",
    "\n",
    "plt.figure(figsize=(10, 4))\n",
    "plt.imshow(edges, cmap='gray')\n",
    "plt.title(f\"Canny (low={low}, high={high})\")\n",
    "plt.axis('off')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# Try: low=0, high=255 (everything is an edge)\n",
    "# Try: low=200, high=255 (only strongest edges)\n"
]))

# ============================================================
# Section 4: Corner Detection
# ============================================================
cells.append(md([
    "---\n",
    "\n",
    "## 4. Corner Detection — Where Edges Meet\n",
    "\n",
    "Corners are intersections where intensity changes in **all directions**. They're far more informative than edges because they're stable across viewpoints.\n",
    "\n",
    "> **Historical timeline:**\n",
    "> - **1988:** Harris & Stephens publish the corner detector.\n",
    "> - **1999:** David Lowe discovers SIFT — scale-invariant keypoints.\n",
    "> - **2004:** SIFT wins the International Competition on Feature-detection.\n",
    "\n",
    "### The Harris Corner Response\n",
    "\n",
    "Look at a small window around each pixel. Compute the **structure matrix** $M$ from image gradients. The eigenvalues of $M$ tell us:\n",
    "\n",
    "- **Both large** -> corner\n",
    "- **One large, one small** -> edge\n",
    "- **Both small** -> flat region\n",
    "\n",
    "Harris simplified this to: $R = \\det(M) - k \\cdot \\text{trace}(M)^2$"
]))

cells.append(code([
    "# Harris corner detection\n",
    "harris = cv2.cornerHarris(edge_scene.astype(np.float32), blockSize=2, ksize=3, k=0.04)\n",
    "harris_dilated = cv2.dilate(harris, None)\n",
    "threshold = 0.01 * harris_dilated.max()\n",
    "corners = harris_dilated > threshold\n",
    "\n",
    "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
    "axes[0].imshow(corner_scene, cmap='gray')\n",
    "axes[0].set_title(\"Original\")\n",
    "axes[0].axis('off')\n",
    "axes[1].imshow(harris, cmap='hot')\n",
    "axes[1].set_title(\"Harris Response (bright = corner)\")\n",
    "axes[1].axis('off')\n",
    "axes[2].imshow(corner_scene, cmap='gray')\n",
    "y_coords, x_coords = np.where(corners)\n",
    "for x, y in zip(x_coords, y_coords):\n",
    "    axes[2].plot(x, y, 'r+', markersize=8, markeredgewidth=2)\n",
    "axes[2].set_title(f\"Detected Corners ({len(x_coords)} found)\")\n",
    "axes[2].axis('off')\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]))

cells.append(md([
    "### Why Corners Matter\n",
    "\n",
    "Corners are **repeatable features** — find the same corner in two images. This enables:\n",
    "- Image stitching (match corners -> homography)\n",
    "- 3D reconstruction (triangulate from multiple views)\n",
    "- Object tracking (track corners across frames)\n",
    "- Robot localization (match corners to a map)"
]))

# ============================================================
# Section 5: Contours
# ============================================================
cells.append(md([
    "---\n",
    "\n",
    "## 5. Contours & Shapes — Outlining What We See\n",
    "\n",
    "Binary images (edges, thresholds) -> **contours** — boundaries of connected components.\n",
    "\n",
    "> **Historical note:** The Suzuki topological contour tracing algorithm (1985) is the foundation of OpenCV's `findContours`. Pure algorithmic geometry."
]))

cells.append(code([
    "# Find contours in saturation mask\n",
    "binary = (s > 50).astype(np.uint8) * 255\n",
    "contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)\n",
    "\n",
    "contour_img = bgr.copy()\n",
    "cv2.drawContours(contour_img, contours, -1, (255, 255, 0), 2)\n",
    "\n",
    "print(f\"Found {len(contours)} contours\")\n",
    "print(f\"{'Idx':<4} {'Area':<10} {'Perimeter':<10} {'Bounding Box':<18} {'Shape'}\")\n",
    "print(\"-\" * 60)\n",
    "for i, c in enumerate(contours):\n",
    "    area = cv2.contourArea(c)\n",
    "    if area < 100:\n",
    "        continue\n",
    "    peri = cv2.arcLength(c, True)\n",
    "    x, y, w, h = cv2.boundingRect(c)\n",
    "    approx = cv2.approxPolyDP(c, 0.04 * peri, True)\n",
    "    n = len(approx)\n",
    "    shape = {3: \"Triangle\", 4: \"Rectangle\"}.get(n, \"Circle\" if n > 6 else f\"{n}-gon\")\n",
    "    print(f\"{i:<4} {area:<10.0f} {peri:<10.1f} ({x},{y},{w}x{h})    {shape}\")\n",
    "\n",
    "plt.figure(figsize=(12, 6))\n",
    "plt.imshow(contour_img)\n",
    "plt.title(f\"{len([c for c in contours if cv2.contourArea(c) > 100])} Shapes Detected\")\n",
    "plt.axis('off')\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]))

# ============================================================
# Section 6: Feature Matching
# ============================================================
cells.append(md([
    "---\n",
    "\n",
    "## 6. Feature Detection & Matching — The 2000s Golden Age\n",
    "\n",
    "Harris corners aren't scale-invariant. The breakthrough came with **scale-invariant features**:\n",
    "\n",
    "> **Timeline:**\n",
    "> - **1999:** Lowe publishes SIFT (scale, rotation invariant)\n",
    "> - **2006:** SURF (Speeded-Up Robust Features)\n",
    "> - **2006:** ORB (patent-free, blazingly fast)\n",
    "\n",
    "### ORB: FAST Keypoints + BRIEF Descriptors\n",
    "\n",
    "- **Keypoints:** FAST detects corners quickly\n",
    "- **Descriptors:** BRIEF encodes each keypoint as a binary string\n",
    "- **Matching:** Hamming distance between binary strings"
]))

cells.append(code([
    "# ORB feature detection\n",
    "orb = cv2.ORB_create(nfeatures=500)\n",
    "kp1, desc1 = orb.detectAndCompute(edge_scene, None)\n",
    "kp2, desc2 = orb.detectAndCompute(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), None)\n",
    "\n",
    "print(f\"Image 1: {len(kp1)} keypoints\")\n",
    "print(f\"Image 2: {len(kp2)} keypoints\")\n",
    "\n",
    "img_kp = cv2.drawKeypoints(edge_scene, kp1, None,\n",
    "    color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)\n",
    "plt.figure(figsize=(12, 6))\n",
    "plt.imshow(img_kp)\n",
    "plt.title(f\"{len(kp1)} ORB Keypoints\")\n",
    "plt.axis('off')\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]))

cells.append(md([
    "### Matching with Lowe's Ratio Test\n",
    "\n",
    "For each keypoint, find the **two nearest** descriptors. Keep only if the best match is much better than the second:\n",
    "\n",
    "$$d_1 < 0.7 \\times d_2$$\n",
    "\n",
    "This filters out ambiguous matches."
]))

cells.append(code([
    "# BFMatcher with Hamming distance (binary descriptors)\n",
    "bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)\n",
    "matches = bf.knnMatch(desc1, desc2, k=2)\n",
    "\n",
    "# Lowe's ratio test\n",
    "good = [m for m, n in matches if m.distance < 0.7 * n.distance]\n",
    "print(f\"Raw matches: {len(matches)} -> After ratio test: {len(good)} good matches\")\n",
    "\n",
    "match_img = cv2.drawMatches(edge_scene, kp1, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), kp2,\n",
    "    good[:50], flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)\n",
    "plt.figure(figsize=(16, 8))\n",
    "plt.imshow(match_img)\n",
    "plt.title(f\"Feature Matching: {len(good)} good matches\")\n",
    "plt.axis('off')\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]))

# ============================================================
# Section 7: Image Stitching
# ============================================================
cells.append(md([
    "---\n",
    "\n",
    "## 7. Image Stitching — The Classical CV Capstone\n",
    "\n",
    "This is where classical CV peaked — build something genuinely useful with nothing but geometry and linear algebra.\n",
    "\n",
    "> **Historical note:** Panorama stitching was one of the first consumer CV applications. Under the hood: feature detection -> matching -> homography -> warping -> blending.\n",
    "\n",
    "### The Pipeline:\n",
    "1. Detect features in both images\n",
    "2. Match features (BFMatcher + ratio test)\n",
    "3. Estimate **homography** (RANSAC)\n",
    "4. Warp one image using the homography\n",
    "5. Blend the overlap region"
]))

cells.append(code([
    "# Create overlapping panorama pair\n",
    "def create_panorama_pair(size=512, overlap=150):\n",
    "    scene = np.zeros((size, size + overlap, 3), dtype=np.uint8)\n",
    "    for x in range(scene.shape[1]):\n",
    "        t = x / scene.shape[1]\n",
    "        scene[:, x] = (int(50+200*t), int(100+100*(1-t)), int(200-100*t))\n",
    "    cv2.circle(scene, (100, size//2), 40, (0, 255, 0), -1)\n",
    "    cv2.rectangle(scene, (size//2-30, size//2-30), (size//2+30, size//2+30), (0, 0, 255), -1)\n",
    "    cv2.circle(scene, (size+overlap-100, size//2), 35, (255, 255, 0), -1)\n",
    "    return scene[:, :size-overlap//2], scene[:, overlap//2:]\n",
    "\n",
    "left, right = create_panorama_pair()\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 6))\n",
    "axes[0].imshow(left)\n",
    "axes[0].set_title(\"Left Image\")\n",
    "axes[0].axis('off')\n",
    "axes[1].imshow(right)\n",
    "axes[1].set_title(\"Right Image\")\n",
    "axes[1].axis('off')\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]))

cells.append(code([
    "# === STEP 1: Detect & match ===\n",
    "orb = cv2.ORB_create(nfeatures=1000)\n",
    "kp_l, desc_l = orb.detectAndCompute(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), None)\n",
    "kp_r, desc_r = orb.detectAndCompute(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), None)\n",
    "\n",
    "bf = cv2.BFMatcher(cv2.NORM_HAMMING)\n",
    "matches = bf.knnMatch(desc_l, desc_r, k=2)\n",
    "good = [m for m, n in matches if m.distance < 0.7 * n.distance]\n",
    "print(f\"Matches: {len(good)}\")\n",
    "\n",
    "# === STEP 2: Homography via RANSAC ===\n",
    "if len(good) >= 4:\n",
    "    src_pts = np.float32([kp_l[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)\n",
    "    dst_pts = np.float32([kp_r[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)\n",
    "    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)\n",
    "    print(f\"Homography:\\n{H}\")\n",
    "    print(f\"Inliers: {np.sum(mask)} / {len(good)}\")\n"
]))

cells.append(code([
    "# === STEP 3: Warp & blend ===\n",
    "if H is not None:\n",
    "    h_l, w_l = left.shape[:2]\n",
    "    h_r, w_r = right.shape[:2]\n",
    "    \n",
    "    # Warp right image\n",
    "    stitched = cv2.warpPerspective(right, H, (w_l + w_r, h_l))\n",
    "    stitched[0:h_l, 0:w_l] = left\n",
    "    \n",
    "    # Simple linear blend in overlap\n",
    "    for x in range(max(0, w_l-150), min(w_l+w_r, w_l+150)):\n",
    "        if x < w_l:\n",
    "            lw = 1.0 - (x - max(0, w_l-150)) / 150.0\n",
    "        else:\n",
    "            lw = 0.0\n",
    "        rw = 1.0 - lw\n",
    "        stitched[0:h_l, x] = (lw * stitched[0:h_l, x] + rw * right[0:h_r, x-w_l]).astype(np.uint8)\n",
    "    \n",
    "    fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
    "    axes[0].imshow(left)\n",
    "    axes[0].set_title(\"Left\")\n",
    "    axes[0].axis('off')\n",
    "    axes[1].imshow(right)\n",
    "    axes[1].set_title(\"Right\")\n",
    "    axes[1].axis('off')\n",
    "    axes[2].imshow(stitched)\n",
    "    axes[2].set_title(\"Stitched Panorama\")\n",
    "    axes[2].axis('off')\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n"
]))

cells.append(md([
    "### The Homography Matrix\n",
    "\n",
    "$$\\begin{bmatrix} x' \\\\ y' \\\\ w' \\end{bmatrix} = H \\begin{bmatrix} x \\\\ y \\\\ 1 \\end{bmatrix}$$\n",
    "\n",
    "OpenCV uses **RANSAC** (Random Sample Consensus): randomly sample 4 point pairs, compute candidate homography, keep the one with the most inliers. Robust to outlier matches.\n",
    "\n",
    "> **This is the capstone of classical CV:** Feature detection -> matching -> geometric estimation -> warping -> blending. No training, no data, no neural networks."
]))

# ============================================================
# Summary
# ============================================================
cells.append(md([
    "---\n",
    "\n",
    "## Summary: The Classical CV Pipeline\n",
    "\n",
    "| Step | Technique | Era | What it does |\n",
    "|------|-----------|-----|-------------|\n",
    "| 1 | Image as array | 1960s | Pixels = numbers |\n",
    "| 2 | Color spaces (HSV, Lab) | 1970s | Separate brightness from color |\n",
    "| 3 | Edge detection (Sobel/Canny) | 1960s-80s | Find boundaries |\n",
    "| 4 | Corner detection (Harris) | 1988 | Find distinctive points |\n",
    "| 5 | Contours & shapes | 1985 | Outline and classify objects |\n",
    "| 6 | Feature matching (ORB/SIFT) | 1999-2006 | Match points across images |\n",
    "| 7 | Image stitching (homography) | 2000s | Combine images into panoramas |\n",
    "\n",
    "> **The through-line:** Each step adds a layer of *understanding*. Pixels -> edges -> corners -> shapes -> matches -> geometric transformation. Each layer abstracts away irrelevant detail and preserves what matters.\n",
    "\n",
    "> **The bridge to deep learning:** Modern neural networks do exactly the same thing, but automatically. CNN first layers learn Sobel-like edges. Second layers learn corner-like patterns. Deeper layers learn shape-like features. **Deep learning didn't replace classical CV — it automated it.**\n",
    "\n",
    "---\n",
    "\n",
    "*Next: A small CNN built with Flax NNX (comparing classical vs AI workflows), then pretrained models (AlexNet, ResNet, ViT) with ImageNet, DDPM diffusion, and JEPA overview.*"
]))

# ============================================================
# Section 8: Small CNN with Flax NNX
# ============================================================
cells.append(md([
    "---\n",
    "\n",
    "## 8. A Small CNN with Flax NNX — Classical vs. AI Workflows\n",
    "\n",
    "Now let's build a neural network from scratch using **Flax NNX** (JAX's neural network library). We'll train a small CNN to classify our synthetic shapes.\n",
    "\n",
    "> **The workflow difference:**\n",
    "> - **Classical CV:** Engineer features (edges, corners, descriptors) -> hand-craft pipeline -> geometric matching\n",
    "> - **AI/CV:** Define architecture -> feed data -> let optimization learn the features automatically\n",
    "\n",
    "The CNN we'll build is simple but captures the key concepts: convolutional layers, pooling, activation functions, and classification heads."
]))

cells.append(code([
    "# Flax NNX imports (Flax 0.8+)\n",
    "import flax.nnx as nnx\n",
    "import jax\n",
    "import jax.numpy as jnp\n",
    "from jax import random\n",
    "\n",
    "# Print device info\n",
    "print(f\"Devices: {jax.devices()}\")\n"
]))

cells.append(md([
    "### CNN Architecture\n",
    "\n",
    "Our network:\n",
    "\n",
    "$$\\text{Conv(3->16, 3x3)} \\rightarrow \\text{ReLU} \\rightarrow \\text{MaxPool} \\rightarrow\n",
    "\\text{Conv(16->32, 3x3)} \\rightarrow \\text{ReLU} \\rightarrow \\text{MaxPool} \\rightarrow\n",
    "\\text{Flatten} \\rightarrow \\text{Dense(32->5)} \\rightarrow \\text{Classes}$$\n",
    "\n",
    "This mirrors what deeper networks like ResNet do, just much smaller:\n",
    "- **Conv layers** learn edge/corner/shape features (like classical CV, but learned)\n",
    "- **Pooling** provides spatial invariance\n",
    "- **Dense layers** classify based on learned features"
]))

cells.append(code([
    "class SimpleCNN(nnx.Module):\n",
    "    def __init__(self, rngs: nnx.Rngs):\n",
    "        self.conv1 = nn.Conv(in_channels=3, out_channels=16, kernel_size=3)\n",
    "        self.conv2 = nn.Conv(in_channels=16, out_channels=32, kernel_size=3)\n",
    "        self.fc1 = nn.Linear(in_features=32 * 32 * 32, out_features=5)\n",
    "        self.relu = nnx.ReLU\n",
    "\n",
    "    def __call__(self, x):\n",
    "        x = self.relu(self.conv1(x))\n",
    "        x = nnx.avg_pool(x, window_size=(2, 2), strides=(2, 2))  # 64->32\n",
    "        x = self.relu(self.conv2(x))\n",
    "        x = nnx.avg_pool(x, window_size=(2, 2), strides=(2, 2))  # 32->16\n",
    "        x = x.reshape(x.shape[0], -1)  # flatten\n",
    "        x = self.fc1(x)\n",
    "        return x\n",
    "\n",
    "# Create model with RNGs for initialization\n",
    "rng = random.key(42)\n",
    "model = SimpleCNN(nnx.Rngs(rng))\n",
    "optimizer = nnx.Optimizer(model)\n",
    "\n",
    "n_params = sum(p.size for p in nnx.state(model).values())\n",
    "print(f\"CNN created. Parameters: {n_params:,}\")\n"
]))

cells.append(md([
    "### Preparing Data\n",
    "\n",
    "We'll create simple synthetic images for each class:\n",
    "- **Class 0:** Red circles\n",
    "- **Class 1:** Green squares\n",
    "- **Class 2:** Blue triangles\n",
    "- **Class 3:** Yellow rectangles\n",
    "- **Class 4:** Mixed shapes"
]))

cells.append(code([
    "def make_dataset(rng, n_per_class=100, size=64):\n",
    "    \"\"\"Generate synthetic classification dataset.\"\"\"\n",
    "    images, labels = [], []\n",
    "    for cls in range(5):\n",
    "        for _ in range(n_per_class):\n",
    "            img = np.zeros((size, size, 3), dtype=np.uint8)\n",
    "            colors = [\n",
    "                (0, 0, 255),   # Red (BGR)\n",
    "                (0, 255, 0),   # Green\n",
    "                (255, 0, 0),   # Blue\n",
    "                (0, 255, 255), # Yellow\n",
    "                (255, 255, 255), # White\n",
    "            ]\n",
    "            color = colors[cls]\n",
    "            cx, cy = np.random.randint(15, size-15, 2)\n",
    "            if cls == 0:  # Circle\n",
    "                cv2.circle(img, (cx, cy), np.random.randint(10, 20), color, -1)\n",
    "            elif cls == 1:  # Square\n",
    "                s = np.random.randint(15, 30)\n",
    "                cv2.rectangle(img, (cx-s//2, cy-s//2), (cx+s//2, cy+s//2), color, -1)\n",
    "            elif cls == 2:  # Triangle\n",
    "                pts = np.array([[cx, cy-15], [cx-15, cy+15], [cx+15, cy+15]], np.int32).reshape((-1,1,2))\n",
    "                cv2.fillPoly(img, [pts], color)\n",
    "            elif cls == 3:  # Rectangle\n",
    "                w, h = np.random.randint(20, 35), np.random.randint(10, 15)\n",
    "                cv2.rectangle(img, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), color, -1)\n",
    "            else:  # Mixed\n",
    "                cv2.circle(img, (cx-10, cy), 8, color, -1)\n",
    "                cv2.rectangle(img, (cx+5, cy-8), (cx+15, cy+8), color, -1)\n",
    "            # Add some noise\n",
    "            noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)\n",
    "            img = cv2.bitwise_or(img, noise)\n",
    "            images.append(img)\n",
    "            labels.append(cls)\n",
    "    images = np.array(images, dtype=np.float32) / 255.0\n",
    "    labels = np.array(labels)\n",
    "    return images, labels\n",
    "\n",
    "X, y = make_dataset(rng)\n",
    "print(f\"Dataset: {X.shape} images, {y.shape} labels\")\n",
    "print(f\"Class distribution: {np.bincount(y)}\")\n"
]))

cells.append(md([
    "### Training Loop\n",
    "\n",
    "Standard mini-batch training with cross-entropy loss. The key JAX pattern:\n",
    "- `jax.grad` computes gradients\n",
    "- Gradients update the optimizer\n",
    "- `jax.jit` compiles the training step for speed"
]))

cells.append(code([
    "@jax.jit\n",
    "def train_step(optimizer, images, labels):\n",
    "    def loss_fn(model):\n",
    "        logits = model(images)\n",
    "        log_softmax = logits - jax.nn.log_softmax(logits)\n",
    "        loss = jnp.mean(-log_softmax[jnp.arange(len(labels)), labels])\n",
    "        return loss\n",
    "    loss, grads = jax.value_and_grad(loss_fn)(optimizer)\n",
    "    optimizer.update(grads)\n",
    "    return loss\n",
    "\n",
    "@jax.jit\n",
    "def evaluate(model, images, labels):\n",
    "    logits = model(images)\n",
    "    preds = jnp.argmax(logits, axis=1)\n",
    "    return jnp.mean(preds == labels)\n",
    "\n",
    "# Train\n",
    "batch_size = 32\n",
    "epochs = 10\n",
    "n = X.shape[0]\n",
    "\n",
    "for epoch in range(epochs):\n",
    "    perm = random.permutation(rng, n)\n",
    "    for i in range(0, n, batch_size):\n",
    "        idx = perm[i:i+batch_size]\n",
    "        train_step(optimizer, X[idx], y[idx])\n",
    "    acc = evaluate(model, X, y)\n",
    "    print(f\"Epoch {epoch+1:2d}/{epochs} | Accuracy: {acc:.1%}\")\n"
]))

cells.append(md([
    "### What Did the CNN Learn?\n",
    "\n",
    "Compare with classical CV:\n",
    "\n",
    "| Aspect | Classical CV (Sobel/Canny) | CNN (Flax NNX) |\n",
    "|--------|---------------------------|-----------------|\n",
    "| Features | Hand-crafted kernels | Learned from data |\n",
    "| Pipeline | Edge -> Corner -> Match -> Geometry | Raw pixels -> Classes |\n",
    "| Generalization | Limited to designed invariances | Learned from examples |\n",
    "| Interpretability | High (we know exactly what each step does) | Lower (black box) |\n",
    "| Data needed | None | Hundreds of examples |\n",
    "\n",
    "> **Key insight:** The CNN's first conv layer weights, when visualized, look remarkably like edge detectors and color blobs — similar to Sobel/Gabor filters. **The network learned classical CV features automatically.**"
]))

# ============================================================
# Section 9: AlexNet & ResNet with ImageNet
# ============================================================
cells.append(md([
    "---\n",
    "\n",
    "## 9. AlexNet & ResNet — The Deep Learning Revolution\n",
    "\n",
    "### AlexNet (2012) — The ImageNet Breakthrough\n",
    "\n",
    "Alex Krizhevsky's AlexNet won the ImageNet Competition with an 8% lower top-5 error than the runner-up (classical CV). Key innovations:\n",
    "- **Deep architecture:** 8 layers (conv + dense)\n",
    "- **ReLU activations:** Much faster training than sigmoid\n",
    "- **Dropout:** Regularization that prevented overfitting\n",
    "- **GPU training:** Used two GPUs for parallel convolution\n",
    "\n",
    "### ResNet (2015) — Going Deeper with Skip Connections\n",
    "\n",
    "Kaiming He's ResNet solved the degradation problem: deeper networks were *worse*, not better. The fix: **skip connections** that let gradients flow directly through the network.\n",
    "\n",
    "$$\\text{Output} = \\text{Activation}(\\text{Layers}(x) + x)$$\n",
    "\n",
    "The `+ x` is the skip connection — it lets the network learn *residuals* (what to add to the identity) rather than complete mappings."
]))

cells.append(code([
    "# Load pretrained models with ImageNet classes\n",
    "import timm\n",
    "from timm.data import resolve_data_config\n",
    "from timm.data.transforms_factory import create_transform\n",
    "import torch\n",
    "\n",
    "# Load ImageNet-1K label mapping (1000 classes)\n",
    "try:\n",
    "    from timm.data._imagenet_classes import _class_names\n",
    "    imagenet_labels = _class_names\n",
    "except ImportError:\n",
    "    # Fallback: generate placeholder labels\n",
    "    imagenet_labels = [f\"class_{i}\" for i in range(1000)]\n",
    "\n",
    "# Load pretrained ResNet-18 (classic CNN architecture)\n",
    "# Note: timm doesn't include classic AlexNet (2012) architecture.\n",
    "# ResNet-18 represents the \"classic CNN\" era (2012-2015).\n",
    "print(\"Loading pretrained ResNet-18 (ImageNet-1K)...\")\n",
    "model = timm.create_model(\"resnet18\", pretrained=True, num_classes=1000)\n",
    "model.eval()\n",
    "\n",
    "config = resolve_data_config({}, model=model)\n",
    "transform = create_transform(**config)\n",
    "\n",
    "print(f\"Model: resnet18\")\n",
    "print(f\"Parameters: {sum(p.numel() for p in model.parameters()):,}\")\n",
    "print(f\"ImageNet-1K classes loaded: {len(imagenet_labels)}\")\n"
]))

cells.append(code([
    "# Run inference on our synthetic image\n",
    "img = torch.tensor(transform(color_scene)).unsqueeze(0)\n",
    "\n",
    "with torch.no_grad():\n",
    "    logits = model(img)\n",
    "    probs = torch.softmax(logits, dim=1)[0]\n",
    "\n",
    "# Top 5 predictions with ImageNet labels\n",
    "top5_idx = probs.topk(5).indices.tolist()\n",
    "top5_probs = probs.topk(5).values.tolist()\n",
    "\n",
    "print(\"\\nTop 5 ImageNet predictions:\")\n",
    "for idx, prob in zip(top5_idx, top5_probs):\n",
    "    label = imagenet_labels[idx] if idx < len(imagenet_labels) else f\"class_{idx}\"\n",
    "    print(f\"  [{prob:.2%}] {label}\")\n",
    "\n",
    "# Visualize\n",
    "plt.figure(figsize=(8, 6))\n",
    "plt.imshow(color_scene)\n",
    "label = imagenet_labels[top5_idx[0]] if top5_idx[0] < len(imagenet_labels) else f\"class_{top5_idx[0]}\"\n",
    "plt.title(f\"Top: {label} ({top5_probs[0]:.2%})\")\n",
    "plt.axis('off')\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]))

cells.append(md([
    "### Visualizing What AlexNet/CNNs See\n",
    "\n",
    "The first layer of a CNN learns filters that respond to specific patterns. For AlexNet and early CNNs, these look like:\n",
    "- **Edges** at various orientations (like Sobel)\n",
    "- **Color blobs** (like the color space separation we did)\n",
    "- **Gabor-like patterns** (orientation + frequency selective)\n",
    "\n",
    "This is the \"aha\" moment: **deep learning automated classical CV.** The network learned edge detectors, color separators, and pattern recognizers — all from data, no hand-crafting required.\n",
    "\n",
    "Let's visualize the first layer filters:"
]))

cells.append(code([
    "# Extract first conv layer weights\n",
    "# ResNet has conv1; ViT has patch_embed.proj\n",
    "if hasattr(model, 'conv1'):\n",
    "    weights = model.conv1.weight\n",
    "elif hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'proj'):\n",
    "    weights = model.patch_embed.proj.weight\n",
    "else:\n",
    "    weights = list(model.parameters())[0]\n",
    "\n",
    "weights = weights.cpu()\n",
    "if weights.ndim == 4:\n",
    "    # Average across input channels for visualization\n",
    "    weights = weights.mean(dim=1)\n",
    "\n",
    "# Normalize and display first 16 filters\n",
    "n = min(16, weights.shape[0])\n",
    "fig, axes = plt.subplots(4, 4, figsize=(10, 10))\n",
    "\n",
    "for i in range(n):\n",
    "    w = weights[i]\n",
    "    w_min, w_max = w.min(), w.max()\n",
    "    if w_max - w_min < 1e-8:\n",
    "        w_display = np.zeros_like(w.numpy())\n",
    "    else:\n",
    "        w_display = ((w - w_min) / (w_max - w_min) * 255).astype(np.uint8)\n",
    "    axes[i // 4, i % 4].imshow(w_display, cmap='gray')\n",
    "    axes[i // 4, i % 4].axis('off')\n",
    "\n",
    "axes[0, 0].set_title(\"First Layer Filters (learned edges/patterns)\")\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]))

# ============================================================
# Section 10: Vision Transformer (ViT)
# ============================================================
cells.append(md([
    "---\n",
    "\n",
    "## 10. Vision Transformer (ViT) — Images as Sequences (2020)\n",
    "\n",
    "Transformers dominated NLP. Could they work for images?\n",
    "\n",
    "> **Historical note:** The Vision Transformer (Dosovitskiy et al., 2020) showed that transformers could match or beat CNNs on image classification — *if trained on enough data*. The key insight: treat image patches as tokens in a sequence.\n",
    "\n",
    "### CNN vs. ViT\n",
    "\n",
    "| Aspect | CNN (ResNet) | ViT |\n",
    "|--------|-------------|-----|\n",
    "| Inductive bias | Local connectivity, translation equivariance | None (learned globally) |\n",
    "| Receptive field | Grows with depth (local) | Global from layer 1 |\n",
    "| Data hunger | Moderate (pretraining helps) | High (needs lots of data) |\n",
    "| Parallelization | Sequential layers | Fully parallel (like LLMs) |\n",
    "\n",
    "### How ViT Works\n",
    "\n",
    "1. **Patchify:** Split image into fixed-size patches (e.g., 16x16 pixels)\n",
    "2. **Linear embed:** Project each patch to a vector\n",
    "3. **Add position embeddings:** Patches lose spatial info, so add positional encoding\n",
    "4. **Transformer encoder:** Self-attention across all patches\n",
    "5. **Classification head:** [CLS] token output -> classes"
]))

cells.append(code([
    "# Load a ViT model\n",
    "vit = timm.create_model(\"vit_base_patch16_224.augreg_in21k_ft1k\", pretrained=True, num_classes=1000)\n",
    "vit.eval()\n",
    "\n",
    "vit_config = resolve_data_config({}, model=vit)\n",
    "vit_transform = create_transform(**vit_config)\n",
    "\n",
    "print(f\"ViT Model: {vit.__class__.__name__}\")\n",
    "print(f\"Parameters: {sum(p.numel() for p in vit.parameters()):,}\")\n",
    "print(f\"Patch size: 16x16 -> {224//16}x{224//16} = {(224//16)**2} patches per image\")\n",
    "\n",
    "# Run inference\n",
    "img_vit = torch.tensor(vit_transform(color_scene)).unsqueeze(0)\n",
    "with torch.no_grad():\n",
    "    logits_vit = vit(img_vit)\n",
    "    probs_vit = torch.softmax(logits_vit, dim=1)[0]\n",
    "\n",
    "top5_vit = probs_vit.topk(5)\n",
    "print(\"\\nViT Top 5 predictions:\")\n",
    "for idx, prob in zip(top5_vit.indices.tolist(), top5_vit.values.tolist()):\n",
    "    print(f\"  [{prob:.2%}] Class {idx}\")\n"
]))

cells.append(md([
    "### Key Insight: Self-Attention in Images\n",
    "\n",
    "In a CNN, a pixel at position (100, 100) can only \"see\" its local neighborhood (e.g., 3x3 or 7x7). To understand the whole image, information must propagate through many layers.\n",
    "\n",
    "In a ViT, **every patch attends to every other patch** from layer 1. The self-attention mechanism:\n",
    "\n",
    "$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$\n",
    "\n",
    "This means the ViT can immediately relate the top-left corner to the bottom-right corner — no propagation delay. But it also means no built-in assumption that nearby pixels are more related (the CNN's inductive bias)."
]))

# ============================================================
# Section 11: DDPM - Diffusion Models
# ============================================================
cells.append(md([
    "---\n",
    "\n",
    "## 11. Diffusion Models — Learning to Dream (2020–present)\n",
    "\n",
    "After GANs (2014) promised perfect generation but suffered from mode collapse and training instability, diffusion models emerged as the stable alternative.\n",
    "\n",
    "> **Historical note:**\n",
    "> - **2014:** Goodfellow introduces GANs — generator vs. discriminator battle\n",
    "> - **2015:** Sohl-Dickstein et al. introduce first diffusion models (deep generative models via nonequilibrium thermodynamics)\n",
    "> - **2020:** DDPM (Ho et al.) makes diffusion practical and scalable\n",
    "> - **2021:** Stable Diffusion combines diffusion with latent space for efficiency\n",
    "\n",
    "### How Diffusion Works\n",
    "\n",
    "**Forward process (noising):** Gradually add Gaussian noise over T timesteps:\n",
    "\n",
    "$$q(x_t | x_{t-1}) = \\mathcal{N}(x_t; \\sqrt{1-\\beta_t} \\cdot x_{t-1}, \\beta_t \\cdot I)$$\n",
    "\n",
    "After T steps, $x_T$ is pure noise.\n",
    "\n",
    "**Reverse process (denoising):** A neural network learns to predict the noise at each step:\n",
    "\n",
    "$$p_\\theta(x_{t-1} | x_t) = \\mathcal{N}(x_{t-1}; \\mu_\\theta(x_t, t), \\sigma_t^2 \\cdot I)$$\n",
    "\n",
    "The network predicts $\\epsilon_\\theta(x_t, t)$ — the noise that was added — and we subtract it to go backward."
]))

cells.append(code([
    "# DDPM inference with diffusers\n",
    "from diffusers import DDIMPipeline\n",
    "import torch\n",
    "\n",
    "print(\"Loading DDPM pipeline (first run downloads model weights ~2GB)...\")\n",
    "pipe = DDIMPipeline.from_pretrained(\"google/ddpm-celebahq-256\")\n",
    "# Alternative smaller models:\n",
    "# pipe = DDIMPipeline.from_pretrained(\"google/ddpm-bedroom-256\")\n",
    "# pipe = DDIMPipeline.from_pretrained(\"google/ddpm-cat-256\")\n",
    "\n",
    "# Generate an image from noise\n",
    "print(\"Generating image from noise (this may take a minute)...\")\n",
    "torch_rng = torch.Generator().manual_seed(42)\n",
    "image = pipe(num_inference_steps=50, generator=torch_rng)\n",
    "generated_img = image.images[0]\n",
    "\n",
    "plt.figure(figsize=(8, 8))\n",
    "plt.imshow(generated_img)\n",
    "plt.title(\"Generated from Noise via DDPM\")\n",
    "plt.axis('off')\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]))

cells.append(md([
    "### Why Diffusion Works (Where GANs Failed)\n",
    "\n",
    "| Property | GAN | Diffusion |\n",
    "|----------|-----|-----------|\n",
    "| Training stability | Unstable (minimax game) | Stable (simple denoising MSE) |\n",
    "| Mode collapse | Common | Rare |\n",
    "| Generation speed | Fast (1 step) | Slow (50-1000 steps) |\n",
    "| Sample quality | Can be sharp | Slightly softer |\n",
    "| Diversity | Poor (mode collapse) | Excellent |\n",
    "\n",
    "> **The key insight:** Diffusion turns generation into a series of simple denoising steps. Instead of learning to generate a perfect image in one shot (GAN), it learns to remove a little noise at a time. Each step is easy, and the composition is magical."
]))

# ============================================================
# Section 12: JEPA Overview
# ============================================================
cells.append(md([
    "---\n",
    "\n",
    "## 12. JEPA — Predicting in Latent Space (2022–present)\n",
    "\n",
    "### The Problem with Pixel/Token Prediction\n",
    "\n",
    "LLMs predict the next token. Diffusion predicts pixels. Both operate at the **surface level** of data:\n",
    "- They try to reproduce every detail, including unpredictable noise\n",
    "- They waste capacity on details that don't matter for understanding\n",
    "- They can't reason about abstract concepts\n",
    "\n",
    "> **Yann LeCun's argument (2022):** To build true AI, we need models that learn **internal models of the world** — not just memorize surface patterns.\n",
    "\n",
    "### JEPA: Joint Embedding Predictive Architecture\n",
    "\n",
    "Proposed by Yann LeCun in \"A Path Towards Autonomous Machine Intelligence\" (Feb 2022).\n",
    "\n",
    "**Core idea:** Instead of predicting pixels or tokens, predict **abstract representations (embeddings)** in latent space.\n",
    "\n",
    "### How JEPA Works\n",
    "\n",
    "1. **Encode** context region -> embedding $s_x$\n",
    "2. **Encode** target region -> embedding $s_y$ (with stop-gradient)\n",
    "3. **Predict** target embedding from context: $\\hat{s}_y = \\text{Predictor}(s_x)$\n",
    "4. **Loss:** $L = \\|\\hat{s}_y - s_y\\|^2$ (L2 in latent space)\n",
    "\n",
    "The target encoder uses **EMA (Exponential Moving Average)** updates to prevent representation collapse.\n",
    "\n",
    "$$\\bar{\\theta} \\leftarrow \\tau \\bar{\\theta} + (1-\\tau) \\theta$$\n",
    "\n",
    "### JEPA vs. Alternatives\n",
    "\n",
    "| Method | Predicts | Space | Needs Negatives? |\n",
    "|--------|----------|-------|------------------|\n",
    "| Pixel reconstruction | Raw pixels | Pixel space | No |\n",
    "| Contrastive learning | Same embedding for augmentations | Embedding space | Yes (InfoNCE) |\n",
    "| **JEPA** | **Target embedding from context** | **Latent space** | **No** |\n",
    "| LLM | Next token | Token space | No |\n",
    "| Diffusion | Next denoised image | Pixel space | No |\n",
    "\n",
    "### The JEPA Family\n",
    "\n",
    "- **I-JEPA** (Jun 2023): Images — predict masked patch representations\n",
    "- **MC-JEPA** (Jul 2023): Motion + Content — joint learning of optical flow\n",
    "- **V-JEPA** (Feb 2024): Video — spatio-temporal prediction\n",
    "- **V-JEPA 2** (Jun 2025): Video + Robotics — world models for planning\n",
    "- **VL-JEPA**: Vision-Language — predict meaning, then decode text\n",
    "\n",
    "### Why JEPA Matters\n",
    "\n",
    "1. **Compute efficiency:** No pixel reconstruction overhead\n",
    "2. **Semantic learning:** Predicts in latent space -> learns meaning, not texture\n",
    "3. **No negative samples:** Unlike contrastive learning, no need for huge batches\n",
    "4. **World models:** The predictor IS a primitive world model — it models spatial/temporal uncertainty\n",
    "5. **Planning:** Predict in latent space -> simulate futures efficiently\n",
    "\n",
    "> **The big picture:** JEPA represents a shift from \"predict everything\" to \"predict what matters.\" The encoder discards unpredictable detail; the predictor learns causal structure. This is closer to how humans learn — we remember the structure of events, not every pixel of every scene."
]))

# ============================================================
# Final Summary
# ============================================================
cells.append(md([
    "---\n",
    "\n",
    "## Complete CV Timeline\n",
    "\n",
    "| Era | Year | Milestone | Key Idea |\n",
    "|-----|------|-----------|----------|\n",
    "| **Foundations** | 1957 | First digital photo | Image = pixel grid |\n",
    "| **Classical CV** | 1960s | RETINEX, edge theory | Edges > absolute brightness |\n",
    "| | 1986 | Canny edge detector | Optimal edge detection |\n",
    "| | 1988 | Harris corner detector | Repeatable keypoints |\n",
    "| | 1999 | SIFT (Lowe) | Scale-invariant features |\n",
    "| | 2006 | ORB | Fast, patent-free features |\n",
    "| | 2000s | Image stitching | Classical CV capstone |\n",
    "| **Deep Learning** | 2012 | AlexNet | Hierarchical feature learning |\n",
    "| | 2015 | ResNet | Skip connections, very deep nets |\n",
    "| | 2020 | ViT | Images as sequences |\n",
    "| **Generation** | 2014 | GANs | Adversarial generation |\n",
    "| | 2020 | DDPM | Stable diffusion |\n",
    "| **Understanding** | 2022 | JEPA (LeCun) | Predict in latent space |\n",
    "| | 2023+ | V-JEPA, VL-JEPA | World models for robotics |\n",
    "\n",
    "> **Final thought:** We started with Sobel filters and ended with models that predict in latent space. But the Sobel filter is still in every CNN first layer — just learned instead of hand-crafted. The story of computer vision is not replacement, but **automation and abstraction**: each new era builds on the insights of the previous one, scaling them up and automating the hand-crafting.\n",
    "\n",
    "---\n",
    "\n",
    "*End of Classical Computer Vision section.*"
]))

# ============================================================
# Build notebook
# ============================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("src/computer_vision/01_Classical_Computer_Vision.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

print(f"Written {len(cells)} cells to 01_Classical_Computer_Vision.ipynb")
