# Computer Vision History Lesson — Brainstorm

**Duration:** 1.5–2.0h
**Format:** Single Jupyter notebook
**Narrative:** "From Pixels to Intelligence" — walking through CV history, building classical methods by hand, then exploring SoA model architectures via inference.
**Hands-on ratio:** 90% demo (pre-run, students follow along), 10% hands-on (students run code — light OpenCV ops).
**Execution:** Light OpenCV ops run hands-on by students on cluster. Heavy AI models (diffusion, JEPA, pretrained) all pre-run on GPU cluster.

---

## Part 1: The Classical Era (OpenCV + NumPy/JAX) — ~40 min

### Act 1 — "Seeing is not yet understanding" (1990s–2000s)

1. **What is an image?** — An image is a 3D tensor (H x W x C). Show it as a NumPy array. Connect to the project's existing parallel computing theme: image operations are embarrassingly parallel.

2. **Color spaces** — RGB -> Grayscale -> HSV -> Lab. Show how different spaces expose different structure.
   - *Story:* Early CV systems needed to separate illumination from reflectance. HSV separates intensity from chroma, making it robust to lighting changes.

3. **Edge detection** — Sobel -> Canny. Build a Canny edge detector from scratch using NumPy:
   - Gaussian blur -> gradient computation -> non-max suppression -> hysteresis thresholding
   - *Story:* The 1986 Canny paper defined the optimality criteria for edges. Edges as the "skeleton" of visual understanding.

4. **Corner detection** — Harris corner detector from scratch.
   - *Story:* Corners are the original "keypoints" — essential for image stitching, 3D reconstruction, motion tracking.

5. **Contours & shapes** — Find contours, approximate polygons, match shapes.
   - *Story:* The shape-from-contour era — before learning, geometry was king.

6. **Feature detection & matching** — ORB/SIFT descriptors + BFMatcher.
   - *Story:* The 2000s feature engineering golden age. SIFT won the competition, changed robotics, AR, panorama stitching.

### Act 2 — "Putting it together" (late 2000s)

7. **Image stitching / panorama** — ORB features -> BFMatcher -> homography -> warpPerspective -> multi-band blend.
   - *Story:* This is where classical CV peaked — you could build something genuinely useful with nothing but geometry and linear algebra.

8. **Template matching** — match a template in a larger image using TM_CCORR, TM_CCOEFF, etc.
   - *Story:* Simple but powerful — barcode readers, quality control on assembly lines still use this today.

---

## Part 2: The Deep Learning Revolution — ~30 min

### Act 3 — "Let the data teach us" (2012–2015)

9. **AlexNet (2012)** — Show the architecture diagram, explain why it won ImageNet (8% top-5 error drop). Key insight: hierarchical feature learning.
   - Use a pretrained model to run inference on an image.
   - Visualize what the first conv layer learned (edges, colors, bars).
   - *The "aha" moment:* AlexNet's first layer filters look remarkably like Gabor filters / Sobel operators from Part 1. Deep learning didn't replace classical CV — it automated it.

### Act 4 — "Going deeper, smarter" (2015–2020)

10. **ResNet (2015)** — The residual connection. Show the architecture, explain why skip connections solved the degradation problem. Run inference and show classification.

11. **Vision Transformer (ViT) (2020)** — Patch embeddings -> self-attention -> classification head. Explain how it treats images like sequences.
    - Compare: CNNs are local + translation-equivariant; ViTs are global + data-driven.

---

## Part 3: Understanding, Not Just Recognizing — ~30 min

### Act 5 — "What comes after classification?" (2021–present)

12. **JEPA (2022)** — Yann LeCun's **Joint Embedding Predictive Architecture**. Key idea: predict in latent space, not pixel/token space.
    - **History:** Proposed in LeCun's Feb 2022 paper "A Path Towards Autonomous Machine Intelligence." First concrete model: **I-JEPA** (Meta AI, June 2023, CVPR 2023) — a 632M ViT trained on ImageNet-1K in <72h on 16 A100s.
    - **How it works:** Mask parts of an image -> encode visible context with a ViT -> predictor (narrow ViT) forecasts the *embedding* of masked regions -> target encoder (EMA-updated, stop-gradient) provides stable targets -> L2 loss between predicted and target embeddings.
    - **Why it matters:** Unlike diffusion (predicts pixels) or LLMs (predicts tokens), JEPA predicts abstract representations. The encoder discards unpredictable surface detail; the predictor learns semantic structure. No negative samples needed (unlike contrastive learning), no reconstruction overhead (unlike autoencoders).
    - **Lineage:** I-JEPA (images, Jun 2023) -> MC-JEPA (motion+content, Jul 2023) -> V-JEPA (video, Feb 2024) -> V-JEPA 2 (video+robotics planning, Jun 2025).
    - **Demo idea:** Show I-JEPA's predictor output — it recognizes semantics of masked regions (e.g., "dog's head", "wolf's legs") without rendering pixels. A stochastic decoder can map predictions back to pixel space for visualization.
    - **Contrast:** Autoencoders reconstruct pixels (blurry, wasteful); contrastive learning needs negatives and huge batches; JEPA predicts latent codes directly, more compute-efficient, learns semantic features without hand-crafted augmentations.

13. **Diffusion models** — The forward noising process -> reverse denoising.
    - Show a diffusion model generating an image from noise (or at least a few denoising steps).
    - *Story:* The 2014 GAN era had mode collapse; diffusion won because it's stable training via score matching.

---

## Part 4: Synthesis — ~10 min

14. **Side-by-side comparison** — A table/plot comparing:

    | Era | Method | Features | Inference | Generative | Interpretability |
    |-----|--------|----------|-----------|------------|------------------|
    | Classical (2000s) | Sobel/Canny/SIFT | Hand-crafted | Fast | No | High |
    | CNN (2012–2017) | AlexNet/ResNet | Learned hierarchical | Fast | No | Medium |
    | Transformer (2020+) | ViT | Learned global | Medium | No | Medium |
    | Diffusion (2020+) | DDPM/Stable Diffusion | N/A | Slow | Yes | Low |
    | JEPA (2022+) | i-JEPA | Latent prediction | Fast | No | Low-Medium |

15. **Final thought** — "We started with Sobel filters and ended with models that dream. But the Sobel filter is still in every ResNet first layer — just learned instead of hand-crafted."

---

## Dependencies to Add

```toml
"opencv-python-headless",   # Classical CV operations
"timm",                     # Pretrained vision models (AlexNet, ResNet, ViT)
"diffusers",                # Diffusion model inference
# Note: JEPA official code is at facebookresearch/ijepa (not in transformers/timm)
"torch",                    # PyTorch backbone for timm/diffusers
```

**Note on JAX:** The project has `jax[cuda12]` but most SoA vision models (timm, diffusers) are PyTorch-native. Strategy:
- Use **JAX/NumPy** for the classical CV math (convolutions, gradient computation for Sobel/Canny from scratch) — ties into the project's existing deps.
- Use **PyTorch** for pretrained model inference (practical necessity — timm/diffusers are PyTorch-first).
- Optional: Show a tiny CNN trained from scratch in JAX as a "from-scratch" bridge between the two worlds.

---

## Key Design Decisions

1. **Classical CV is hands-on** — Build Canny edge detector, Harris corner detector, panorama stitcher from scratch using NumPy/OpenCV.
2. **Deep learning is inference-first** — No training from scratch (too slow for a 2h session). Use pretrained models, visualize internals.
3. **The through-line is "what does the model see?"** — Connect each era back to classical intuition (e.g., "AlexNet layer 1 = learned Sobel").
4. **JAX appears in the classical section** — Use it for the math operations (convolutions, gradients) to tie into the project's existing JAX/CuPy deps.
5. **Architecture diagrams** are drawn with matplotlib (simple block diagrams) rather than external images — keeps the notebook self-contained.

---

## Potential Concerns

- **Size:** This is a LOT for 1.5–2h. The classical CV section alone could fill 2h. Some parts need to be "read-only" demos vs interactive.
- **Model downloads:** First run will need to download pretrained weights (hundreds of MB). Could pre-warm or use tiny models (MobileNet instead of ResNet-152).
- **GPU vs CPU:** Diffusion on CPU is painfully slow. Need GPU for the diffusion demo.
- **PyTorch in a JAX project:** Feels slightly inconsistent, but practical. Could frame it as "JAX for building, PyTorch for borrowing" — build the classical math in JAX, use PyTorch for pretrained inference.
- **Notebook file size:** Pretrained models downloaded during the session will bloat the notebook. Use code-based downloads, not embedded weights.

---

## Decisions Made

- **Hands-on ratio:** 90% demo, 10% hands-on (for 35 students on cluster — minimize connection issues)
- **Diffusion:** DDPM (simple, educational)
- **JEPA:** Overview only — architecture explanation + high-level demo
- **Small CNN:** Yes — Flax NNX CNN to compare classical vs AI workflows
- **Image dataset:** ImageNet-1K for pretrained model inference
- **Part 1 priority:** Classical CV with OpenCV, telling the story through running code = success
- **JAX for CNN:** Flax NNX (Flax 0.8+)

## Notebook Structure (01_Classical_Computer_Vision.ipynb — 57 cells)

| # | Section | Type | Content |
|---|---------|------|---------|
| 0-5 | Intro + Synthetic Images | Code | Generate test scenes with NumPy/OpenCV |
| 6-9 | 1. What Is an Image? | Code | Array inspection, matrix operations |
| 10-13 | 2. Color Spaces | Code+Hands-on | BGR/RGB/Gray/HSV/Lab, saturation segmentation |
| 14-19 | 3. Edge Detection | Code+Hands-on | Sobel from scratch, Canny pipeline, threshold experiment |
| 20-22 | 4. Corner Detection | Code | Harris corner detector, visualization |
| 23-24 | 5. Contours & Shapes | Code | findContours, approxPolyDP, shape classification |
| 25-28 | 6. Feature Matching | Code | ORB detection, BFMatcher, Lowe's ratio test |
| 29-33 | 7. Image Stitching | Code | Panorama pair, homography (RANSAC), warp & blend |
| 34 | Classical Summary | Markdown | Timeline table, bridge to deep learning |
| 35-43 | 8. CNN with Flax NNX | Code | SimpleCNN architecture, synthetic dataset, JIT training loop |
| 44-48 | 9. AlexNet/ResNet + ImageNet | Code | Pretrained ResNet-18 inference, first-layer filter visualization |
| 49-51 | 10. Vision Transformer | Code | ViT inference, self-attention explanation |
| 52-54 | 11. DDPM Diffusion | Code | DDIMPipeline generation, GAN vs Diffusion comparison |
| 55 | 12. JEPA Overview | Markdown | Architecture explanation, family tree, why it matters |
| 56 | Complete CV Timeline | Markdown | Full timeline from 1957 to present |
