# Egg Segmentation and Tracking (CV Intern Task)

Segmenting eggs with a pretrained model (SAM 3.1), with a classical image-processing baseline, evaluating the
baseline against SAM, and tracking two eggs in a video.

- **Full write-up (methods, experiment log, evaluation, analysis):** [`report/CV_Intern_Task_Report.pdf`](report/CV_Intern_Task_Report.pdf)
- **Task 4 output video (public Google Drive):** `<paste your public Drive link here>`
- A copy of the same video is in [`results/egg_tracking_sam3_1_stable_ids.mp4`](results/egg_tracking_sam3_1_stable_ids.mp4)

## Results at a glance

| Task | Method | Result |
|------|--------|--------|
| 1 | SAM 3.1, text prompt `"egg"`, confidence >= 0.50 | 110/110 images, 137 egg instances, 0 errors |
| 2 | HSV threshold (S <= 65, V >= 165) + morphology + component filters, chosen from 8 candidates on the dev split | Dev macro Dice 0.475 / IoU 0.382 |
| 3 | Frozen Task 2 method vs. SAM masks on 22 held-out images | Macro Dice 0.553, IoU 0.453, boundary F1 (2 px) 0.267 |
| 4 | SAM 3.1 per frame + Hungarian-assignment tracker with 2 persistent IDs | IDs 1 and 2 matched in 66 and 65 of 120 frames |

SAM masks are used as **pseudo ground truth**, so Task 3 measures agreement with SAM, not with human labels.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── report/
│   └── CV_Intern_Task_Report.pdf
├── notebooks/
│   ├── Task_1_Pretrained_SAM_3_1_egg_segmentation.ipynb
│   ├── Task_2_Classical_image_processing_segmentation.ipynb
│   ├── Task_3_Held_out_evaluation_and_error_analysis.ipynb
│   └── Task_4_Video_segmentation_and_persistent_tracking_of_eggs.ipynb
└── results/                      # small outputs copied from Google Drive (see "Outputs")
    ├── egg_tracking_sam3_1_stable_ids.mp4
    ├── sam3_instance_manifest.csv
    ├── task2_split_manifest.csv
    ├── near_duplicate_pairs_review.csv
    ├── task2_experiment_log.csv
    ├── selected_classical_config.json
    ├── task3_eval_metrics_per_image.csv
    ├── task3_eval_summary.json
    ├── egg_tracking_sam3_1_tracks.csv
    └── egg_tracking_sam3_1_summary.json
```

## What you need

The notebooks are written for **Google Colab** (they mount Google Drive and use Colab file helpers).

| Requirement | Needed for | Details |
|-------------|-----------|---------|
| Google account + Colab | all tasks | Notebooks mount Drive at `/content/drive` |
| GPU runtime (T4 or better) | Tasks 1 and 4 | Python >= 3.12, PyTorch >= 2.7, CUDA >= 12.6 (the first cell checks this and stops with a message if not met) |
| Hugging Face account with access to `facebook/sam3.1` | Tasks 1 and 4 | The checkpoint is gated: request access on the model page, then create a **read** token |
| Dataset ZIP | all tasks | Roboflow *egg* dataset, "folder" export, saved as `egg.v1i.folder.zip` (110 images) |

Tasks 2 and 3 need only a CPU once the Task 1 masks exist.

## Step-by-step: reproducing the experiments

### 0. Prepare Google Drive

1. Download the dataset ZIP from the link in the assignment (Roboflow link or the alternative zip link).
2. In your Google Drive, create the folder `MyDrive/egg_cv_assignment/` and put the ZIP there with the exact name
   `egg.v1i.folder.zip`.
3. Request access to <https://huggingface.co/facebook/sam3.1> and create a Hugging Face **read** token.
   Do not paste the token into a notebook or commit it; the notebooks call `login()` and ask for it interactively.

All outputs are written to `MyDrive/egg_cv_assignment/` (SAM outputs under `sam3_1_outputs/`, video outputs under `video_tracking/`).

### 1. Task 1 - SAM 3.1 segmentation (GPU, about 11 minutes on a T4)

1. Open `notebooks/Task_1_...ipynb` in Colab, set **Runtime > Change runtime type > T4 GPU**.
2. Run all cells in order. Paste your Hugging Face token when `login()` asks. The 3.5 GB checkpoint is downloaded on first use.
3. Output: per-instance masks, union masks (`combined_masks/`), overlays, JSON metadata and `sam3_instance_manifest.csv`.

### 2. Task 2 - classical segmentation (CPU)

1. Open `notebooks/Task_2_...ipynb`.
2. The notebook is cumulative: its first cells repeat Task 1. **If the Task 1 outputs are already on Drive, skip to the cell
   after the markdown "Task 2 Code Start From Here"** (run the Drive-mount cell of that section first).
3. This creates the grouped 88 : 22 dev/eval split (SHA-256 + perceptual-hash groups, seed 2026), runs the 8 experiments on the dev
   split only, and saves `task2_experiment_log.csv`, `selected_classical_config.json` and the split manifest under
   `sam3_1_outputs/task2_task3_results/`.

### 3. Task 3 - held-out evaluation (CPU)

1. Open `notebooks/Task_3_...ipynb`. It needs the Task 1 masks and the Task 2 files above on Drive.
2. Skip to the cell after the markdown "Task 3 Code Starts From Here" if those exist (it reloads the split, experiment log and frozen
   configuration from Drive).
3. This applies the frozen configuration to the 22 evaluation images, writes `task3_eval_metrics_per_image.csv` and
   `task3_eval_summary.json`, the histogram, the IoU-vs-area plot and the error-map figures for the 8 worst images
   (`failure_visualizations/`), and saves classical masks for all images.

### 4. Task 4 - video tracking (GPU)

1. Open `notebooks/Task_4_...ipynb` on a T4 runtime and run all cells.
2. The video (<https://www.youtube.com/watch?v=Y4JyV94y2_A>) is downloaded with `yt-dlp`. If YouTube blocks the Colab download, the
   notebook asks you to upload the MP4 by hand.
3. For memory reasons the video is re-encoded to 2 fps, 640 px wide (120 frames) before SAM 3.1 runs on every frame. The tracker keeps
   two persistent IDs (see the report, Section 6). You can raise `INFERENCE_FPS` in the preprocessing cell if you have more RAM.
4. Output: `video_tracking/egg_tracking_sam3_1_stable_ids.mp4`, `..._tracks.csv`, `..._summary.json`.
5. Upload the MP4 to Google Drive, set sharing to **Anyone with the link can view**, and paste the link in the report and at the top of this README.

### Outputs to commit

After running, copy the small result files (listed under `results/` above) from `MyDrive/egg_cv_assignment/` into this repo.
The mask folders and overlay images are large and are better left on Drive; link to them if needed.

## Running without Colab (untested)

Tasks 2 and 3 only use OpenCV, scikit-learn and pandas, so they can run locally with `pip install -r requirements.txt` if you
(a) replace the Drive-mount cell with local paths for `PROJECT_DIR` and the dataset ZIP, and (b) place the SAM combined masks
under `<PROJECT_DIR>/sam3_1_outputs/combined_masks/`. Tasks 1 and 4 need a CUDA GPU and the SAM 3 package
(`git clone https://github.com/facebookresearch/sam3.git && pip install -e sam3`).

## Known limitations

- The reference masks come from SAM 3.1, not from human annotation; one evaluation image has an empty SAM mask and is scored as a
  perfect 1.0 by the metric code (see report, Section 4).
- The classical pipeline fails on dim eggs and on bright, low-saturation backgrounds (report, Section 5).
- Task 4 uses SAM 3.1 for the video (not the Task 2 pipeline), runs at 2 fps, is hard-coded for two eggs and has no frame-level
  ground truth, so ID switches and recall are not measured.
- Package versions printed in the notebooks (Python 3.13, PyTorch 2.14, OpenCV 4.14, scikit-learn 1.6.1) are the ones used for the
  reported results; the Colab pip step reports numpy version conflicts that did not stop the runs.

## AI tools

Claude (Anthropic) was used to help draft the report and this README from the executed notebook outputs.
*Add any other AI tools you used for the notebook code here.*
The author is responsible for the methods and can explain them.