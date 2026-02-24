# CW2 Constraints & Requirements

## Submission Requirements (up to 10% penalty if not followed)
- Single Word/PDF document with questions in ascending order
- Explain all mathematical reasoning in detail for every step
- All calculations, graphs, and figures must be labelled with relevant parameters and units
- Python code must appear in an Appendix AND as separate .py source files
- Code must be commented explaining each step — source code is assessed

## Technical Constraints
- Libraries: scikit-image, scikit-learn, scipy, opencv-python
- Environment: sds anaconda environment
- Missing data: 5 images at [8][9], [12][7], [13][9], [14][0], [15][3] have score = -1 and must be ignored
- Participants 1-7 = expert group, Participants 8-20 = novice group

## Question-Specific Constraints
- Q3ii: Regression coefficients smaller than 0.01 should be treated as not contributing
- Q4: Must use MOTION_EUCLIDEAN (2x3 warp matrix) with cv2.warpAffine, NOT the homography mode in image_align2.py

## Academic Integrity
- Do not share or copy solutions or code from peers, even partially
- Do not publish assessment materials on external forums or homework help sites
- Research and literature references are encouraged but must be cited
