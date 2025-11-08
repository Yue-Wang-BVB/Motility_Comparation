# Motility_Comparation
Unified NDMI for Small Bowel Cine MRI: Repeatability and Sensitivity to Multicenter Variability

In the example, we provide a partial segment of a sample sequence to illustrate the analysis of noise effects in the following steps:

Run generate_noise_sequences.py to create noisy versions of the sequence.

Run compute_FB_clahe.py to simulate the optical flow algorithm used in the software and generate flow fields.

Run visualize_flow_field.py to visualize the computed flow fields.

Run evaluate_MI_jacobian.py to compute motion evaluation metrics and plot the frame-to-frame MI curve; the results are saved in summary.csv.

Similarly, after obtaining summary.csv files for the morphological and INR-based methods, run compute_sensitivity.py to perform sensitivity analysis of motion estimation errors.
