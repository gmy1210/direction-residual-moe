# ABIDE I (PCP, QC-871)

1. Get the phenotype CSV and join its `FILE_ID` to the preprocessed ROI files. Use the C-PAC pipeline, the `filt_noglobal` strategy (band-pass filtering, no global signal regression), and the `rois_cc200` derivative. [Phenotype CSV](https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv) · [PCP download instructions](https://preprocessed-connectomes-project.org/abide/download.html)
2. Apply the QC-871 filters. Keep rows with an available `FILE_ID`, `qc_rater_1 = OK`, both `qc_anat_rater_2` and `qc_func_rater_2` in `{OK, maybe}`, and both `qc_anat_rater_3` and `qc_func_rater_3` equal to `OK`. The benchmark contains 871 participants, with 403 ASD and 468 controls. [PCP quality assessment](https://preprocessed-connectomes-project.org/abide/quality_assessment.html)
3. Download one `.1D` file per retained row using the URL below. Read the numeric ROI values and preserve the column order and native time length. Map `DX_GROUP=1` (ASD) to label 1 and `DX_GROUP=2` (control) to label 0. Keep `SITE_ID` for site-stratified splits. The benchmark recordings have 78–316 observed time points.

```text
https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_noglobal/rois_cc200/<FILE_ID>_rois_cc200.1D
```

## Model input

See [the shared NPZ format](../../../README.md#input-data). Use `diremoe extract --dataset abide` with the prepared NPZ.
