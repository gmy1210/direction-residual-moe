# REST-MDD (REST-meta-MDD)

1. Download the REST-meta-MDD release from the Psychological Science Data Bank. Follow the consortium instructions to email the signed Data Use Agreement to `rfmrilab@gmail.com` and obtain the archive password and phenotypic information. [Data release](https://doi.org/10.57760/sciencedb.o00115.00013) · [Consortium access instructions and agreement](https://rfmri.org/REST-meta-MDD)
2. Use the participant list in `Stat_Sub_Info_848MDDvs794NC.mat` under `Yan_2019_PNAS/StatsSubInfo`. This is the Yan et al. analysis cohort of 848 participants with MDD and 794 normal controls across 17 sites. [Original analysis files](https://github.com/Chaogan-Yan/PaperScripts/tree/master/Yan_2019_PNAS/StatsSubInfo)
3. For each included participant, use the unsmoothed, no-GSR `ROISignals_FunImgARCWF` derivative. Select columns 229 through 428 inclusive (MATLAB indexing, Python slice `[:, 228:428]`) to obtain the 200 CC200 ROI traces. Encode normal controls as 0 and MDD as 1. Keep subject and site IDs for the split. The benchmark recordings have 140–240 observed time points.

## Model input

See [the shared NPZ format](../../../README.md#input-data). Use `diremoe extract --dataset rest_mdd` with the prepared NPZ.
