# Peptide identity and abundance analyze from raw data and pretrain model for predict abundance

In the realms of proteomics and bioinformatics, the accurate identification and quantification of peptides are pivotal. This project, "Peptide Identity and Abundance Analysis from Raw Data and Pretrained Model for Predict Abundance," aims to unravel the intricate details of peptide sequences and their abundance, integrating both conventional analytical workflows and advanced machine learning models. Through a synergy of OpenMS-based workflows and BERT-based pretrained models, we endeavor to provide a comprehensive solution that caters to both the accuracy and efficiency required in modern proteomic analysis.

## Get peptides and abundance from raw data

To get the sequences and abundances from raw data. There are three ways. 

### OpenSwathWorkflow

The OpneSwath workflow has high accuracy, but the workflow costs time. To run this workflow, you first need to download OpenMS, you can download by the following link, https://openms.readthedocs.io/en/latest/openms-applications-and-tools/installation.html. Then, you can follow paper “Building high-quality assay libraries for targeted analysis of SWATH MS data” procedure part to run the prepare and identity part. (https://www.nature.com/articles/nprot.2015.015#Sec18). After procedures in the paper, you can run the last quantity analyze, which like

```shell
OpenSwathWorkflow.exe
-in data.mzML -tr library.tsv
-tr_irt iRT_assays.TraML
-swath_windows_file SWATHwindows_analysis.tsv
-sort_swath_maps -batchSize 1000
-readOptions cacheWorkingInMemory -tempDirectory C:\Temp
-use_ms1_traces
-mz_extraction_window 50
-mz_extraction_window_unit ppm
-mz_correction_function quadratic_regression_delta_ppm
-TransitionGroupPicker:background_subtraction original
-RTNormalization:alignmentMethod linear
-Scoring:stop_report_after_feature 5
-out_tsv osw_output.tsv
```

### OpenmsLFQ

This workflow is faster than the OpneSwath and it is also based on OpenMS, so you also need to download the OpenMS(see first part). I provided a built workflow named openmsLFQ.knwf you can directly use. To use this built workflow, you need to download an analytics platform KNIME(https://www.knime.com/downloads). After downloading it, you can open the file I provided. 
To run this workflow, you First need to convert the raw data to mzML data.  To convert the raw data, you can use MSconvert, you can download from (https://proteowizard.sourceforge.io/download.html). To use MSconvert, you can add the raw files and choose the output file data type mzML, and then run and get the mzML data.
After you get mzML data, you can run the openmsLFQ workflow. First, you can add the mzML data in input files. At the same time, you also need to add FASTA database, and I provide the uniport.fasta for you. Then you can run the workflow and get the result.

### Maxquant
Maxquant is the easiest workflow, but the peptide search results are not complete.To use this workflow, you can download the maxquant from (https://www.maxquant.org/download_asset/maxquant/latest) . To run this workflow, you can first load the raw data. Then, click no fraction to set experiment. After that, set quantity index. You can click label-free quantification and choose LFQ. Then, click global parameters and add  the uniport.fasta I provided to you. Then you can run the workflow and get the result.


## Bert-based Pretrain model

This section delves into the utilization of pre-trained BERT models, specifically focusing on the intricacies of applying fine-tuned Camembert and Distilbert models to the data procured. These models are meticulously adapted and established with the aim of predicting the abundance values of peptides, demonstrating a harmonious blend of machine learning and bioinformatics. The method involves the technique of masking sequences, a strategy that enables the models to comprehend and infer the contextual relationships within the peptide sequences, thereby enhancing the accuracy and reliability of the abundance predictions. The application of these advanced models exemplifies a pivotal step in leveraging computational intelligence to unravel the complexities of proteomic data.

### Installation
After cloning this repo, please enter the folder and run:
```shell
pip install -r requirements.txt
```

## Usage

Run Distilbert

```shell
python Distilbert.py
```


Run Camembert

```shell
python Camembert.py
```
