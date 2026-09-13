AIRCRAFT MAINTENANCE PREDICTION

CS610 Applied Machine Learning – Section G2

Group 10



Group Members:



\* Kimberly Bertoli

\* Do Thi Phuong Linh

\* Nguyen Kim Hau

\* Tanya Arora

\* Cheung See Lin



\## Submission Contents



The submission package consists of three main folders:



1\. **input**

Contains the datasets (.tar) used for data preprocessing, exploratory analysis, model development, training, and evaluation.



2\. **ipynb**

Contains the Jupyter Notebook (.ipynb) files used throughout the project. The notebooks were developed and executed collaboratively on Kaggle and subsequently downloaded for submission.

&#x20;  - 01\_EDA

&#x20;    Input: Raw Dataset

&#x20;  - 02\_Data Preprocessing

&#x20;    Input: Raw Dataset

&#x20;  - 03\_XGB

&#x20;    Input: Aggregated tabular features *(tabular\_features\_scaled.csv)*

&#x20;  - 04\_MiniRocket Features Extract

&#x20;    Input: Data Preprocessing pipeline outputs *(sequences.npy, seq\_indices.npy, labels)*

&#x20;    Output: Chunked MiniRocket-transformed feature matrices (not included in submission due to file size

&#x20;    limitations)

&#x20;  - 05\_MiniRocket

&#x20;    Input: MiniRocket extracted features generated from the MiniRocket Feature Extraction notebook and

&#x20;    preprocessing outputs *(sequences.npy, seq\_indices.npy, labels)*

&#x20;  - 06\_InceptionTime

&#x20;    Input: Data Preprocessing pipeline *(sequences.npy, seq\_indices.npy, labels)*

&#x20;  - 07\_Transformer

&#x20;    Input: Data Preprocessing pipeline *(sequences.npy, seq\_indices.npy, labels)*

&#x20;  - 08\_Ensemble

&#x20;    Input: Model predictions *(XGB.csv, MiniRocket.csv, InceptionTime.csv, Transformer.csv)*



3\. **output**

Contains the generated outputs, including processed datasets, model results, evaluation metrics, figures, and other artefacts produced during experimentation.

&#x20;  - Data Preprocessing: Includes extracted time-series sequences with indices (npy), aggregated tabular features

&#x20;    scaled (CSV), labels (CSV)

&#x20;  - Predictions: Includes each model's and ensemble prediction CSVs



\## Data Source



The input dataset was obtained from Kaggle:



Aviation Maintenance Dataset from the NGAFID

https://www.kaggle.com/datasets/hooong/aviation-maintenance-dataset-from-the-ngafid



The dataset is derived from the National General Aviation Flight Information Database (NGAFID) and was used as the primary data source for this project.



\## Development Environment



All group members worked collaboratively using the Kaggle platform for coding, experimentation, and model development. The submitted notebooks are exported versions of the notebooks used on Kaggle.



\## Notes



The submitted files represent the final versions used for the project report and analysis. Folder names and file structures have been retained to facilitate reproducibility and ease of review.



MiniRocket Extracted Features generate MiniRocket-transformed features from the extracted time-series sequences. Due to the large file size of the transformed feature matrices, these outputs are not included in the submission package. The feature extraction notebook employs chunked processing to generate the features efficiently in batches. To reproduce the MiniRocket model results, the MiniRocket Feature Extraction notebook should be executed first, and the generated features should then be supplied as input to the MiniRocket Model notebook.

