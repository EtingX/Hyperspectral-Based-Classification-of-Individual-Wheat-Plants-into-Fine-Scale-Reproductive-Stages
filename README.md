# Hyperspectral-Based-Classification-of-Individual-Wheat-Plants-into-Fine-Scale-Reproductive-Stages
This is share code and dataset for 'Hyperspectral-Based-Classification-of-Individual-Wheat-Plants-into-Fine-Scale-Reproductive-Stages'

Field trials are critical in the development of genetically modified and genome-edited biotechnology plants to evaluate the growth and yield of breeding lines and to test commercial viability or any potential off-target effects. In Australia, conducting field trials of biotechnology derived crops requires compliance with federally mandated regulations, including strict protocols for forecasting flowering times. Conventional practices are based on time consuming, subjective and costly visual field inspections of individual wheat plants at respective growth stages (Zadoks growth stages Z37, Z39, and Z41). To enable automatic forecasting, hyperspectral and RGB images were captured in the greenhouse, and hyperspectral reflectance data were acquired in a semi-natural environment. In the greenhouse, imaging was conducted under controlled lighting with a fixed top-view setup; in semi-natural environments, spectral data were collected manually from multiple oblique angles under supplemented natural light. Support Vector Machine classification achieved F1 scores above 0.8 for anthesis prediction when reflectance data were transformed using Standard Normal Variate, Hyper-hue, or Principal Component Analysis. After feature selection, F1 scores above 0.75 could be achieved with only five wavelengths. Furthermore, the SNV transformation demonstrated robust performance under limited training conditions, maintaining high classification accuracy and strong generalizability across varying data sizes. These findings highlight the effectiveness of transformation-enriched data and optimized feature selection for accurate growth stage classification. This study provides a low-cost approach to alleviate manual inspection burdens, improve regulatory compliance, and increase biosafety during biotechnology field trial practices.

Key words: hyperspectral sensing, individual wheat phenotyping, fine-scale growth stage classification, reproductive development, data transformation

Result:

<img width="3600" height="2400" alt="Figure 4" src="https://github.com/user-attachments/assets/e2a7860f-c006-412c-ae77-dc4e326d5e31" />


Model performance on data collected under controlled environment conditions. (a) F1 scores of deep learning models trained on RGB images captured from side and top views. (b) F1 scores of classical machine learning models trained on hyperspectral reflectance data. Validation results are shown as solid grey bars, while test results are indicated by hatched bars (//). The dashed horizontal line at 0.75 marks the threshold for acceptable predictive performance.

<img width="3000" height="2400" alt="Figure 6" src="https://github.com/user-attachments/assets/0d6207d5-a8d5-4de3-8312-1478bff0ca1d" />
 

Results of hyperspectral data analysis from semi-naturally grown plants. F1 scores for the SVM analysis of the Full (a), Early (b), Mid (c), and Late (d) hyperspectral datasets are presented. The abbreviations ‘org’, ‘hyp’, ‘snv’, and ‘pca’ represent the original reflectance, Hyper-hue transformed data, SNV transformed data, and PCA transformed data, respectively. Combinations such as ‘org+hyp’ indicate that the training dataset included both original reflectance and Hyper-hue transformed data. The red dashed line marks the 0.80 threshold for satisfactory performance, while the yellow dashed line indicates the 0.75 threshold for acceptable performance.

<img width="5117" height="4235" alt="Figure 8" src="https://github.com/user-attachments/assets/56ce562f-2bbd-4c08-ad56-612beca62f1a" />

F1 scores for ‘SNV’ and ‘Hyper-hue’ data transformations across different training ratios used in bootstrap analysis, ranging from 0.25 to 3.00.

