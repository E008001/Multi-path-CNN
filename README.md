## Multi-path CNN
Multi – path Convolution Neural Networks for Lung Cancer Detection
## The Problem 
lung cancer strikes 225,000 people every year in the United States,and more and more in all ao the world. Early detection is critical to give patients the best chance at recovery and survival.   

In 2017, the Data Science Bowl will be a critical milestone in support of the Cancer Moonshot by convening the data science and medical communities to develop lung cancer detection algorithms.  

Using a data set of thousands of high-resolution lung scans provided by the National Cancer Institute, participants will develop algorithms that accurately determine when lesions in the lungs are cancerous. This will dramatically reduce the false positive rate that plagues the current detection technology, get patients earlier access to life-saving interventions, and give radiologists more time to spend with their patients. 

### Lung cancer
According to the World Health Organization (WHO) report 2018, lung cancer is responsible for an estimated 1.76 million deaths  
This number is expected to be higher in developing countries.  
the finest solution for lung cancer is early diagnosis and treatment. To this end, the primary and critical step for early diagnosis and treatment of lung cancer is identifying the lung whether it is infected by cancer or not, with better screening approaches leading to polished patient result.  

### Data Science Bowl challenge
![kaggle data science](https://github.com/E008001/Multi-path-CNN/blob/master/breath.jpg)
The Data Science Bowl, presented by Booz Allen and Kaggle, is the world’s premier data science for social good competition. It convenes data scientists, technologists, domain experts, and organizations to take on the world’s challenges with data and technology. It’s a platform through which individuals can harness their passion, unleash their curiosity, and amplify their impact to effect change on a global scale.

### Dataset
This is a project to detect lung cancer from CT scan images using Deep learning (CNN) 

KDSB dataset - https://www.kaggle.com/c/data-science-bowl-2017/data

LUNA dataset-https://luna16.grand-challenge.org/download/

## Data & Pre-processing
The competition organizers have provided 2 categories of data sets.  
The first category is a set of images of the CT scans of different patients.  
The second aspect of the dataset involves a set of labels for the patients. The number of CT scan images for every patient is not fixed , the number of the images is different  
Since the data provided by the contributors are DICOM files of patients’ CT scans, it involves complicated pre-processing methods in order to get into the form that is usable to apply deep learning and machine learning methodologies to it.  
  
An overview of the chronology of tasks involved in the preprocessing steps are as follows:  
Loading the DICOM files: Pixel information for each file and the respective metadata from each scan of each patient is extracted and is extremely useful.  

Adding missing metadata: Missing metadata of the 'z' axis is inferred.  
Converting the pixel values to Hounsfield Units (HU): Pixel units are converted to Hounsfeld Units, which represent the density of the matter shown in that position in the scan.   
Lung Segmentation: Lung segmentation is a process to identify boundaries of lungs in a CT scan image. Lung Tissue, Blood in Heart, Muscles and other lean tissues are removed by thresholding the pixels, setting a particular color for air background and using dilation and erosion operations for better separation and clarity.   

Normalization: Normalization is a good approach in deep learning and particularly for this problem, since it involves Hounsfeld Units, the pixel values need to be normalized.  
Zero Centering: Zero Scaling is important to reduce the effect of the scaling differenced and depend only on the content of the images. 

### U-Net Network
![unet](https://github.com/E008001/Multi-path-CNN/blob/master/unet.png)
Image segmentation using U-Net
During training, the modified U-Net takes as an input 256×256 2D CT slices, and their corresponding labels are provided by masking 256×256, where nodule pixels are 1 and the
rests are 0.  
The output of the model is an image having the same size with an input. Each pixel of the output has a value between 0 and 1, showing the probability the pixel belongs to a nodule.  
This is utilized by taking the slice belongs to label 1 of the softmax of the final U-Net layer. 
Finally, the trained U-Net is then used to segment the KDSB CT scan slices. These
candidates have variable size (small, medium and large) and shape (circular, elliptical and
others), where we categorized them into training set, validation set, and test set to train the
proposed mp-CNN

## Model Training

### Multi-path CNN architecture
![mp-cnn](https://github.com/E008001/Multi-path-CNN/blob/master/fig7.jpg)
Multi-path CNN architecture design is a CNN Network having multi-path convolutional
layers, the path considering smaller, medium, and larger receptive field sizes. these paths first, second, and third path, respectively.  
That the receptive field size of the first path is 3×3, the second path is 5×5, and the third paths is 7×7.
