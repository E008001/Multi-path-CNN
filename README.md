## Multi-path CNN
Multi – path Convolution Neural Networks for Lung Cancer Detection  
https://link.springer.com/article/10.1007/s11045-018-0626-9
## The Problem 
lung cancer strikes 225,000 people every year in the United States,and more and more in all of the world. Early detection is critical to give patients the best chance at recovery and survival.   

In 2017, the Data Science Bowl will be a critical milestone in support of the Cancer Moonshot by convening the data science and medical communities to develop lung cancer detection algorithms.  

Using a data set of thousands of high-resolution lung scans provided by the National Cancer Institute, participants will develop algorithms that accurately determine when lesions in the lungs are cancerous. This will dramatically reduce the false positive rate that plagues the current detection technology, get patients earlier access to life-saving interventions, and give radiologists more time to spend with their patients. 

## Lung cancer
According to the World Health Organization (WHO) report 2018, lung cancer is responsible for an estimated 1.76 million deaths  
This number is expected to be higher in developing countries.  
the finest solution for lung cancer is early diagnosis and treatment. To this end, the primary and critical step for early diagnosis and treatment of lung cancer is identifying the lung whether it is infected by cancer or not, with better screening approaches leading to polished patient result.  
#### Screening (The importance of lung cancer screening)
The finest solution for lung cancer is early diagnosis and treatment
The national lung screening trial NLST determined that screening with CT scan decreased death rate by 20%.  
[NLST - National Lung Screening Trial](https://www.cancer.gov/types/lung/research/nlst)
#### Low-dose computed tomography (LDCT). 
The only recommended screening test for lung cancer is low-dose computed tomography (also called a low-dose CT scan, or LDCT). In this test, you are lying down and moved through a donut-shaped X-ray machine while holding your breath. A low-dose CT scan for lung cancer screening uses no dyes, no injections, and requires nothing to swallow by mouth. The scan is called “low dose” because radiation exposure is less than with a standard CT scan. The LDCT scan takes several X-ray images of the lungs, and a computer combines the images for interpretation by a radiologist.  
#### Who Should Be Screened for Lung Cancer?
Lung cancer screening is recommended only for adults who have no symptoms but who are at high risk for developing the disease because of their smoking history and age.  
more information: [Basic Information About Lung Cancer](https://www.cdc.gov/cancer/lung/basic_info/)  

### Data Science Bowl challenge  

![kaggle data science](https://github.com/E008001/Multi-path-CNN/blob/master/breath.jpg)
The Data Science Bowl, presented by Booz Allen and Kaggle, is the world’s premier data science for social good competition. It convenes data scientists, technologists, domain experts, and organizations to take on the world’s challenges with data and technology. It’s a platform through which individuals can harness their passion, unleash their curiosity, and amplify their impact to effect change on a global scale.

## Dataset
This is a project to detect lung cancer from CT scan images using Deep learning (CNN) 

KDSB dataset - https://www.kaggle.com/c/data-science-bowl-2017/data

LUNA dataset-https://luna16.grand-challenge.org/download/

In this dataset, you are given over a thousand low-dose CT images from high-risk patients in DICOM format. Each image contains a series with multiple axial slices of the chest cavity. Each image has a variable number of 2D slices, which can vary based on the machine taking the scan and patient.

The DICOM files have a header that contains the necessary information about the patient id, as well as scan parameters such as the slice thickness.

###  DICOM Format
DICOM — Digital Imaging and Communications in Medicine — is the international standard for medical images and related information. It defines the formats for medical images that can be exchanged with the data and quality necessary for clinical use.  
DICOM is implemented in almost every radiology, cardiology imaging, and radiotherapy device (X-ray, CT, MRI, ultrasound, etc.), and increasingly in devices in other medical domains such as ophthalmology and dentistry. With hundreds of thousands of medical imaging devices in use, DICOM is one of the most widely deployed healthcare messaging Standards in the world. There are literally billions of DICOM® images currently in use for clinical care.  
[About DICOM](https://www.dicomstandard.org/)

## Data & Pre-processing  

The competition organizers have provided 2 categories of data sets.  
The first category is a set of images of the CT scans of different patients.  
The second aspect of the dataset involves a set of labels for the patients. The number of CT scan images for every patient is not fixed , the number of the images is different(around 100–400)    
Since the data provided by the contributors are DICOM files of patients’ CT scans, it involves complicated pre-processing methods in order to get into the form that is usable to apply deep learning method to it.  
  
An overview of the chronology of tasks involved in the preprocessing steps are as follows:  
Loading the DICOM files: Pixel information for each file and the respective metadata from each scan of each patient is extracted and is extremely useful.  

data augmentation: The KDSB 2017 dataset is highly data imbalanced, To circumvent this, first perform data augmentation on images whose label is 1 ( augment the set of malignant
nodules by filliping and 90-degree rotations) and perform training. Next,  considering the unbalanced data and retrain only the output layer  

Converting the pixel values to Hounsfield Units (HU): Pixel units are converted to Hounsfeld Units, which represent the density of the matter shown in that position in the scan.   
Lung Segmentation: Lung segmentation is a process to identify boundaries of lungs in a CT scan image. Lung Tissue, Blood in Heart, Muscles and other lean tissues are removed by thresholding the pixels, setting a particular color for air background and using dilation and erosion operations for better separation and clarity. (using U-Net Network for Image segmentation here)

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
Multi-path CNN architecture design is a CNN Network having multi-path convolutional layers, the path considering smaller, medium, and larger receptive field sizes. these paths first, second, and third path, respectively.  
That the receptive field size of the first path is 3×3, the second path is 5×5, and the third paths is 7×7. 
unlike the traditional CNN, the mp-CNN has three pathways.These paths are designed to better approximate local and global dependencies of the neighboring
pixels. The first and the second path more focused on the details (local dependency) and the third path focused on the contextual information (global dependency).
### Concatenation  
To average the effect of receptive field size, concatenatenation of the output of the last convolution layer of each path is the next step, and this concatenation followed by a soft-max function to predict the input.
#### Training  
use stochastic gradient method by repeatedly choosing labels at a random subset of patches within each lung, and calculating the mean negative log-probabilities
for mini batch of patches and doing a gradient descent step on the CNNs parameters is the summary of this step.
#### Evaluation metrics
To measure how the model well perform, we compute the commonly used image detection performance measures accuracy, specificity, recall.
#### Retrain the Model

