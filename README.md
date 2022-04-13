# APS360 Project Final Report
|                                              |                                          |
|:---------------------------: | --------------------------------------------------------------|                         
|Submission Date: | April 13, 2022 |
|Group: | 50 |   
|Members: | Dasha Moskvitina, Aric Leather, Hao Xiang Yang, James Du|
|Word Count | 1988 |
|Link to Github | https://github.com/dasha-moskv/APS360-Project |

## 1.0 Introduction
The world has become highly interconnected and is full of diverse cultures and languages. As globalization accelerates, a need for fast and high-quality language translation has emerged to facilitate communication between people. An essential component of language translation is language detection, so that speech can be appropriately parsed and translated. Our objective for this project is to develop a machine learning model which can successfully identify the top ten most commonly spoken languages in the world. The identification of languages requires knowledge and experience with their sound and structure, which are qualities that machine learning models are capable of discerning. This proposal outlines the architecture for our model, identifies ethical considerations and risks, and details our project plan.


## 2.0 Illustrations and Figures
<p align="center">
<img src=/Research/Screenshot_1.png>	
</p>
<p align="center">
Figure 2.1 Overall project model
</p>

## 3.0 Background and Related Work 
In 2018, researchers at The Johns Hopkins University developed a system of Deep Neural Networks (DNN) which outperformed previous machine learning models trained for spoken language recognition (SLR). It achieved very high performance while training on a dataset consisting of 57 different languages. A strategy used to diversify training data, which results in a better performing model, is the augmentation of samples in the dataset. Samples would, at random, have background noise added, have their playback speed changed slightly, or even have instrumental music added in the background [1].

## 4.0 Data Processing
When training a machine learning model, being able to acquire a large dataset with high quality, appropriate training data is crucial. In this section we will detail how we constructed our dataset and the preprocessing we performed to get the best possible model.

### 4.1 Data Collection and Splitting
We sourced our data from Mozilla CommonVoice [2]. We selected the ten most commonly spoken languages in the world specified in [3] with the exception of Bengali (as it was unavailable in Mozilla CommonVoice), which we replaced with Japanese, and downloaded approximately 9000 random speech samples for each. Then for each language, 80% of the samples became part of the training dataset, 10% part of the validation dataset, and the remaining 10% part of the testing dataset.

### 4.2 Data Preprocessing 
We developed a Python script which automatically generated the three datasets. Using an existing technique, we downsampled the audio files to 8KHz and resized the clips to 10 seconds, then generated a spectrogram representing the Mel Frequency Cepstral Coefficients (MFCC) of the audio clip  [4]. The MFCC are a set of features in the frequency spectrum which better characterize the human voice [5]. This resulted in a conversion from audio sample to image file representing the spectrogram.

<p align="center">
<img src=/Research/Screenshot_2.png>	
</p>
<p align="center">
Figure 4.2.1 Mel-scale spectrogram extraction from audio file [6]
</p>

## 5.0 Architecture
The final model is a combination of a convolutional neural network (CNN) and a fully-connected (FC) artificial neural network (ANN) similar to AlexNet. Since raw speech input is “non-stationary”, i.e. the duration and amplitude of each speech input differs from one to another, in nature, the best way to process data is to extract useful features from the spectrograms using CNNs [7]. The first CNN layers were used to extract short-term time window patterns while the later CNN layers further processed those outputs into high-level patterns. The outputs of the CNN were used as inputs to an ANN with fully connected layers for classification (see Figure 5.1). 

<p align="center">
<img src=/Research/Screenshot_3.png>	
</p>
<p align="center">
Figure 5.1: Proposed architecture schematic
</p>

**Input:** Spectrogram (.png) of a 10 second 8KHz audio file (.mp3).  

**Prior Data Processing:** Separate the relevant features of the waveform using the Python librosa library described in the previous section [5]. 

**CNNs:** Extract defining features from the spectrogram.

**Fully-connected ANN:** Use the extracted features to classify the audio file in a predetermined number of categories.

**Output:** A 1-D tensor with each index corresponding to a language, and each entry corresponding to a likelihood that the audio file contained speech in that language.


## 6.0 Baseline Model
Based on our research regarding the architecture in section 5.0, our baseline model is a simple LeNet5 CNN with a limited number of layers, which is a series of 2 CNNs with an FC ANN at the end for classification. We believe that this is a reasonable baseline model as our final model is an AlexNet CNN, which is more complex. The results of training our baseline model are shown in Figure 6.1. We achieved peak values of 81% and 78% accuracy on the validation and test set respectively. 

<p align="center">
<img src=/Research/Screenshot_4.png>	
</p>
<p align="center">
Figure 6.1: Error and loss graphs for baseline model
</p>

## 7.0 Quantitative Results
We present the validation and test accuracies of our baseline and final models as measures of its performance on the dataset. As described in Section 6.0, our baseline model achieved an 81% accuracy on the validation set and a 78% accuracy on the test set. However, investigating the error in Figure 6.1 reveals that the model began to overfit quickly, with the error becoming close to 0 at epoch 6. Figure 7.1 presents our accuracies for our final model based on the AlexNet architecture. We were able to achieve a validation accuracy of 89% with this architecture, and a test set accuracy of 87%. This indicates that our final model is performing as expected and better than the baseline model. 

<p align="center">
<img src=/Research/Screenshot_5.png>	
</p>
<p align="center">
Figure 7.1: Error and loss graphs for final model
</p>

## 8.0 Qualitative Results
The summary of our final model’s predictions on the test set is represented in the confusion matrix below. Each row in the matrix corresponds with one of the 10 languages chosen for analysis, indexed alphabetically. Each column represents the number of predictions for the corresponding language. The number of correct predictions is represented by the diagonal in the matrix. Since our model is classifying multiple languages, it is important to understand which languages are more likely to be falsely classified in terms of false positives and false negatives. Using this method, we can quickly identify which languages confuse the model and which languages the model can immediately recognise.

<p align="center">
<img src=/Research/Screenshot_6.png>	
</p>
<p align="center">
Figure 8.1: Confusion matrix for test set
</p>

By observing the rows in Figure 7, we can get a sense of where the model is strongest and where it most often makes mistakes. For example, the model seems to be good at classifying Indonesian, with only 61 misclassifications out of 854 (7%). When trying to classify a Hindi sound clip, a surprising 101 out of 853 clips (12%) were misclassified as English.

### 9.0 Evaluate Model on New Data
Before evaluating our model on new data, we made some considerations regarding our dataset to ensure our results were fair, given the dataset’s traits. The dataset consisted of primarily male voices according to the statistics available on the Mozilla CommonVoice download page [2]. Additionally, the clips in the dataset generally appear to be recorded on lower quality microphones and are spoken monotonically, as if one is reading off a script.

With these considerations in mind, we produced and gathered a small set of audio clips to process into spectrograms in order to evaluate our model. In total, we tested 10 new clips which were not a part of our dataset at all when training the model.


	
| Language Spoken | Source | Model Prediction |
| --------------- | --------------- | --------------- |
| English | Aric (Group) | English |
| Mandarin | James (Group) | Mandarin |
| French | Hao (Group) | English (incorrect) |
| Arabic | Tatoeba [9] | Arabic |
| French | Tatoeba [9] | French |
| Mandarin | Tatoeba [9] | Mandarin |
| English | Tatoeba [9] | English |
| Japanese | Tatoeba [9] | Japanese |
| Indonesian | Tatoeba [9] | Hindi (incorrect) |
| Spanish | Tatoeba [9] | Spanish |

<p align="center">
Table 9.1: Evaluations on new data
</p>



Our model correctly predicted the language being spoken in 8 of the 10 new clips we tried. This is very close to the test set accuracy we measured of 87%, indicating that the model is performing very well even on completely new data.

## 10.0 Discussion 
Based on our observed test set accuracy and the results we gathered evaluating the model on new data, we believe that our model is performing very well. Our model has a high overall accuracy and high accuracy within individual languages. However, there were some unexpected results from the model that we believe we could mitigate with some minor adjustments. At the outset of our project, we predicted that our model might not perform optimally between languages that were phonetically or linguistically similar, such as Spanish and Portuguese. While the model did have difficulty distinguishing between some languages, they were not languages that we had originally expected. We believe that this is due to a lack of diversity in the dataset. While our dataset consisted of approximately 9000 voice clips per language, the unique voices within the dataset for each language numbered between 200-300. In addition, upon closer inspection, we found that the voices in the dataset consisted primarily of lower-pitched voices with a monotonous speaking style.

For such a large net like AlexNet, it is possible our model began learning features of the voices as well as language characteristics, and when languages appeared to be ambiguous, classified voice clips based on speech characteristics instead. As a result, our model would also have had difficulty classifying voices within a range it had not previously encountered, such as higher-pitched voices. This would explain our model’s unexpected misclassifications, but overall high accuracy.

## 11.0 Ethical Considerations
We considered consent and data privacy throughout the project. We used speech samples that are in the open domain for our training, but had no feasible way to verify if each sample was consensually obtained. Should an individual want their speech samples removed, we would retrain our model without the samples. Upon deploying our model, we will ensure that each user consents to being recorded and that their personal data is not attached to their sample. 

Another important ethical consideration is a source of representation bias. In the Mozilla CommonVoice dataset, some languages, such as Japanese, had a roughly equal distribution of male and female voices. However, most languages had a skewed distribution towards male voices. To achieve better results moving forward, we should also ensure that different accents and speech patterns are equally represented in the dataset.

### 12.0 Project Difficulty/Quality 
Our group faced many obstacles while collecting, building, and training the model. According to [10], a vast amount of data is necessary to achieve a decent accuracy on the classification of languages. Our data consists of about 9000 clips for each language, which amounted to approximately 90000 clips total. This alone amounted to just over 3 GBs of data. Then since we had to transform these sound files into .png spectrograms, another 4.4 GBs of data were added to the dataset. The size of the data collection made it hard to push and pull from Github due to technological constraints such as slow internet or insufficient local storage space. 

Another obstacle we faced was the training time for hyperparameter tuning. Since we had a large amount of data and a large neural network with multiple layers (refer to Section 5.0), training took 2 hours to complete each time. One way we circumvented this problem was by saving the model at each epoch and stopping the training at the first sign of overfitting. However, due to the size of our neural network, the saved model storage space became a tradeoff. 

For our final model, we could have imported a pretrained AlexNet to classify the spectrograms. However, we discovered that the model would overfit very quickly at a very low accuracy (20%~30%). One reason for this result is that a pretrained AlexNet, trained with the dataset ImageNet, could only classify images of animals/objects in everyday life and does poorly on abstract images like spectrograms. Hence, we trained our own model following the architecture of AlexNet, making our project meaningful. Given the complexity and length of the project, we are satisfied with the quality of our final model. The complete project can be found at the Github link present in the cover page of this document. 

## 13.0 References
|                                              |                                          |
|:---------------------------: | --------------------------------------------------------------|                         
| [1] | D. Snyder, D. Garcia-Romero, A. McCree, G. Sell, D. Povey, and S. Khudanpur, “Spoken language recognition using X-vectors,” The Speaker and Language Recognition Workshop (Odyssey 2018), 2018. [Accessed: Feb. 9, 2022] |
| [2] | “Mozilla Common Voice,” Common Voice. [Online]. Available: https://commonvoice.mozilla.org/. [Accessed: Feb. 9, 2022]. |   
| [3] | “The most spoken languages in the world,” Berlitz, May. 31, 2021. [Online]. Available: https://www.berlitz.com/en-uy/blog/most-spoken-languages-world. [Accessed: Feb. 9, 2022]. |
| [4] | M. Fabien, “Sound feature extraction,” Dec. 7, 2019. [Online]. Available: https://maelfabien.github.io/machinelearning/Speech9/#7-mel-frequency-cepstral-differential-coefficients. [Accessed: Feb. 9, 2022]. |
| [5] | N. Singh Chauhan, “Audio data analysis using Deep Learning with python (part 1),” KDnuggets, Feb. 19, 2020. [Online]. Available: https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html#:~:text=Networks(ANN).-,Audio%20file%20overview,taken%2044%2C100%20times%20per%20second). [Accessed: Feb. 9, 2022]. |
| [6] | J. Rieber, “Spoken language recognition using convolutional neural networks,” Towards AI, Dec. 16, 2020. [Online]. Available: https://towardsai.net/p/deep-learning/spoken-language-recognition-using-convolutional-neural-networks-6aec5963eb18. [Accessed: Feb. 9, 2022]. |
| [7] | V. Passricha and R. Kumar Aggarwal, “Convolutional Neural Networks for Raw Speech recognition,” From Natural to Artificial Intelligence - Algorithms and Applications, 2018. [Accessed: Feb. 9, 2022] |   
| [8] | D. Lăpușneanu, “Spanish vs. Portuguese: How similar are they?,” Mondly. [Online]. Available: https://www.mondly.com/blog/2020/01/06/spanish-vs-portuguese-how-similar-are-they/. [Accessed: Feb. 9, 2022]. |
| [9] | Tatoeba. [Online]. Available: https://tatoeba.org/en/. [Accessed: Apr. 5, 2022]. |
| [10] | G. Singh, S. Sharma, V. Kumar, M. Kaur, M. Baz, and M. Masud, “Spoken language identification using Deep Learning,” Computational Intelligence and Neuroscience, vol. 2021, pp. 1–12, 2021. [Online]. Available: https://www.researchgate.net/publication/354375023_Spoken_Language_Identification_Using_Deep_Learning. [Accessed: Apr. 5, 2022]. |
