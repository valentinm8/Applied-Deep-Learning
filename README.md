# Applied Deep Learning - Computer Vision - Image Classification

# Milestone 1

## Kaggle Competition - PetFinder.my - Pawpularity Contest

Millions of stray animals suffer on the streets or are euthanized in shelters every day around the world. We may expect that pets with attractive photos generate more interest and are adopted faster. The most important question is what makes a good picture? Using data science methods may be able to accurately determine a pet photo's appeal and even suggest improvements to give these rescue animals a higher chance of loving homes. 

[PetFinder.my](https://petfinder.my/) is Malaysia’s leading animal welfare platform, featuring over 180,000 animals with 54,000 happily adopted. PetFinder collaborates closely with animal lovers, media, corporations, and global organizations to improve animal welfare.

Currently, PetFinder.my uses a basic [Cuteness Meter](https://petfinder.my/cutenessmeter) to rank pet photos. It analyzes picture composition and other factors compared to the performance of thousands of pet profiles. While this basic tool is helpful, it's still in an experimental stage and the algorithm could be improved

In this competition, idea is to analyze raw images and metadata to predict the "Pawpularity" of pet photos. The model will be trained and tested on the thousands of PerFinder's pet profiles. 


### How Pawpularity Score Is Derived

-   The  **Pawpularity Score**  is derived from each pet profile's page view statistics at the listing pages, using an algorithm that normalizes the traffic data across different pages, platforms (web & mobile), and various metrics.
-   Duplicate clicks, crawler bot accesses and sponsored profiles are excluded from the analysis.


### Project Idea & Approach

This project is focused on building or re-implementing neural network architecture that operates on an existing dataset that is already publicly available. Since the project idea was inspired by the already closed Kaggle competition, we will try to re-implement some of the most prominent shared notebooks neural networks, and even build our neural network to improve the state of the art. 

The task is to predict engagement with a pet's profile based on the photograph for that profile. We are also provided with hand-labelled metadata for each photo. The dataset for this competition therefore comprises both images and tabular data.

If successful, our solution would be adapted to the AI tools of the application. Using the application, shelters, and rescuers around the world may be able to improve the appeal of their pet profiles, automatically enhancing photo quality and recommending composition improvements. As a result, stray dogs and cats can find their "furever" homes much faster.


### Dataset Description


#### Training Data

-   **train/**  - Folder containing training set photos of the form  **{id}.jpg**, where  **{id}**  is a unique Pet Profile ID.
-   **train.csv**  - Metadata (described below) for each photo in the training set as well as the target, the photo's  **Pawpularity**  score. The  **Id**  column gives the photo's unique Pet Profile ID corresponding the photo's file name.

#### Example Test Data
-   **test/**  - Folder containing randomly generated images in a format similar to the training set photos. The actual test data comprises about 6800 pet photos similar to the training set photos.
-   **test.csv**  - Randomly generated metadata similar to the training set metadata.
-   **sample_submission.csv**  - A sample submission file in the correct format.


#### Photo Metadata

**Photo Metadata**, was created by manually labeling each photo for key visual quality and composition parameters.

These labels are  **not used**  for deriving our Pawpularity score, but it may be beneficial for better understanding the content and co-relating them to a photo's attractiveness. Our end goal is to deploy AI solutions that can generate intelligent recommendations (i.e. show a closer frontal pet face, add accessories, increase subject focus, etc) and automatic enhancements (i.e. brightness, contrast) on the photos, so we are hoping to have predictions that are more easily interpretable.

We may use these labels as you see fit, and optionally build an intermediate / supplementary model to predict the labels from the photos. If our supplementary model is good, we may integrate it into our AI tools as well.

In our production system, new photos that are dynamically scored will not contain any photo labels. If the Pawpularity prediction model requires photo label scores, we will use an intermediary model to derive such parameters, before feeding them to the final model.


The  **train.csv**  and  **test.csv**  files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of  **1**  (Yes) or  **0**  (No) for each of the following features:

-   **Focus**  - Pet stands out against uncluttered background, not too close / far.
-   **Eyes**  - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
-   **Face**  - Decently clear face, facing front or near-front.
-   **Near**  - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
-   **Action**  - Pet in the middle of an action (e.g., jumping).
-   **Accessory**  - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
-   **Group**  - More than 1 pet in the photo.
-   **Collage**  - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
-   **Human**  - Human in the photo.
-   **Occlusion**  - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
-   **Info**  - Custom-added text or labels (i.e. pet name, description).
-   **Blur**  - Noticeably out of focus or noisy, especially for the pet’s eyes and face. For Blur entries, “Eyes” column is always set to 0.

#### Additional Information

- **Files**: 9923 files
- **Size**: 1.04 GB
- **Type**: jpg, csv


### Work-breakdown structure

Here we can see the project overview, which includes three key steps: Re-Implementation & Build NN, Application Development, and Documentation. 
  
<p align="center">
  <img src="https://user-images.githubusercontent.com/96443138/197839045-2919911a-33ac-4aab-b214-e4b65dc06402.jpg">
</p>



Planning complex projects can be challenging, since many unexpected issues may be encountered along the way.

Here is a work-breakdown structure for the individual tasks with rough time estimates: 


- Re-Implementation **(80 hours)**
  - Data pre-processing (12 hours) 
  - Defining NN (14 hours)
  - Implementation (25 hours)
  - Training (5 hours)
  - Evaluation (2 hours)
  - Fine-tuning (20 hours)
  - Prediction (2 hours)


- Application Development **(37 hours)**
  - Analysis (10 hours)
  - Design (5 hours)
  - Implementation (19 hours)
  - Testing (3 hours)

- Documentation **(8 hours)**
  - Final Report (6h)
  - Presentation (2h)

In total: **125 hours**


### References

#### Scientific Papers


##### Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf

##### EfficientNetV2: Smaller Models and Faster Training
https://arxiv.org/pdf/2104.00298.pdf

##### A ConvNet for the 2020s
https://arxiv.org/pdf/2201.03545.pdf


#### Other

##### A Comprehensive Guide to Convolutional Neural Networks
https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53


##### Kaggle - 1st place solution
https://www.kaggle.com/competitions/petfinder-pawpularity-score/discussion/301686


##### Kaggle Competition - PetFinder.my - Pawpularity Contest
https://www.kaggle.com/competitions/petfinder-pawpularity-score/overview


# Milestone 2

## Introduction

In this part, the focus was to re-implement a method that reflects the state of art result. As it was mentioned in Milestone 1, the project idea was inspired by the already closed Kaggle competition, in which we re-implemented the winner's solution. The reason why we implemented the winner's solution is not only based on the best score, but the solution itself is very interesting. The solution consists of two parts, one tabular based on Transfer Learning and Support Vector Regression, and one image regression based on Deep Learning image architectures. 

*Comment:*
Additionally, I have completed the Coursera Deep Learning Specialization, which includes five courses. It has taken me a lot of time considering that I am doing over 30 ECTS this semester, but at the same time, it significantly helped me to understand the concepts in the Deep Learning field. 

## Error Metric and Target

The error metric which was used is already mentioned Pawpularity score. The Pawpularity score ranges from 1 to 100, and it is rounded to integers. The target follows a distribution with a mean average of 38.04 and a standard deviation of 20.59, which means if we predict 38.04 for all images, the RMSE score is 20.59. In other words, any model that scores **below 20.59** is considered to have some predictive power. 

The Kaggle competition solutions usually consist of three **RMSE** scores as follows:
- CV RMSE score (based on the training set)
- Public Leaderboard RMSE (based on the partial test set)
- Private Leaderboard RMSE (based on the full test set)

Since the public and private leaderboard use test set which is available only through limited daily submissions, CV score based on the training set was used for evaluation. 

We wanted to achieve the winner's solution's best CV RMSE score, which was **16.80935891258082**.

## First part - Transfer Learning + SVR

In the first part of the solution, the idea was to extract features from already pretrained architectures to transform the problem into a tabular approach, then fit a model using the extracted features. In the winning solution, the imagenet pretrained models available in timm and OpenAI CLIP libraries were used to extract features. Unfortunately, we were not able to implement the OpenAI CLIP library. The timm library contains 575 pretrained models, and most of them were trained using 1000 class imagenet images. 

OpenAI CLIP (not implemented) works differently. The architecture was trained using a combination of text and image, here is more information: https://github.com/openai/CLIP. 

Only by extracting the features from the individual architectures and running SVR directly on top was possible to score competitive RMSE scores. Here is the table:

| Architecture | Feature dimension | RMSE
|:--|:--:|:--:
| tf_efficientnet_l2_ns_475 | 1000 |17.56
|tf_efficientnet_l2_ns_475_512|1000| 17.57
|tf_efficientnet_l2_ns_475_hflip_384|1000|17.62
|ig_resnext101_32x48d|1000|17.63
|ig_resnext101_32x48d_hflip_384|1000|17.66
|tf_efficientnet_b6_ns|1000|17.66
|tf_efficientnet_b7_ns|1000|17.67
|deit_base_distilled_patch16_384|1000|17.68
|deit_base_distilled_patch16_384_hflip_384|1000|17.66
|tf_efficientnet_b8_ap|1000|17.70
|ig_resnext101_32x8d|1000|17.82
|vit_base_patch16_384|1000|17.82
|vit_large_patch16_384|1000|17.89
|resnest269e|1000|17.90
|swsl_resnext101_32x8d|1000|17.91
|vit_large_r50_s32_384|1000|17.92
|rexnet_200|1000|17.96
|resnetv2_152x4_bitm|1000|18.06
|repvgg_b0|1000|18.21
|fbnetc_100|1000|18.52

The diversity between models was used to boost the RMSE score. A simple forward feature selection with hill climbing was used to select the best models. We have used already prepared three subsets of pretrained features that scored well because we were limited with the GPU units. Here is the simple algorithm used for the model selection:

```
Features = [‘tf_efficientnet_l2_ns_475’] # Start using only 1 model
Best_feat = None
bestRMSE = np.inf
currentRMSE = 0
while currentRMSE < bestRMSE: # Keep adding models while rmse decreases
    bestRMSE = currentRMSE
    Features.append(best_feat)
    rmse_scores= [compute_SVR_RMSE(Features + [feat]) for feat in all_pretrained_models]
    currentRMSE = np.min(rmse_scores)
    best_feat = np.argmin(rmse_scores)
```

The algorithm keeps adding model features while RMSE keeps decreasing. The key was to use GPU-accelerated cuML SVR, otherwise, this step would take more than 3 months to execute. 

We have ended up with these three subsets of pretrained features:

- **SVR A** includes: ['deit_base_distilled_patch16_384', 'ig_resnext101_32x48d', 'repvgg_b0', 'resnetv2_152x4_bitm', 'swsl_resnext101_32x8d', 'tf_efficientnet_l2_ns_475', 'vit_base_patch16_384', 'vit_large_r50_s32_384']
- **SVR B** includes: ['fbnetc_100', 'ig_resnext101_32x8d', 'rexnet_200', 'resnest269e', 'tf_efficientnet_b6_ns', 'tf_efficientnet_b8_ap', 'tf_efficientnet_b7_ns', 'vit_large_patch16_384']
- **SVR C** includes: ['tf_efficientnet_l2_ns_hflip_384', 'deit_base_distilled_patch16_384_hflip_384', 'ig_resnext101_32x48d_hflip_384', 'tf_efficientnet_l2_ns_512', 'ig_resnext101_32x48d', 'vit_large_r50_s32_384']

RMSE scores for SVRs:
 - SVR A: **17.200158563547443**
 - SVR B: **17.297487754393913**
 - SVR C: **17.221901868223828**

Clipping target Pawpularity to 85 slightly improved the SVRs models RMSE.
Additionally, using the multiplier of 1.032 (1.032*ypredictions) boosted the RMSE score, probably because SVR uses MSE-based optimization. 



## Second Part - CNN + Vision Transformers Ensembling

In the second part of the solution, we have a weighted average of 5 classical image regressions models using different backbones and augmentations. Backbones used:

| Backbone | Image Size |TTA|Training Augmentations|Mixup+Cutmix|RMSE CV|Ensemble weight
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
|swin_large_patch4_window7_224|224|hflip|light|no|17.48|3
|beit_large_patch16_224|224|normal|light|no|17.42|4
|swin_large_patch4_window12_384_in22k|384|hflip|light|Yes(15%)|17.47|3
|beit_large_patch16_224|224|hflip|heavy|Yes(50%)|17.38|4
|tf_efficientnet_b6_ns|528|normal|heavy|Yes(60%)|17.75|2

Listed models were chosen using a feature selection based on the set of around 25 models which the winner built during the competition. Submission is limited to 9h, and the first (SVR) part takes around 4h to execute, which is why only a subset of the models could be used.
**BCE loss function** was used in all models. 
**TTA** (Test-time augmentation) gave only a **slight improvement**, around 0.0005.

Finally, we have weighted the models as follows:
**Ensemble weights**: [SVR_A`×`0.18395402 `+` SVR_B`×`0.08305837 `+` SVR_C`×`0.19805342 `+` ImagesEnsamble`×`0.55311892]

The best CV RMSE score we achieved was:
**RMSE: 16.872972151586403**.

We can see that it is very close to the target RMSE score. (**RMSE: 16.80935891258082**)



## Work-breakdown structure
### Initial 
NOTE: This is only work-breakdown structure for the second part

- Re-Implementation **(80 hours)**
  - Data pre-processing (12 hours) 
  - Defining NN (14 hours)
  - Implementation (25 hours)
  - Training (5 hours)
  - Evaluation (2 hours)
  - Fine-tuning (20 hours)
  - Prediction (2 hours)


### Realistic

- Re-Implementation **(100 hours)**
  - Data pre-processing (3 hours) 
  - Defining NN and understanding the models (24 hours)
  - Implementation (15 hours)
  - Training (6 hours)
  - Evaluation (2 hours)
  - Fine-tuning (13 hours)
  - Prediction (4 hours)
  - Issues and Problems (25)
  - Documentation (8 hours)


*Comment:*
NOTE: This is official work-breakdown structure, but in reality it took me less time to complete. 
- Deep Learning Specialization **(107 hours)** 
	- Convolutional Neural Networks (25 hours)
	- Sequence Models (25 hours)
	- Structuring Machine Learning Projects (7 hours)
	- Neural Networks and Deep Learning (25 hours)
	- Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization (25 hours)

Certificate:
https://coursera.org/share/eff732ac1a7a3895641c078fda481516

### Hyperparameter Tuning


In order to keep the report short, the results of hyperparameter tuning are included in the Experiments.txt file. It includes 11 different experiment results. Here are the best results:
~~~
Function fit_gpu_svr:
- SVR(C=30.0, kernel='rbf', degree=3, max_iter=4000, output_type='numpy')

Model beit_large_patch16_224:
- Dropout(0.05)
Model tf_efficientnet_b6_ns:
- Dropout(0.15)

Weighted average image models
a = 3
b = 4
c = 3
d = 4
e = 2
--------
Results:
Ensemble weights: [0.18395402 0.08305837 0.19805342 0.55311892]
Final RMSE: 16.872972151586403
~~~

Interestingly, these results scored better than the initial hyperparameters, and if we have manged to include the OpenAI CLIP library, we would probably outperformed the state-of-art. 

### Comment
- I have spent a lot of time finding the right solution to implement, and in the end, the 1st winner's solution was by far the most interesting. Especially this part with the embedding extraction, nobody expected that would perform this well. 
- Since I am using M1Pro chip (Mac), it was not possible to run this experiment locally. I have spent a lot of time trying to set it up, but some packages just did not work for me. Afterwards, I have chosen to use Google Colab where I had issues installing cuML, but I have fixed it. The Kaggle notebook was the easiest to set up at the end. 
- The disadvantage of using Kaggle notebooks is limited GPU usage (only 30 hours per week). That is the reason why my insights are slightly limited, and I could not experiment as much. 
- Unfortunately, the OpenAI CLIP library was not possible to implement because there was some issue with the build. After spending a lot of time, I have decided to skip it because it has almost no effect on the result. 
- Please find in the references the link to the OneDrive which includes all models used (~53GB), and the dataset (1.04GB).


## What's Next

The next step is to build an iOS app that evaluates how attractive your pet is based on the Pawpularity score. 


### References

##### Kaggle Competition - PetFinder.my - Pawpularity Contest
https://www.kaggle.com/competitions/petfinder-pawpularity-score/overview

##### OneDrive - Applied Deep Learning (Includes models and dataset)
https://tuwienacat-my.sharepoint.com/personal/e12122084_student_tuwien_ac_at/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fe12122084%5Fstudent%5Ftuwien%5Fac%5Fat%2FDocuments%2FApplied%20Deep%20Learning&ga=1

