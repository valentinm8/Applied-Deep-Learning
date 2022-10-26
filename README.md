# Applied Deep Learning - Computer Vision - Image Classification

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
