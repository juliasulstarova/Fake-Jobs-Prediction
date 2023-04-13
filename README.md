# Fake-Jobs-Prediction
This project analyzes a highly imbalanced dataset of 10K jobs descriptions to create a prediction model that can identify fraudulent jobs postings. Three main ML models are used for this project: Logistic Regression, Random Forest and Neural Network.
<br>
As most of the features consist of text, this project explores cleaning textual data, stemming, BoW, TF-IDF, text augmentation as well as under-sampling techniques and class weightage adjustment to handle the significant class imbalance. 
<br>
## 1. Dataset 
The original dataset for this project consists of circa 18K job descriptions. The dataset has 18 features:
<br>
### a.	Textual features

![Screenshot 2023-04-13 155410](https://user-images.githubusercontent.com/87027062/231785818-a4f03c39-e312-4479-bbf2-dc2185a68ed5.png)

### b.	Numerical features

![Screenshot 2023-04-13 155500](https://user-images.githubusercontent.com/87027062/231785936-f04c95bb-b87b-4f07-a484-0c9fb612607d.png)

## 2. EDA

### A. Data Imbalance
<br> 
The original dataset has 17880 job descriptions, of which 17014 are descriptions of real jobs whereas only 866 account for fraudulent jobs.

![image](https://user-images.githubusercontent.com/87027062/231786410-971d4a53-321e-45c9-baec-90a64724d888.png)

### B. Null Values
![image](https://user-images.githubusercontent.com/87027062/231786578-c1a6773f-d5c6-42d1-96f8-d2416250a313.png)

<br>
As it can seen from the figure above, the features: department and salary range have a significant percentage of missing values, 65% and 84% respectively.<b> For this reason, I decided to drop these two features from the dataset.</b>
<br>
As fraudulent jobs are only 5% of the whole dataset, let’s take a look at how the percentage of null values changes when considering only fake jobs.

![Screenshot 2023-04-13 160723](https://user-images.githubusercontent.com/87027062/231787176-5204d9fb-6051-4636-8717-37bfc2f43165.png)

<br>
The only significant difference is noticed in the company_profile feature, where fake jobs tend to not include this feature in 68% of the cases, whereas, for real jobs there is only 16% missing values. <b>Thus, there is a strong relation between not including this feature and the job being fraudulent.</b>

### C. Location of job postings

![image](https://user-images.githubusercontent.com/87027062/231787571-f5008f25-2e5d-4802-9100-edbda0168357.png)

<br>
<b> Because 85% of the fake jobs in the dataset are located in the US, and to ensure that all the textual categories are only in English, I will only use jobs posted in the US for further analysis and predictions. </b>

### D. Visualizing jobs per state

![Screenshot 2023-04-06 143728](https://user-images.githubusercontent.com/87027062/231787880-a724e473-1f0f-43ae-b591-87fa8992df21.png)

<br>
As it can seen by the figures the highest number of job postings are in California and New York, whereas <b> the biggest number of fraudulent job postings are in Texas and California. </b> I take a further look at the ratio of fake jobs posted for each state:

![image](https://user-images.githubusercontent.com/87027062/231787987-9f33cbb4-0657-4766-83c2-c269bcdbb784.png)

<br>
Only the 25 states with the highest number of job postings are analyzed to prevent focusing on states with a high ratio of fake jobs because of an overall low number of job postings.
<br>
<b>Interestingly, Maryland accounts for the biggest percentage of fraudulent jobs, more than 30% of the 72 total job postings in this state. In Texas, fake job postings account for more than 15% of all jobs included in this dataset. </b>

## 3. Data Preprocessing

    A.	Drop the salary range and department columns from the dataset. 
    B.	Fill in the null values with blank spaces.
    C.	Create a text column that includes all the textual categories to fit into the prediction model.
    D.	Clean the textual data:
      1.	Convert to lower case.
      2.	Remove punctuation, numbers, links (https), symbols etc. 
      3.	Remove stop words like: a, an, and, I etc.
      4.	Use stemming, which groups words by their root stem. 
          For instance, words like: working, worked, works etc. all have the same root work. 
    E.	Convert the text data into numerical form to feed the predicting model. 
        I used the Bag of Words approach, which gives each word a score based on its occurrence in the text.
    F.	TF-IDF. 
        As BoW does not take into consideration how frequent the word is in all the text, I used TF-IDF that considers all the texts, to assign a weightage to the word.
    G.	Split the data, 70% training and 30% testing data.
    

## 4. Predictions
### A. Three models were used for this project
1. <b> Logistic Regression </b> <br>
2. <b> Random Forest Classifier </b> <br>
3. <b> Neural Network </b>

### B. Evaluation Metrics
Considering that the dataset has a high data imbalance problem, with fraudulent cases only accounting for 6.8% of all the job posting, the accuracy metrics would not be a good evaluation metrics for this project. This is because the models could predict all the job postings as real and still have an accuracy above 0.9. 
<br>
<b> For this reason, special attention is given to the F1-Score for fraudulent job, which combines the Precision and Recall metrics. </b>

### C. Class Imbalance
Three techniques were used to handle the class imbalance problem:
#### 1. Random Under-sampling
When applying Random Under-sampling to the Random Forest Classifier, I observed a significant increase in the recall score (from 0.60 to 0.92) for the fraudulent cases, however, the precision is lowered to 0.38, the lowest value I have seen so far for this prediction problem. This can be explained as the model is predicting a lot more cases as fake jobs, even cases that are in fact not fake. <b> This is due to the under-sampling method losing valuable information from the majority class to balance the dataset. </b>
<br>
<b> For this reason, I chose to rather than balance the number of real and fake jobs, to under-sample the majority class to a 1/2 ratio of majority/minority class. </b> 
This way I observed an improved f1-score better than the Logistic Regression and Random Forest previously used on the original imbalanced data. This method performs significantly better as the models are trained on a less imbalanced dataset, however, less information from the majority class is lost in the process. 
#### 2. Class Weight Adjustment
Using the Class Weight balancing option when training the models, I noticed an important increase in the F1-score for the Logistic Regression model, from 0.57 to 0.79. However, in all cases the Random Under-sampling technique provides better results for the models used in this project. 
#### 4. Data Augmentation 

### D. Text Data Augmentaion
A great focus in this project was given to research text augmentation methods to provide new data entries to train the prediction models. Considering that there is relatively less research conducted in data augmentation for textual features, this process involved a great deal of trial and error. 
<br>
An important challenge faced in this project was deciding the right augmentation techniques to use to augment the fraudulent jobs postings. It was important to decide on an augmentation method that could create new data entities different enough from the original ones to risk overfitting the model but without changing the context too much so that the description would not account for fake jobs anymore.
<br>
<b>From the three main types of augmentation offered by NLPAug: character, word and sentence level augmentation, I decided to opt for word level augmentation. I opted out of using character level augmentation considering that before being trained the training data will go through the text cleaning process and stemming that would most likely get rid of the new added characters and result in duplicate data entities that would most likely overfit the models. </b>
<br>
After an extensive trial and error process, I decided to go for a mix of these two word-level argumentation methods:

        1.	Synonym replacement, to obtain sentences with same meaning but different words.
        2.	Contextual word embedding, to insert new words to the sentences based on similar context. 

## 5.Results
![image](https://user-images.githubusercontent.com/87027062/231792930-21be72d3-1e31-450f-81d8-d4ef31545618.png)
![image](https://user-images.githubusercontent.com/87027062/231793112-52c3e082-0b08-4ae6-9125-9e50bb6f4393.png)


