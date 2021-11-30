# Predictive Modelling - User Adoption

### Purpose
Defining a "adopted  user" as a user who has logged into the product on three separate days in at least one seven day period, identify which factors predict future user adoption.

### Data Wrangling & Feature Engineering

I first created the adopted_user feature by resampling the logins table to each day and grouping by user_id. I then used a rolling 7 day count to set the adopted_user column to a 1 if the count was 3 or more, and a 0 if less than 3. This is the main feature used in predicting user adoption. I then merged the login table and the user information table using the user_id to join them.

I created 2 features from the user information table. The first is "invited_by_user", is set it to 1 if the "invited_by_user_id" is greater than 0, otherwise it is a 0. Then I created a feature based on whether a user is part of an organization using the 'org_id' column, if it not empty set it to 1, otherwise 0.

### Exploratory Data Analysis
The graphs below show the count of adopted users based on the features that will be used in predicting user adoption.

<img src="https://user-images.githubusercontent.com/41071502/133360773-d421f9ee-dae2-4fec-abb1-1ccce424f498.png" alt="alt text" width="750">
<img src="https://user-images.githubusercontent.com/41071502/133360803-45f56dc9-0220-4ccc-be38-416e3da55279.png" alt="alt text" width="750">
<img src="https://user-images.githubusercontent.com/41071502/133360815-fa7d3e29-3572-4ef0-9664-05aef3971bf2.png" alt="alt text" width="750">
<img src="https://user-images.githubusercontent.com/41071502/133360821-3d84f4b2-0bc2-48ce-bc13-d9ff63a868b8.png" alt="alt text" width="750">
<img src="https://user-images.githubusercontent.com/41071502/133360835-df06e62d-3e58-4ba3-b0e5-c6d16511f4d0.png" alt="alt text" width="750">
<img src="https://user-images.githubusercontent.com/41071502/133360842-620c9dc4-6089-41c3-aec8-9ac553a96449.png" alt="alt text" width="750">


The first image is the heatmap of correlation between features in the data set after removing features that are redundant or not useful. The heatmap below it is after I used pandas get_dummies method to encode the data.
![image](https://user-images.githubusercontent.com/41071502/133361122-18afb92b-a1c7-47e7-80f4-57bb59e63902.png)
![image](https://user-images.githubusercontent.com/41071502/133361131-c77cd7f6-ba2a-4eda-a759-329972a35dff.png)


### Modeling
I created a machine learning pipeline using sklearn to assemble a random forest classifier alrgorithm with a standard scaler. I split the data into 90% training 10% test data. I would like to play around with the training-test split and some other parameters in the future to see if any quick improvements can be made.

### Results

The accuracy score of the model is approximately 91%. The area under the curve is 51%, this does not necessarily mean the model has not performed well, the link below provides a detailed explanation as to why an ROC curve could look like the graph below. 

https://www.r-bloggers.com/2019/03/what-it-the-interpretation-of-the-diagonal-for-a-roc-curve/

![image](https://user-images.githubusercontent.com/41071502/133364648-a977936b-45da-4630-83be-f00ff0808d73.png)

The graph below shows the importance of each feature in. The higher the value the more important the feature.  
![image](https://user-images.githubusercontent.com/41071502/133364600-fc26e7d6-a4e6-439a-ac34-b746441205f9.png)


### Future Improvements
I added a bonus modeling section in my notebook, I used the same data and features but applied 5 different ensemble methods. Credit goes out to Tim Head (link below), using these methods produced accuracy scores close to 94% and AUC of 97%. In terms of future improvements this is a very good example already. However, the main improvement I would like to work on in more feature engineering, I want to see if there are any other features I did not think of that would increase the accuracy of the model.

https://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html






