# Soccer Injury Risk Prediction

## Table of Contents

- [Introduction](#introduction)
- [Data Collection](#data-collection)
- [Model Engineering](#model-engineering)
- [Model Building](#model-building)
- [Results](#results)
- [Next Steps](#next-steps)

## Introduction

Brazil, favourites to win the 2014 FIFA World Cup, saw their prospects diminish following the injuries of star player Neymar. With their subsequent 7-1 loss to Germany, Brazil became a key example of how injuries can alter the outcome of a match and dictate team success. This investigation will attempt to address the complex nature of soccer injuries and predict whether players are at risk of injury using various binary classifications models. We also hope to also discover underlying factors that may or may not contribute to soccer injuries. 

![alt text](https://www.denverpost.com/wp-content/uploads/2018/07/c4889a81c7884e7f9296abe22b6c7d83.jpg?w=640)

### Methods Used

- Web Scraping
- Data Visualization
- Inferential Statistics
- Predictive Modelling
- Machine Learning

### Technologies

- Python
- BeautifulSoup, pandas, numpy, scikit-learn, matplotlib
- Google Colaboratory

## Data Collection

To build our injury predictor model, we collected a comprehensive dataset that includes biometric data (such as age, height, weight, etc.) and in-game statistics (such as minutes played, distance covered, tackles made, etc.) of soccer players from Europe's top leagues, with the target feature being the current injury status of a player (injured or non-injured). The data was obtained from reliable sources, including [Transfermarkt](https://www.transfermarkt.com/) and [FBREF](https://fbref.com/en/).

Before proceeding, we performed thorough cleaning on the collected dataset. This step involved unpacking lists, handling missing values, removing outliers, and resolving any inconsistencies or errors present in the data.

## Model Engineering

In this phase, we conducted exploratory data analysis (EDA) to identify patterns, correlations, and potential factors contributing to injuries. Feature engineering techniques were then applied to derive new features and transform existing variables for improved predictive power. Additionally, we employed feature selection and scaling in hopes of improving model performance by reducing overall noise in the dataset.

## Model Building

Using the processed dataset, we attempted to use various machine learning models to predict the likelihood of player injuries based on the available biometric and in-game data. We experimented with different algorithms such as logistic regression, random forest, and artificial neural networks, and evaluated their performance using appropriate metrics. The model with the highest predictive accuracy was selected and fine-tuned through hyperparameter optimization to ensure optimal results.

## Results

Part of the intention of this model was also to help clubs make informative decisions when it comes to limiting injuries. However, the obtained and engineered features did not show much correlation at all with the target. Although this limits the ability to show what features contribute to player injuries, it provided insight into features that do not need to be monitored when trying to monitor injurty risk.

Despite poor performance across the board due to low correlation coefficients, our best model, LR-L2, yields a recall of 0.851 and AUC of 0.726 which is satisfactory when considering the goal of our investigation.

## Next Steps

The intention of this study was to gain some insight into the ambiguity surrounding soccer injuries and build predictive models capable of identifying whether a player is at risk of injury. Going forward, the investigation should also include data from a range of global soccer leagues. In addition to the small sample size, the dataset also contained many features with very low correlation to the target feature. More relevant features such as training regime, diet and detailed medical history may be better suited to accurately predict a player’s risk of injury. 

In the end, we were able to somewhat address the initial problem statement by obtaining a satisfactory LR-L2 model with a recall of 0.851 and gained insight into factors which do not contribute to injuries.


