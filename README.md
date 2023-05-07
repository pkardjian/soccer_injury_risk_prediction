# Soccer Injury Risk Prediction

## Table of Contents

- [Introduction](#introduction)
- [Data Collection](#data-collection)
- [Model Engineering](#model-engineering)
- [Model Building](#model-building)
- [Usage](#usage)

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

To build our injury predictor model, we collected a comprehensive dataset that includes biometric data (such as age, height, weight, etc.) and in-game statistics (such as minutes played, distance covered, tackles made, etc.) of soccer players from Europe's top leagues. The data was obtained from reliable sources, including [Transfermarkt](https://www.transfermarkt.com/) and [FBREF](https://fbref.com/en/).

Before proceeding, we performed thorough cleaning on the collected dataset. This step involved unpacking lists, handling missing values, removing outliers, and resolving any inconsistencies or errors present in the data.

## Model Engineering

In this phase, we conducted exploratory data analysis (EDA) to identify patterns, correlations, and potential factors contributing to injuries. Feature engineering techniques were then applied to derive new features and transform existing variables for improved predictive power. Additionally, we employed feature selection and scaling in hopes of improving model performance by reducing overall noise in the dataset.

## Model Building

Using the processed dataset, we attempted to use various machine learning models to predict the likelihood of player injuries based on the available biometric and in-game data. We experimented with different algorithms such as logistic regression, random forest, and artificial neural networks, and evaluated their performance using appropriate metrics. The model with the highest predictive accuracy was selected and fine-tuned through hyperparameter optimization to ensure optimal results.






