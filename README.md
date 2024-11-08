# CO<sub>2</sub> Storage Capacity Prediction for Saudi Arabia 🌍💨

## Project Overview
In this project, I dove into predicting CO<sub>2</sub> storage capacity for different geological sites across Saudi Arabia! The idea was to see which places would make the best storage spots in a fictional scenario. This project was all about learning, with a big focus on sustainability 🌱 and reducing carbon emissions 💨 (even if it was just for practice!).


> **Disclaimer**: The data used here is **completely made up** and was created just for fun practice. None of this represents actual CCS sites.

## What I Did 🛠️
With a dataset full of imaginary geological factors like temperature, pressure, and CO<sub>2</sub> density, I built models to predict storage capacity. I tried out **Linear Regression** and **CART (Classification and Regression Trees)** models to see what kind of insights I could pull out of the data. 

### Steps Involved:
- **Data Cleaning** 🧹: Handling missing values and prepping the dataset.
- **Exploratory Data Analysis** 🔍: Looking at patterns, correlations, and key features in the data.
- **Modeling** 🤖: Trained a linear regression model and a decision tree (CART) to predict CO<sub>2</sub> storage capacity.

## What I Learned 💡
- **Importance of Features**: Thanks to CART, I could see which factors (like temperature and pressure) are really pulling their weight when it comes to CO<sub>2</sub> storage potential.
- **When Linear Isn’t Enough**: The linear model gave a good baseline, but CART shone when it came to more complex interactions. Now I see why it’s handy to have a mix of simple and complex models in the toolkit!
- **How to Prune a Tree 🌳**: This was big! In a previous project (predicting rainfall), I skipped pruning, and my model took forever to run. Here, I finally learned to prune the CART model to avoid overfitting and keep things snappy.

## Limitations ⚠️
- **Overfitting with CART**: Without proper pruning, the **CART model** may fit the training data too closely and lose its ability to generalise well to new, unseen data.
- **Data Complexity**: The **Linear Regression** model performed well in simpler scenarios but didn't capture complex, non-linear interactions as effectively as **CART**.

## Skills and Tools 🛠️
- **Programming**: R
- **Libraries**: `data.table`, `dplyr`, `rpart`, `ggplot2`
- **Machine Learning**: Linear Regression, CART

---
This project gave me hands-on experience with predictive modeling, model comparison, and tree-pruning magic! Though the data is fictional, I got to play around with real-life techniques and build my skills in managing model complexity and spotting key features. Plus, I got a fun reminder of how machine learning can support sustainability efforts in real-world contexts 🌍💡.
