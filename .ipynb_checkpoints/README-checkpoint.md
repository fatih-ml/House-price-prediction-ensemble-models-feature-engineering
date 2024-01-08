# Real Estate Price Prediction: Advance Regression Techniques, Ensemble Methods

## Objective and Framework  
As an experienced data scientist with a deep-rooted focus on machine learning and a robust background in the real estate sector, I found a unique intersection of my expertise and passion in the realm of house price prediction. Having dedicated my career to providing data-driven solutions, the Kaggle competition featuring the "House Prices: Advanced Regression Techniques" dataset presented an exciting opportunity to contribute to the data science community.

In the pursuit of enhancing predictive modeling, my goal with this notebook is to amalgamate domain knowledge from the real estate sector with the intricacies of machine learning. This amalgamation aims to provide a nuanced perspective, enabling not only accurate predictions but also valuable insights into the factors influencing house prices. Leveraging my experience, I set out to contribute to the field by incorporating various aspects of feature engineering, exploratory data analysis, complex blended models, and layered machine learning regression techniques.  

### Dataset    
The Kaggle competition revolves around a dataset containing 80 crucial property features, ranging from basic attributes like total area and bathroom count to more nuanced factors such as neighborhood and kitchen quality. These properties are situated in Ames, Iowa, USA.

With a training dataset comprising 1460 properties and a validation set of 1459, our regression task is to predict house prices based on these diverse features. In the dynamic real estate landscape, where every decision is monumental, accurate predictions become imperative. This dataset becomes our canvas, and through advanced regression techniques, we aim not only to forecast prices accurately but also to gain profound insights into the intricate factors influencing real estate values.  

### The Outline of This Study:  
1. Introduction & Summary
2. Initial data Exploration
3. Data Preprocessing & Feature Engineering
4. Exploratory Data Analysis
5. Model Building & Evaluation
6. Base Layer: Blended Model
7. Stacked Model: Add Meta Layer
8. Deployment

## Summary

### Data Preprocessing & Feature Engineering   
In meticulous exploration of the train and validation datasets, a decision was made to conduct separate preprocessing to prevent data leakage. The preprocessing phase included robust feature engineering, resulting in a professional feature selection process. After comprehensive exploratory data analysis, feature engineering, and preprocessing, the dataset was streamlined to 35 features from the original 80. Notably, this dimension reduction was achieved primarily through sophisticated feature engineering techniques rather than mathematical feature selection.

Addressing missing values involved a careful imputation process. Two new features, "Neighborhood Median House Price" and "Neighborhood Square Foot Property Price," were introduced to capture essential locality nuances. Year-related features were transformed into ages, and a systematic approach with custom functions was applied to features related to basement, terrace, garage, and bathrooms, where we successfully reduced these features into single or double features. Ordinal categorical variables were handled through custom ordinal mappings to retain information without inflating dimensionality via one-hot encoding.

### Exploratory Data Analysis  
The exploratory data analysis uncovered crucial insights into the dataset. Residential Low Density Areas housed the majority of properties, with 90% being Detached Single Family Houses. A significant portion of sales (90%) comprised conventional and normal transactions. The creation of two neighborhood-related features aimed at emphasizing the pivotal role of location in real estate pricing.

Correlation analyses highlighted that the Overall Quality of a house holds the strongest correlation coefficient (0.79) with Sale Price. The importance of features such as Ground Living Area, External Quality, Kitchen Quality, and others was underscored through high correlation coefficients. Outliers, detected using scatterplots and boxplots, led to a careful curation process, dropping four observations among 1460.

A nuanced approach to handling outliers in variables like LotFrontage, LotArea, and others was taken, considering the intricate dynamics of the real estate sector. Log transformation was applied to the target variable, SalePrice, to address skewness and align the distribution more closely with a normal distribution.

This detailed preprocessing and exploratory analysis set the stage for a robust foundation upon which advanced regression techniques and model building would be applied. The challenging yet crucial task of feature engineering was demonstrated as the linchpin of this study.

### Machine Learning Modelling Strategy:    
The model building process commenced with a systematic approach, involving the compilation of 10 diverse algorithms spanning linear models, tree-based models, and ensemble methods such as Gradient Boosting Regressor and LightGBM. Each algorithm was encapsulated within a pipeline, and their performance was thoroughly evaluated, considering metrics such as R2 score, MAE, MSE, RMSE, and cross-validated scores. The primary focus was on RMSE, given its significance for addressing the correctness of predictions, where as R2 score can sometimes be misleading.

<img src="data/model_diagram.png" alt="Model Diagram" width="700"/>

### Key findings & The Importance of Feature Engineering:  
One of the key outcomes of this notebook is to underline the importance of domain knowledge and feature engineering. The default models reached a 91.6 R2 score and 0.115 RMSE score without any hyperparameter tuning, model blending or stacking, or further feature selection methods. This achievement is attributed to feature engineering and exploratory data analysis.

Ridge, Lasso, and Gradient Boost Models performed best, whereas KNeighborsRegressor and DecisionTreeRegressor were the poorest. Specifically, Gradient Boosting Regressor and Light GBM Regressor were top-performing among tree-based models, whereas XGBoost which is known as the Kaggle champion was poor in our study.

### Base Layer: Building Blended Model  
Models exhibiting superior performance were identified based on their RMSE and diversity. A swift hyperparameter tuning using RandomizedSearchCV was employed to optimize these promising models. Eventually Ridge Regressor, Support Vector Regressor, Light GBM Regressor and Gradient Boosting Regressor were choosed for
blending phase which introduced a novel approach to model fusion, creating a base layer with a blended model. A variety of algorithms were experimented with, and different weightings were explored to optimize performance by the help of VotingRegressor. While best single model had 0.91 R2 Score and 0.1162 RMSE, the blending process contributed notably and boosted the performance to 0.92 R2 Score and 0.1121 RMSE score.

### Stacked Model: Add Meta Layer  
Taking a step beyond blending, the stacking phase added a meta layer to the model architecture. The base layer predictions were integrated into the dataset as new features, creating an enriched dataset for the meta model. Rigorous evaluation of potential meta models, including Gradient Boosting Regressor, ensued. The performance boost achieved through stacking validated the effectiveness of this advanced ensemble modeling technique. 

The comprehensive evaluation strategy involved continuous adjustment of models, resulting in a remarkable achievement of a 0.95 R2 score and a 0.088 RMSE score—a substantial improvement from the initial stages, also securing a 4% standing on the Kaggle scoreboard.

### Model Deployment  
The culmination of the model-building journey involved deploying the stacked model with the entire dataset. The stacked model, with its intricate architecture, underwent a final training iteration, leveraging the entirety of the data for enhanced predictive capabilities. This deployment phase marked the conclusion of a comprehensive guide for data preprocessing, exploratory data analysis, regression techniques, and advanced ensemble modeling. Despite not explicitly optimizing for Kaggle scores, the model secured a top 4% position in the Kaggle Leaderboard, highlighting its effectiveness in predicting house prices accurately.


## Conclusion

### Final Reflections, Challenges, and Opportunities for Enhancement    

This notebook prioritized swift runtime, minimizing detailed model tuning to ensure accessibility, aiming for a five-minute runtime on a low-end computer including also entire hyperparameter tuning. While the current model achieved impressive scores through the incorporation of domain knowledge, robust data analysis, and advanced machine learning techniques, potential enhancements lie in fine-tuning aspects like houses' remodeling age, OverallCondition, possible inflation rates, and Month Sold. Addressing outliers in features such as LotArea and MasVnrArea offers additional opportunities for model improvement. The central challenge faced was efficient dimension reduction amid the wealth of available features.

# Contact  
For questions, collaborations, or further discussions, feel free to reach out on [Linkedin](https://www.linkedin.com/in/fatih-calik-469961237/), [Github](https://github.com/fatih-ml) or [Kaggle](https://www.kaggle.com/fatihkgg) and reach original dataset on [this link](https://www.kaggle.com/c/titanic)

# Libraries Used (Mainly)
Python
Pandas, Numpy for data manipulation
Matplotlib and Seaborn for data visualization
Scikit-learn for model building and evaluation

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Acknowledgements

I'd like to express my gratitude to KAGGLE to create this dataset and learning possibility to all data science enthusiasts.


