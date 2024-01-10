<img src="data/cover_house_ensemble_notebook.png" alt="Model Diagram" width="1280"/>


# Real Estate Price Prediction: Advance Regression Techniques, Ensemble Methods

## Objective and Framework  
Bringing together my extensive experience as a **data scientist** with a strong focus on machine learning and my **practical involvement in real estate price valuation and prediction systems**, the Kaggle competition on "House Prices: Advanced Regression Techniques" emerged as a captivating opportunity to make a meaningful contribution to the data science community.

The objective of this notebook is to elevate predictive modeling by synergizing real-world insights from the real estate domain with the intricacies of machine learning. In this pursuit, I aim not only to achieve accurate predictions but also to uncover valuable insights into the myriad factors shaping house prices. Capitalizing on my professional background, I embarked on this journey to contribute to the field by integrating diverse elements, including feature engineering, exploratory data analysis, and the implementation of sophisticated blended models and layered machine learning regression techniques. This approach is geared towards not only predicting house prices effectively but also fostering a deeper understanding of the underlying dynamics.

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

<img src="data/northridge_ames_iowa_aerial_calik.png" alt="Model Diagram" width="1280"/>


## Summary

### Data Preprocessing & Feature Engineering     
Train and validation datasets, were preprocessed separately to prevent data leakage. The preprocessing phase included robust feature engineering, resulting in a professional feature selection process. After comprehensive exploratory data analysis, feature engineering, and preprocessing, the dataset was streamlined to 36 features from the original 80. Notably, this dimension reduction was achieved primarily through sophisticated feature engineering techniques and variance of the variables, rather than mathematical feature selection techniques.

Addressing missing values involved a careful imputation process. Two new features, "Neighborhood Median House Price" and "Neighborhood Square Foot Property Price," were introduced to capture essential locality nuances. Year-related features were transformed into ages, and a systematic approach with custom functions was applied to features related to basement, terrace, garage, and bathrooms, where we successfully reduced these features into single or double features. Ordinal categorical variables were handled through custom ordinal mappings to retain information without inflating dimensionality via one-hot encoding.

### Exploratory Data Analysis  
Unveiling crucial insights, the exploration of the dataset revealed intriguing patterns. Northridge emerged as the priciest neighborhood, boasting a median property price of 315K-USD, while Meadow claimed the lowest at 88K-USD, underscoring vast neighborhood price differentials. The majority of properties found residence in Residential Low-Density Areas, comprising 90% detached single-family houses. Diversity in property floors (50% single, 30% double), and the prevalence of attached garages (70%) reflected notable housing characteristics. Additionally, conventional and normal sales transactions dominated at nearly 90%.

Various data visualizations and correlation analyses emphasized that a house's Overall Quality stood as the paramount factor influencing Sale Price. Features like Median Neighborhood House Prices (newly generated), Ground Living Area, External Quality, and Kitchen Quality also wielded substantial impact. In contrast, variables such as Months and Years of property sale, Land Contour, MSSubClass, and surprisingly Overall Condition and External Condition exhibited lower correlations with the target feature.

Outliers, flagged through scatterplots and boxplots, prompted meticulous curation, considering real estate intricacies, leading to the exclusion of four observations among 1460. Notably, LotArea and MasVnrArea faced removal due to a plethora of extreme values, posing a risk of misguidance for the models.

Multicollinearity scrutiny encompassed numerical and categorical features, deploying techniques like correlation analysis and Cramer's V values. In line with real estate norms, intercorrelation among features prompted a discerning approach—some features retained, while others bid farewell. For features on the chopping block, a Random Forest Regressor gauged their impact on the target.

To align SalePrice with a normal distribution and address skewness (1.88), log transformation was applied, marking a pivotal step in preparation for subsequent advanced regression techniques. This intricate yet indispensable interplay of preprocessing and exploratory analysis laid a robust foundation for subsequent model building, reaffirming the pivotal role of feature engineering in this study.


<img src="data/model_diagram.png" alt="Model Diagram" width="700"/>

## Methodology

### Machine Learning Models:
The model-building process began systematically by employing a collection of 10 diverse algorithms, covering linear models, tree-based models, and ensemble methods like Gradient Boosting Regressor and LightGBM. Each algorithm underwent encapsulation within a pipeline through a custom function, with thorough evaluation focusing on metrics such as R2 score, mae, mse, RMSE, and cross-validated scores. But **RMSE was chosen the main metric** due to its sensitivity to the magnitude of errors, interpretability in the original unit of the target variable, mathematical convenience, and widespread use as a benchmark in the field.

### Key Findings & The Importance of Feature Engineering:
A significant outcome of this notebook underscores the importance of domain knowledge and feature engineering. Default models achieved a 0.916 R2 score and 0.115 RMSE score without hyperparameter tuning, model blending or stacking, or further feature selection methods. Feature engineering and exploratory data analysis were pivotal in this accomplishment.

Ridge, Lasso, and Gradient Boost models outperformed, while KNeighborsRegressor and DecisionTreeRegressor lagged. Notably, Gradient Boosting Regressor and Light GBM Regressor excelled among tree-based models, whereas XGBoost, often a Kaggle champion, underperformed in our study.

### Feature Importances and Recursive Feature Elimination (RFE):

Feature importances, particularly of the initial models (specifically GBRegressor), closely aligned with findings from exploratory data analysis. Moreover, we employed SHAP (SHapley Additive exPlanations), a widely recognized technique for interpreting complex machine learning models. The utilization of SHAP allowed us to gain insights into the decision-making processes of our models, elucidating the contribution of each feature towards model predictions. All these analysis showed that:  

* Ground Living Area  
* Overall Quality  
* Median House Price (Neighborhood / location)  
* External Quality
* Total Bathrooms  
* Total Rooms 

have been the most important features for deciding a house price.

<img src="data/best_features.png" alt="Model Diagram" width="600"/>

### Base Layer: Building Blended Model
The blending phase incorporated a linear model (Ridge Regressor), Support Vector Regressor, Light GBM Regressor, and Gradient Boosting Regressor. This approach to model fusion, creating a base layer with a blended model, involved experimentation with different algorithms and weightings. VotingRegressor facilitated this process, elevating performance from a single best model with a 0.91 R2 Score and 0.116 RMSE to 0.92 R2 Score and 0.112 RMSE.

### Stacked Model: Add Meta Layer
Moving beyond blending, the stacking phase introduced a meta layer to the model architecture. Base layer predictions became new features, enriching the dataset for the meta model. Rigorous evaluation of potential meta models, including Gradient Boosting Regressor, confirmed the effectiveness of this advanced ensemble modeling technique.

The comprehensive evaluation strategy, involving continuous model adjustment, culminated in a notable achievement —a 0.95 R2 score and 0.089 RMSE—a substantial improvement from initial stages, securing a 4% position on the Kaggle scoreboard, despite not explicitly optimized for Kaggle scoreboard.

### Model Deployment
The final step involved deploying the stacked model with the entire dataset. This intricate model underwent a final training iteration, leveraging the entire dataset for enhanced predictive capabilities. This deployment marked the conclusion of a comprehensive guide for data preprocessing, exploratory data analysis, regression techniques, and advanced ensemble modeling.


## Conclusion

### Final Reflections, Challenges, and Opportunities for Enhancement    

This notebook prioritized swift runtime, minimizing detailed model tuning to ensure accessibility, aiming for a five-minute runtime on a medium-end computer including also entire hyperparameter tuning. While the current model achieved impressive scores through the incorporation of domain knowledge, robust data analysis, feature engineering and advanced machine learning techniques, potential enhancements lie in fine-tuning aspects like houses' remodeling age, OverallCondition, ExterCond and possible inflation rates. Addressing outliers in features such as LotArea and MasVnrArea offers additional opportunities for model improvement. Moreover, deeper hyperparamtre tuning or incorporating neural networks can contribute as well. The central challenge faced was efficient dimension reduction amid the wealth of available features.


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


