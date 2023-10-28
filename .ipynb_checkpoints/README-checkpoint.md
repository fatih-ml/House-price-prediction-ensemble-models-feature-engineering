# House Prices - Advanced Regression Techniques

## Objective and Framework
With a professional background spanning over seven years in the real estate and finance sectors, I embarked on a comprehensive study of the Kaggle competition dataset: "House Prices - Advanced Regression Techniques."

## Dataset
This dataset comprises 1460 training and 1459 test samples, each equipped with 80 features, covering a wide range from SquareFoot area to Garage specifications. The primary objective is to predict SalePrices, making it a challenging yet intriguing task.

## Overview
In this study, the core focus was on meticulous data exploration, creative feature engineering, and a thorough analysis of what truly drives property prices. Outliers were addressed, revealing intriguing patterns in the residuals. Deeper analysis pointed towards a more aggressive approach to outlier removal, leading to a remarkable performance boost of nearly 10%.

A pivotal transformation was applied to the Neighborhood feature, converting it into median house prices to capture the locality's impact on property values. Additionally, the study delved into multivariate analysis, carefully weighing feature retention against elimination.

The modeling phase encompassed five diverse algorithms—linear regression, ridge regression, lasso regression, random forest regression, and gradient boosting regression. After rigorous cross-validation, gradient boosting emerged as the standout performer. Fine-tuning hyperparameters and optimization with further feature engineering culminated in an impressive 0.11 RMSE, underscoring the efficacy of this approach in predicting house prices.

## Key Findings
Key findings were abundant during this study, highlighting the substantial influence of 'Overall Quality', 'Median House Price', and 'Living Area' on Sale Prices. Moreover, 'External Quality', 'Garage Area' and 'total Bathrooms' emerged as significant factors influencing property values.

## Challenges and Future Improvements
While this notebook already yields promising results, there's ample room for further enhancement. Future iterations could delve into features like property age and renovation year, and explore more sophisticated and layered modeling techniques.

## Purpose of this Repository
The purpose of this repository is both a learning and demonstration. I encourage others to explore and gain insights from this structured and step-by-step methodology.

## Main Components:
Introduction
1. Data Exploration and Preprocessing
2. Exploratory Data Analysis
3. Data Transformations
4. Model Building
5. Hyperparameter Tuning and Optimizing
6. Make Final Predictions and Submission File

## Libraries Used
Python
Pandas for data manipulation
Matplotlib and Seaborn for data visualization
Scikit-learn for model building and evaluation


## Contributing

This project is open to contributions. If you'd like to collaborate, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

I'd like to express my gratitude to KAGGLE to create this dataset and learning possibility to all data science enthusiasts.

## Contact

You can reach out to me via linkedin https://www.linkedin.com/in/fatih-calik-469961237


