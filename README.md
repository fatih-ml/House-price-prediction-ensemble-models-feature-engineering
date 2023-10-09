# House Prices - Advanced Regression Techniques

This repository contains my work in progress for the Kaggle competition [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). The goal of this project is to predict the sale prices of houses based on various features.

## Progress Report

As of now, I have completed the following steps:

1. Read and Explored the Data: I've thoroughly examined the dataset to understand its structure, features, and relationships.

2. Handling Missing Values: I've taken special care in addressing missing values. Some were dropped, but for the majority, I recognized that they actually indicate the absence of a feature. For example, if there's a missing value in the 'GarageBuiltYear' column, it signifies that there is no garage in the property, based on insights from the data explanation file.

3. Data Preprocessing: At the start, I combined both the training and test datasets before proceeding with feature engineering and preprocessing.

## Work in Progress

Currently, I am focused on the following tasks:

1. Feature Engineering: I am in the process of creating new features based on logical deductions and my domain knowledge.

2. Handling Cardinality: Next, I will address high and low cardinality features to streamline the dataset.

3. Addressing Collinearity: I'll identify and handle multicollinearity to ensure that the features used for modeling are independent.

4. Encoding, Normalization, and Standardization: I'll prepare the data for modeling by applying appropriate transformations.

## Next Steps

After completing the preprocessing steps, my plan is to:

1. Model Building: I'll start with selecting appropriate models and training them on the preprocessed data.

2. Hyperparameter Tuning: I'll optimize the models to achieve the best possible performance.

3. Evaluation and Fine-tuning: I'll rigorously evaluate the models and fine-tune them as needed for optimal results.

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


