# Project Report: Comprehensive Analysis of Customer Churn



## 1_Introduction

**Objective and Scope**

The primary objective of this project is to analyze customer churn behavior using a dataset containing diverse attributes such as contract types, internet services, and demographic information. This analysis aims to uncover the key factors driving churn, develop predictive models to forecast customer attrition, and propose actionable strategies to improve customer retention. The project encompasses Exploratory Data Analysis (EDA), Clustering Analysis, and Predictive Modeling.



## 2_Exploratory Data Analysis (EDA)

**2_1_Descriptive Statistics**

Understanding the basic statistics of the dataset is crucial for uncovering trends and patterns. The descriptive statistics for customer churn are segmented by contract type, revealing significant variations in customer characteristics based on churn status and contract type:

 ***Churn by Contract Type:*** <br />
**MonthtoMonth Contracts:** <br />
Nonchurn Customers: The mean monthly charge is \$61.5, with a standard deviation of \$27.9, indicating substantial variability in charges.
<br />

Churn Customers: The mean monthly charge rises to \$73.0, with a standard deviation of \$24.1. This increase in average charges among churned customers suggests a relationship between higher monthly charges and increased churn.

**OneYear Contracts:** <br />
Nonchurn Customers: The mean monthly charge is \$62.5, with a standard deviation of \$31.7.
<br />   

Churn Customers: The mean monthly charge increases to \$85.1, with a standard deviation of \$25.6. This trend suggests that higher charges are also associated with higher churn rates for oneyear contracts.

**TwoYear Contracts:** <br />
Nonchurn Customers: The mean monthly charge is \$60.0, with a standard deviation of \$34.5.
<br />

Churn Customers: The mean monthly charge is \$86.8, with a standard deviation of \$28.9. The higher mean charges for churned customers indicate that higher pricing may contribute to churn even in longterm contracts.


***Churn Rates by Contract Type:*** <br />
   MonthtoMonth: 42.7% <br />
   OneYear: 11.3% <br />
   TwoYear: 2.8%


The data suggests that customers with monthtomonth contracts are significantly more likely to churn compared to those with longerterm contracts. This highlights the potential benefit of encouraging customers to opt for longerterm contracts to improve retention.

**2.2. Feature Correlation**

Correlation analysis was conducted to identify the relationships between various features and churn:
<br />
***Monthly Charges and Tenure:*** <br />
Both features exhibit a positive correlation with churn rate, indicating that customers with higher monthly charges and shorter tenures are more likely to churn. <br />

***Senior Citizen Status:*** <br /> There is a moderate positive correlation with churn rate, suggesting that senior citizens may be slightly more prone to churn compared to nonsenior citizens.

**2.3. Data Distribution and Skewness**

The distribution of key features was analyzed to understand their characteristics and how they may influence churn: <br />

***Tenure:*** <br /> Exhibits rightskewed distribution (Skewness: 0.239, Kurtosis: 1.387). This suggests that a majority of customers have relatively short tenures, with a smaller number having longer tenures. <br />
***Monthly Charges:*** <br /> Approximates a normal distribution (Skewness: 0.014, Kurtosis: 1.300), indicating stable charge patterns across customers. <br />
 
***Churn:*** <br /> Shows significant right skewness (Skewness: 1.063) and flatness (Kurtosis: 0.870), reflecting that churn is less common but has a substantial impact when it occurs.<br />



## 3. Clustering Analysis

**3.1. Cluster Centers**
<br />
Clustering analysis was performed to segment the customer base into distinct groups based on their attributes. The identified cluster centers provide insights into different customer segments:
<br />
1. Cluster 0: Characterized by high charges and high churn rates. <br />
2. Cluster 1: Represents customers with moderate charges and moderate churn rates. <br />
3. Cluster 2: Consists of customers with low charges and low churn rates. <br />
4. Cluster 3: Features customers with high tenure and moderate charges. <br />
5. Cluster 4: Includes customers with low tenure and high charges. <br />
6. Cluster 5: Contains customers with moderate tenure and low charges. <br />

**3.2. Cluster Distribution**
<br />
The distribution of customers across the identified clusters is as follows:
<br />
 Cluster 5: 1,458 customers <br />
 Cluster 2: 1,409 customers <br />
 Cluster 3: 1,378 customers <br />
 Cluster 1: 1,194 customers <br />
 Cluster 4: 922 customers <br />
 Cluster 0: 682 customers <br />

Clusters with high churn rates tend to be characterized by monthtomonth contracts and high monthly charges. In contrast, clusters with lower churn rates typically have longer tenures and lower charges. This clustering insight underscores the importance of contract type and pricing in determining customer churn.

## 4. Predictive Modeling

**4.1. Model Performance**
<br />
An Artificial Neural Network (ANN) model was developed to predict customer churn. The performance metrics for the model are as follows:
<br />
 Training Accuracy: 79.73% <br />
 Training Loss: 0.4192 <br />
 Validation Accuracy: 81.19% <br />
 Validation Loss: 0.4057 <br />

These metrics indicate that the model performs effectively in distinguishing between churn and nonchurn customers, with a slightly higher accuracy on the validation set compared to the training set. <br />

**4.2. Classification Report**
<br />
The classification report for the validation set provides the following metrics:
<br />
No Churn: Precision of 0.86 and Recall of 0.89, demonstrating strong performance in identifying customers who do not churn. <br />
Churn: Precision of 0.67 and Recall of 0.58, showing that while the model can identify churned customers, there is room for improvement in this area. <br />
ROCAUC (Resampled Data): 0.8540 <br />
Average Precision (Resampled Data): 0.6703 <br />

These results highlight that while the model is effective at predicting nonchurn customers, it could benefit from enhancements to improve its ability to detect churn cases.
<br />
**4.3. Feature Importance**
<br />
The importance of various features in predicting churn was assessed:
<br />
Tenure: Contributes 6.9% to the model's predictions. <br />
Monthly Charges: Contributes 2.1%. <br />
Internet Service (DSL): Contributes 2.5%. <br />
<br />
Tenure and monthly charges are identified as the most significant predictors of churn, emphasizing their critical role in the predictive model.

**4.4. Model Insights**
<br />
The ANN model reveals that tenure and monthly charges are the most influential factors in predicting churn. Customers with shorter tenures and higher monthly charges are more likely to churn. These insights suggest that focusing on these attributes could help in designing effective retention strategies.
<br />
 
## 5. Conclusions and Recommendations

**5.1. Key Findings**
<br />
The analysis provides several key insights into customer churn:
<br />

Customer Retention: Customers with monthtomonth contracts and higher monthly charges are at a higher risk of churn. <br />

Contract Types: Longerterm contracts are associated with significantly lower churn rates. <br />

Feature Impact: Tenure and monthly charges are the most significant predictors of churn.

**5.2. Recommendations**
<br />
Based on the findings, the following recommendations are proposed:
<br />

Retention Strategies: Develop targeted retention strategies for customers with monthtomonth contracts and high monthly charges. Consider offering incentives or discounts to encourage these customers to switch to longerterm contracts. <br />

Contract Policy: Revise contract options to make longterm contracts more attractive. This could include offering competitive pricing or additional benefits for longerterm commitments. <br />

Customer Segmentation: Utilize clustering insights to tailor marketing and customer support strategies based on cluster characteristics. This approach can help address the specific needs and preferences of different customer segments. <br />

**5.3. Future Work**
<br />
Future efforts should focus on the following areas: <br />

Model Enhancement: Explore additional features and advanced modeling techniques to improve the predictive accuracy of churn forecasts. <br />

Customer Feedback Integration: Implement mechanisms to gather and analyze customer feedback to gain deeper insights into the reasons behind churn and address these issues proactively. <br />

This comprehensive analysis provides actionable insights into customer churn and offers strategic recommendations to enhance customer retention efforts and reduce churn rates effectively. <br />



## Appendices

Appendix A: Data Preparation

Details of data preprocessing, including encoding categorical variables and scaling features, are documented.

Appendix B: Clustering Analysis Details

Full results of clustering analysis, including cluster characteristics and distribution, are provided.

Appendix C: Predictive Modeling Code

The code used for developing and training the ANN model, including scripts and configurations, is included for reference.

Appendix D: Model Performance Metrics

Detailed performance metrics, including accuracy, loss, and classification reports, are available for review.



This final report encapsulates a thorough analysis of customer churn, providing a detailed examination of the dataset, insights from EDA and clustering, and results from predictive modeling. The recommendations derived from this analysis are aimed at enhancing customer retention and reducing churn rates effectively.


