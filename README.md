# Code for MVNO Purification and Ooptimization

## Table of Contents
[Introduction](#introduction)

[Codebase Organization](#codebase-organization)
 - [Sample Dataset](#sampled-dataset)

 - [Data Usage Prediction](#data-usage-prediction)

 - [Customer Churn Prediction](#custmer-churn-prediction)

 - [Fraud Detection](#fraud-detection)

[Platform Requirements](#platform-requirements)

## Introduction
This repository contains our code for the technical optimization of a commercial MVNO (Xiaomi Mobile) from 3 key aspects (data usage prediction, custmer churn prediction and fraud detection), as well as a dataset with limited samples, which is allowed to be disclosed after negotiation with Xiaomi Mobile. Our solution is adjusted for the sample dataset. Therefore, you'll be able to run codes in this repo with the sampled data.

## Codebase Organization

### [The entire codebase is available in our [Github repo](https://github.com/MVNO-Optimization/MVNO-Optimization.github.io).]

### Sample Dataset
The samples are 100 active customers randomly selected and extracted from the database of Xiaomi Mobile in July 2020. Specific attributes are divided into 3 categories: history data usages, characteristics of drop-out users and features of scammers, as shown in [Data](https://github.com/MVNO-Optimization/MVNO-Optimization.github.io/tree/main/data).

Dataset `samples-data-usage` contains history data usages (18 months at most) of the customers:

| Attribute | Description |
| ---- | ---- |
| `id` | Unique ID generated to identify a user (cannot be related to the user's true indentity) |
| `month_i` | Data usage of a user (in MB) in the i-th month, i ∈ [1,18] |

Dataset `samples-churn-features` contains the 8-dimensional features obtained through feature correlation analysis. Here `age` is discretized by the interval of 10 years, for example, `20` means 20 to 30 years old and the largest discrete value `60` means more than 60 years old:

| Attribute | Description |
| ---- | ---- |
| `id` | Unique ID generated to identify a user (cannot be related to the user's true indentity) |
| `account_balance` | Account balance (in RMB) of a user  in the current month |
| `data_plan` | Monthly data (in GB) of a user's data plan |
| `last_month_expense` | Expense (in RMB) of a user in last month |
| `lifetime` | Life time (in months) of a user |
| `age` | Discretized age of a user |
| `avg_delta_expense` | Average delta of the monthly expenses (in RMB) of a user |
| `gender` | Gender of a user |
| `expense_variance` | Variance of the monthly expenses (in RMB) of a user |

Dataset `samples-scammer-features` contains the 9-dimensional features extracted through risk-return analysis of scammers combined with feature correlation analysis:

| Attribute | Description |
| ---- | ---- |
| `id` | Unique ID generated to identify a user (cannot be related to the user's true indentity) |
| `call_count_per_day` | Average number of outgoing calls per day of a user |
| `avg_duration` | Average duration (in seconds) of outgoing calls of a user  |
| `number_count` | Number of phone numbers in outgoing calls of a user |
| `rate` | Average frequency of per called phone number for out going calls of a user |
| `day_gap` | Interval (in days) between first bill time and activation time of a user |
| `first_cellId_percent` | Usage ration (%) of the most involved BS for outgoing calls of a user |
| `city_count` | Number of cities in outgoing calls of a user |
| `first_city_percent` | Usage ration (%) of the most called city for outgoing calls of a user |
| `province_count` | Number of provinces in outgoing calls of a user |

The developers can train models with the code of [Data Usage Prediction](https://github.com/MVNO-Optimization/MVNO-Optimization.github.io/tree/main/data-usage-prediction), [Customer Churn Prediction](https://github.com/MVNO-Optimization/MVNO-Optimization.github.io/tree/main/customer-churn-prediction) and [Fraud Detection](https://github.com/MVNO-Optimization/MVNO-Optimization.github.io/tree/main/data-usage-prediction) using the sample dataset, but the performance is limited compared with using the full dataset, due to the lack of samples in this sample dataset.

### Data Usage Prediction
We formulate data usage prediction into a supervised time series forecasting problem. We employ a robust statistical method (Grubb’s Test) and neighbor mean interpolation to complement the ML-based method (RandomForest). The proposed approach leads to an average prediction accuracy of 93.3% on the full dataset.

### Customer Churn Prediction
We use the extracted features to proactively predict customers’ churn with RandomForest classifier. Additionally, we employ an under-sampling based method called One-Sided Selection to mitigate the negative impact caused by imbalanced positive and negative samples in the full dataset. Eventually, we manage to achieve both high precision (96.3%) and recall (97.8%) with the full dataset. 

### Fraud Detection
Also considering the class imbalance in our dataset, we adopt a resampling based method named SMOTETomek combined with `BalancedRandomForestClassifier`, which can achieve a precision of 90% and a recall of 92.4% with the full dataset.

## Platform Requirements
python==3.8

numpy==1.20.0+

scikit-learn==0.24.0+

imbalanced-learn==0.7.0+
