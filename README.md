# Short Term Volatility Prediction For Stocks
This project contains the final project of May-Summer 2024 Data Science Boot Camp

# Overview
This project aims to build an efficient model to predict short-term volatility for hundreds of stocks across different sectors. It is based on the Kaggle competition-Optiver Realized Volatility Prediction. Volatility studies are extremely useful for short-term trading and intraday derivatives trading. Scalpers and day traders use volatility to trade in options, as both option buyers and writers expect high volatility for better returns over time.

# Dataset
The dataset is provided by Optiver in Kaggle competition in folder **optiver-realized-volatility-prediction**
- Order book data, Trade book data, train and test csv files
- No data cleanning is needed
- The total number of different stocks is 112
- The total number of different time\_id is 3830
# Exploratory Data Analysis
The dataset contains six files with hundreds of millions of rows of highly granular financial data.To better understand the data, we first perform Exploratory Data Analysis (EDA) to investigate the datasets and summarize their main characteristics. We then calculate the following fundamental statistics to identify connections between features.
- Weighted averaged price (stock valuation):$$WAP = \frac{BidPrice \times AskSize + AskPrice \times BidSize}{BidSize + AskSize}$$
- Log returns: $$r_{t_1, t_2} = \log \left( \frac{S_{t_2}}{S_{t_1}} \right),$$ where $S_t$ is the price (approximated by WAP) of the stock $S$ at time $t$
- Realized volatility: $$\sigma = \sqrt{\sum_{t}r_{t-1, t}^2}$$
# Modeling Approach
Model perfomance is evaluated by root mean square percentage error 
$$RMSPE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} ((y_i - \hat{y}_i)/y_i)^2}$$
We begin by setting up a baseline model using the overall target mean value as the prediction for the next 10 minutes of volatility. During EDA, we add more features and investigate their correlations with the target variable. We select the three most correlated features  — wap1\_realized\_volatility, wap2\_realized\_volatility, and book\_bid\_ask\_price\_ratio\_realized\_volatility — to build linear regression and K Nearest Neighbors models. Ultimately, we use ensemble learning approaches, such as boosting, with a particular focus on the powerful gradient boosting package XGBoost. We can see that XGBoost demonstrated the best performance. 
| Models| RMSPE |
| ----------- | ----------- |
| Baseline Model(Overall mean) | 1.110330|
| Linear Regression | 0.352226 |
| KNN |  0.333281 |
|  XGBoost |  0.028044|


