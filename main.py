import pandas as pd
df = pd.read_csv('Bus_Breakdown_and_Delays.csv')

sampled_df = df.sample(n=10000)
sampled_df.to_csv('sampled_data.csv', index=False)
# # -*- coding: utf-8 -*-
#
# # 01-BU
#
# """
#
# Predict delay time based on some attributes.
#
# """
#
# # 02-DU
#
# import pandas as pd
# import matplotlib.pyplot as plt
# #
# # # Load Dataset
# #
# # file = 'Bus_Breakdown_and_Delays.csv'
# #
# # # read csv file
# # df = pd.read_csv(file)
# #
# # # set max columns to display
# # pd.set_option('display.max_columns', 30)
# #
# # # show first 5 rows
# # df.head()
# #
# # # show information about the dataset
# # df.info()
# #
# # # Explore Data
# #
# # # add various groups of columns
# # colname_ID1 = ['Busbreakdown_ID']
# # colname_ID2 = ['Bus_No', 'Route_Number','Schools_Serviced','Incident_Number']
# # colname_category = ['School_Year', 'Run_Type',  'Reason',   'Boro', 'Bus_Company_Name']
# # colname_time = ['Occurred_On', 'Created_On', 'Informed_On', 'Last_Updated_On']
# # colname_num = ['Number_Of_Students_On_The_Bus']
# # colname_flag = ['Has_Contractor_Notified_Schools', 'Has_Contractor_Notified_Parents', 'Have_You_Alerted_OPT', 'Breakdown_or_Running_Late', 'School_Age_or_PreK']
# # colname_target = ['How_Long_Delayed']
# #
# # # explore target columns
# # delay_time_category = ['16-30 Min', '31-45 Min', '0-15 Min', '46-60 Min', '61-90 Min']
# # df[colname_target][df['How_Long_Delayed'].isin(delay_time_category)].describe()
# # df[colname_target][df['How_Long_Delayed'].isin(delay_time_category)].value_counts()
# # # df[colname_target][df['How_Long_Delayed'].isin(delay_time_category)].groupby('How_Long_Delayed').agg({'How_Long_Delayed': 'count'}).plot(kind='bar')
# # # plt.ylabel('Count')
# # # plt.xticks(rotation=0)
# #
# # # explore numerical columns
# # df[colname_num].describe()
# # # df[df['How_Long_Delayed'].isin(delay_time_category)][df['Number_Of_Students_On_The_Bus'] < 100].groupby('How_Long_Delayed').agg({'Number_Of_Students_On_The_Bus': 'mean'}).plot(kind='bar')
# # # plt.ylabel('Mean of Number of Students on the Bus')
# # # plt.xticks(rotation=0)
# #
# # # explore categorical columns
# # df[colname_category].describe()
# # for item in colname_category:
# #     df[item].value_counts()
# #
# # # for item in colname_category:
# # #     df[item].value_counts().plot(kind='bar')
# # #     plt.ylabel('Count')
# # #     plt.xlabel(item)
# # #     plt.xticks(rotation=15)
# #
# # # for item in colname_category:
# # #     df[df['How_Long_Delayed'].isin(delay_time_category)][item].groupby(df['How_Long_Delayed']).value_counts().unstack(0).plot(kind='bar')
# # #     plt.ylabel('Count')
# # #     plt.xlabel(item)
# # #     plt.xticks(rotation=15)
# #
# # # explore flag columns
# # df[colname_flag].describe()
# #
# # # for item in colname_flag:
# # #     df[df['How_Long_Delayed'].isin(delay_time_category)][item].groupby(df['How_Long_Delayed']).value_counts().unstack(0).plot(kind='bar')
# # #     plt.ylabel('Count')
# # #     plt.xlabel(item)
# # #     plt.xticks(rotation=0)
# # #     plt.show()
# #
# # # explore time columns
# # df[colname_time].describe()
# #
# # # explore ID columns
# # df[colname_ID1].describe()
# # df[colname_ID2].describe()
# #
# # # Verify Data Quality
# # missing_data = df.isnull().sum()
# #
# # # Display columns with missing data and their respective counts
# # print(missing_data[missing_data > 0])
# #
# # from scipy import stats
# #
# # # Calculate Z-scores for a specific column
# # df[df['Number_Of_Students_On_The_Bus'] > 100]['Number_Of_Students_On_The_Bus'].describe()
# # z_scores = stats.zscore(df[df['Number_Of_Students_On_The_Bus'] < 100]['Number_Of_Students_On_The_Bus'])
# # df[df['Number_Of_Students_On_The_Bus'] < 100][abs(z_scores) < 5]['Number_Of_Students_On_The_Bus'].describe()
#
#
# # # 03-DP
# #
# # # import supplementary data
# # file_supplement1 = 'bus-breakdown-and-delays.csv'
# # df1 = pd.read_csv(file_supplement1)
# # file_supplement2 = 'bus-breakdown-and-delays2.csv'
# # df2 = pd.read_csv(file_supplement2)
# #
# # # merge supplementary data
# # set(df1.columns) & set(df2.columns)
# # df_supplement = pd.merge(df1, df2, on=['School_Year', 'Busbreakdown_ID'])
# # print(df_supplement.shape)
# #
# # # merge main data with supplementary data
# # df = pd.concat([df, df_supplement], ignore_index=True)
# # print(df.shape)
# #
# # # fill missing values
# # missing_data = df.isnull().sum()
# # print(missing_data[missing_data > 0])
# # df['How_Long_Delayed'][~df['How_Long_Delayed'].isin(delay_time_category)] = '91-Inf Min'
# # df['How_Long_Delayed'].value_counts()
# # df.dropna(subset=['Run_Type', 'Reason', 'Boro', 'School_Age_or_PreK'], inplace=True)
# # missing_data = df.isnull().sum()
# # print(missing_data[missing_data > 0])
# #
# # # deal with extreme values
# # df[df['Number_Of_Students_On_The_Bus'] > 100]['Number_Of_Students_On_The_Bus'].describe()
# # z_scores = stats.zscore(df[df['Number_Of_Students_On_The_Bus'] < 100]['Number_Of_Students_On_The_Bus'])
# # df[df['Number_Of_Students_On_The_Bus'] < 100][abs(z_scores) < 5]['Number_Of_Students_On_The_Bus'].describe()
# # df = df[df['Number_Of_Students_On_The_Bus'] < 20]
# #
# # # convert time columns to datetime
# # print(df['Occurred_On'].dtype)
# # df['Occurred_On'] = pd.to_datetime(df['Occurred_On'])
# # print(df['Occurred_On'].dtype)
# #
# # # derive new columns hour day of week and month
# # df['Occurred_On_Hour'] = df['Occurred_On'].dt.hour
# # df['Occurred_On_DayOfWeek'] = df['Occurred_On'].dt.dayofweek + 1
# # df['Occurred_On_Month'] = df['Occurred_On'].dt.month
# # colname_time_new = ['Occurred_On_Hour', 'Occurred_On_DayOfWeek', 'Occurred_On_Month']
# # for item in colname_time_new:
# #     df[item].value_counts().sort_index().plot(kind='bar')
# #     plt.ylabel('Count')
# #     plt.xlabel(item)
# #     plt.xticks(rotation=0)
# #
# # for item in colname_time_new:
# #     df[item].groupby(df['How_Long_Delayed']).value_counts().unstack(0).plot(kind='bar')
# #     plt.ylabel('Count')
# #     plt.xlabel(item)
# #     plt.xticks(rotation=0)
# #
# # # data selection
# # df = df[['School_Year', 'Run_Type', 'Reason', 'Boro', 'Bus_Company_Name',
# #          'School_Age_or_PreK', 'Number_Of_Students_On_The_Bus', 'Occurred_On_Hour',
# #          'Occurred_On_Month', 'How_Long_Delayed', 'Has_Contractor_Notified_Schools',
# #             'Has_Contractor_Notified_Parents', 'Have_You_Alerted_OPT']]
# # df.info()
# # df.to_csv('cleaned.csv', index=False)
#
# # 04-DT
#
# # # import cleaned data
# file = 'cleaned.csv'
# df = pd.read_csv(file)
# #
# # # data reduction
# #
# # # define X and y
# # X = df.drop('How_Long_Delayed', axis=1)
# # y = df['How_Long_Delayed']
# #
# # # separate categorical and numerical columns
# # colname_category = ['School_Year', 'Run_Type', 'Reason', 'Boro',
# #                     'Bus_Company_Name', 'School_Age_or_PreK',
# #                     'Has_Contractor_Notified_Schools', 'Has_Contractor_Notified_Parents',
# #                     'Have_You_Alerted_OPT']
# # colname_num = ['Number_Of_Students_On_The_Bus', 'Occurred_On_Hour', 'Occurred_On_Month']
# #
# # # convert categorical columns to dummy variables
# # X = pd.get_dummies(X, columns=colname_category, drop_first=True)
# # y = pd.get_dummies(y, drop_first=True)
# # #
# # # from sklearn.linear_model import LinearRegression
# # # r10=LinearRegression().fit(X, y)
# # # print(r10.score(X, y))
# # #
# # group2 = ['School_Year','Number_Of_Students_On_The_Bus', 'Occurred_On_Hour', 'Occurred_On_Month']
# # group3 = ['Run_Type', 'Reason', 'Boro', 'Bus_Company_Name']
# # group4 = ['School_Age_or_PreK', 'Has_Contractor_Notified_Schools', 'Has_Contractor_Notified_Parents', 'Have_You_Alerted_OPT']
# # X = df.drop('How_Long_Delayed', axis=1)
# # X_2 = pd.get_dummies(X[group2], columns=group2, drop_first=True)
# # X = df.drop('How_Long_Delayed', axis=1)
# # X_3 = pd.get_dummies(X[group3], columns=group3, drop_first=True)
# # X = df.drop('How_Long_Delayed', axis=1)
# # X_4 = pd.get_dummies(X[group4], columns=group4, drop_first=True)
# # #
# # # r20=LinearRegression().fit(X_2, y)
# # # r20.score(X_2, y)
# # # r30=LinearRegression().fit(X_3, y)
# # # r30.score(X_3, y)
# # # r40=LinearRegression().fit(X_4, y)
# # # r40.score(X_4, y)
# #
# # # feature selection
# df = df[['Run_Type', 'Reason', 'Boro', 'Bus_Company_Name', 'How_Long_Delayed']]
# import numpy as np
# #
# # # data projection
# # # logaritmic transformation`
# df['How_Long_Delayed'] = df['How_Long_Delayed'].map({'0-15 Min': 15, '16-30 Min': 30, '31-45 Min': 45,
#                                                      '46-60 Min': 60, '61-90 Min': 90, '91-Inf Min': 360})
# # df['How_Long_Delayed'].describe()
# # # df['How_Long_Delayed'].plot(kind='hist', bins=10)
# #
# df['How_Long_Delayed'] = np.log(df['How_Long_Delayed'])
# # df['How_Long_Delayed'].describe()
# # # df['How_Long_Delayed'].plot(kind='hist', bins=10)
# #
#
# # 05-DMM
# """
# This is a regression problem.
# We want to predict a numeric outcome based on other values.
# """
#
# # 06-DM
#
#
# # Generalized Linear Models
#
# import statsmodels.api as sm
#
#
# # Preprocess categorical variables into dummy variables
# # df = pd.get_dummies(df, columns=['Run_Type', 'Reason', 'Boro', 'Bus_Company_Name'], drop_first=True)
# #
# # # Define the variables
# # X = df.drop('How_Long_Delayed', axis=1)
# # y = df['How_Long_Delayed']
# #
# # # Add a constant term to the predictor variables for the intercept
# # X = sm.add_constant(X)
#
# # # Fit the GLM model
# # glm_model = sm.GLM(y, X, family=sm.families.Gaussian()).fit()
# #
# # # View the summary of the GLM model
# # print(glm_model.summary())
#
# # Decision Tree
#
# from sklearn.tree import DecisionTreeRegressor
#
# # # Fit the Decision Tree model
# # dt_model = DecisionTreeRegressor().fit(X, y)
# #
# # # Sort the feature importances
# # sorted_idx = dt_model.feature_importances_.argsort()
# #
# # # print the first 10 feature importances and the feature names
# # print(dt_model.feature_importances_[sorted_idx][-10:])
# # print(X.columns[sorted_idx][-10:])
# #
# # # Plot the feature importances
# # plt.barh(X.columns[sorted_idx], dt_model.feature_importances_[sorted_idx])
# # plt.xlabel("Decision Tree Feature Importance")
# # plt.show()
#
#
# # Lasso Regression
#
# # from sklearn.linear_model import Lasso
# #
# # # Fit the Lasso model
# # lasso_model = Lasso(alpha=0.01).fit(X, y)
# #
# # # Sort the feature importances
# # sorted_idx = lasso_model.coef_.argsort()
# #
# # # print the first 10 feature importances and the feature names
# # print(lasso_model.coef_[sorted_idx][-10:])
# # print(X.columns[sorted_idx][-10:])
# #
# # # Plot the feature importances
# # plt.barh(X.columns[sorted_idx], lasso_model.coef_[sorted_idx])
# # plt.xlabel("Lasso Feature Importance")
# # plt.show()
# #
# # # Random Forest
# #
# # from sklearn.ensemble import RandomForestRegressor
# #
# # # Fit the Random Forest model
# # rf_model = RandomForestRegressor().fit(X, y)
# #
# # # Sort the feature importances
# # sorted_idx = rf_model.feature_importances_.argsort()
# #
# # # print the first 10 feature importances and the feature names
# # print(rf_model.feature_importances_[sorted_idx][-10:])
# # print(X.columns[sorted_idx][-10:])
# #
# # # Plot the feature importances
# #
# # plt.barh(X.columns[sorted_idx], rf_model.feature_importances_[sorted_idx])
# # plt.xlabel("Random Forest Feature Importance")
# # plt.show()
#
# # GLM
# # select parameters
#
# # glm_model1 = sm.GLM(y, X, family=sm.families.Gaussian(link=sm.families.links.log()))
# # glm_results1 = glm_model1.fit()
# # print(glm_results1.summary())
# #
# # glm_model2 = sm.GLM(y, X, family=sm.families.Poisson(link=sm.families.links.log()))
# # glm_results2 = glm_model2.fit()
# # print(glm_results2.summary())
# #
# # glm_model3 = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.log()))
# # glm_results3 = glm_model3.fit()
# # print(glm_results3.summary())
# #
# # glm_model4 = sm.GLM(y, X, family=sm.families.Gaussian(link=sm.families.links.identity()))
# # glm_results4 = glm_model4.fit()
# # print(glm_results4.summary())
# #
# # glm_model5 = sm.GLM(y, X, family=sm.families.Poisson(link=sm.families.links.identity()))
# # glm_results5 = glm_model5.fit()
# # print(glm_results5.summary())
# #
# # glm_model6 = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.identity()))
# # glm_results6 = glm_model6.fit()
# # print(glm_results6.summary())
#
# # glm_model7 = sm.GLM(y, X, family=sm.families.Binomial(link=sm.families.links.log()))
# # glm_results7 = glm_model7.fit()
# # print(glm_results7.summary())
# #
# # glm_model8 = sm.GLM(y, X, family=sm.families.Binomial(link=))
# # glm_results8 = glm_model8.fit()
# # print(glm_results8.summary())
#
#
# # 07-DM
#
# # Generalized Linear Models
#
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import r2_score
#
# # # Split the data into training and test sets with 15% of the data in the test set
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=123)
# #
# # # Output the training phase and test phase
# # X_train.to_csv('X_train.csv', index=False)
# # X_test.to_csv('X_test.csv', index=False)
# # y_train.to_csv('y_train.csv', index=False)
# # y_test.to_csv('y_test.csv', index=False)
#
# # X_train = pd.read_csv('X_train.csv')
# # y_train = pd.read_csv('y_train.csv')
# # X_test = pd.read_csv('X_test.csv')
# # y_test = pd.read_csv('y_test.csv')
# #
# # # Fit the GLM model
# # glm_model = sm.GLM(y_train, X_train, family=sm.families.Gaussian(link=sm.families.links.identity())).fit()
# #
# # # View the summary of the GLM model
# # print(glm_model.summary())
# #
# # # Make predictions on the test set
# # glm_predictions = glm_model.predict(X_test)
# #
# #
# # # Calculate the mean squared error of the model
# # glm_mse = mean_squared_error(y_test, glm_predictions)
# # print("The mean squared error of the model is:", glm_mse)
# #
# # # Calculate the root mean squared error of the model
# # glm_rmse = np.sqrt(glm_mse)
# # print("The root mean squared error of the model is:", glm_rmse)
# #
# # # Calculate the coefficient of determination
# # glm_r2 = r2_score(y_test, glm_predictions)
# # print("The coefficient of determination of the model is:", glm_r2)
#
# # 08 - Interpretation
#
# # Visualise the delay time against the predictor variables
# # df.plot.scatter(x='Run_Type', y='Reason', c='How_Long_Delayed', colormap='viridis')
# # df.plot.scatter(x='Run_Type', y='Boro', c='How_Long_Delayed', colormap='viridis')
# # df.plot.scatter(x='Reason', y='Boro', c='How_Long_Delayed', colormap='viridis')
# # plt.xticks(rotation=15)
# # plt.show()
#
# # # Visualise the model predictions against the actual values
# # boxplot = pd.DataFrame({'Actual': y_test['How_Long_Delayed'], 'Predicted': glm_predictions})
# # boxplot.plot.box()
# # plt.show()
#
# # Visualise the model residuals
# # residuals = y_test['How_Long_Delayed'] - glm_predictions
# # residuals.plot.hist(bins=20)
# # plt.show()
# #
# # sm.qqplot(residuals, line='45', fit=True)
# # plt.title("Q-Q Plot of Residuals")
# # plt.show()
#
# # iteration1
# # df = df[df['How_Long_Delayed'] < 5.8]
# # df = pd.get_dummies(df, columns=['Run_Type', 'Reason', 'Boro', 'Bus_Company_Name'], drop_first=True)
# #
# # # Define the variables
# # X = df.drop('How_Long_Delayed', axis=1)
# # y = df['How_Long_Delayed']
# #
# # # Add a constant term to the predictor variables for the intercept
# # X = sm.add_constant(X)
# #
# # # Split the data into training and test sets with 15% of the data in the test set
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=123)
# #
# #
# # # Fit the GLM model
# # glm_model = sm.GLM(y_train, X_train, family=sm.families.Gaussian(link=sm.families.links.identity())).fit()
# #
# # # View the summary of the GLM model
# # print(glm_model.summary())
# #
# # # Make predictions on the test set
# # glm_predictions = glm_model.predict(X_test)
# #
# #
# # # Calculate the mean squared error of the model
# # glm_mse = mean_squared_error(y_test, glm_predictions)
# # print("The mean squared error of the model is:", glm_mse)
# #
# # # Calculate the root mean squared error of the model
# # glm_rmse = np.sqrt(glm_mse)
# # print("The root mean squared error of the model is:", glm_rmse)
# #
# # # Calculate the coefficient of determination
# # glm_r2 = r2_score(y_test, glm_predictions)
# # print("The coefficient of determination of the model is:", glm_r2)
#
# # iteration2
# df = df[['Run_Type', 'Reason', 'Boro', 'How_Long_Delayed']]
# df = pd.get_dummies(df, columns=['Run_Type', 'Reason', 'Boro'], drop_first=True)
#
# # Define the variables
# X = df.drop('How_Long_Delayed', axis=1)
# y = df['How_Long_Delayed']
#
# # Add a constant term to the predictor variables for the intercept
# X = sm.add_constant(X)
#
# # Split the data into training and test sets with 15% of the data in the test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=123)
#
#
# # Fit the GLM model
# glm_model = sm.GLM(y_train, X_train, family=sm.families.Gaussian(link=sm.families.links.identity())).fit()
#
# # View the summary of the GLM model
# print(glm_model.summary())
#
# # Make predictions on the test set
# glm_predictions = glm_model.predict(X_test)
#
#
# # Calculate the mean squared error of the model
# glm_mse = mean_squared_error(y_test, glm_predictions)
# print("The mean squared error of the model is:", glm_mse)
#
# # Calculate the root mean squared error of the model
# glm_rmse = np.sqrt(glm_mse)
# print("The root mean squared error of the model is:", glm_rmse)
#
# # Calculate the coefficient of determination
# glm_r2 = r2_score(y_test, glm_predictions)
# print("The coefficient of determination of the model is:", glm_r2)