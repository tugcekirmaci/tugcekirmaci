#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate


# In[51]:


file_path = r'C:\Users\user\Desktop\Politus_database_200K_sample.csv'


# In[52]:


data = pd.read_csv(file_path)


# In[53]:


data['gender']


# In[54]:


gender_mapping = {'female': 0, 'male': 1}
data['gender_numeric'] = data['gender'].map(gender_mapping)
data['gender_numeric']


# In[55]:


data['age_group']


# In[56]:


age_group_map = {'<=18': 1, '19-29': 2, '30-39':3, '>=40': 4}
data['age_group_numeric'] = data['age_group'].map(age_group_map)
data['age_group_numeric']


# In[ ]:


columns_for_pairplot = ['gender_numeric', 'age_group_numeric', '_2023_kk_pro', '_2023_erdogan_pro', '_2023_erdogan_against', '_2023_kk_against', '_2023_emotion_ofke', '_2023_emotion_korku','_2023_emotion_kaygi','_2023_emotion_uzuntu','_2023_emotion_umutsuzluk','_2023_topic_elections_and_voting']
sns.pairplot(data[columns_for_pairplot])
plt.show()

correlation_matrix = data[columns_for_pairplot].corr()
print(correlation_matrix)


# In[57]:


def linear_regression_ols_multiple(X, Z, y):
    X = np.asarray(X)
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    Z = np.asarray(Z)
    if len(Z.shape) == 1:
        Z = Z.reshape(-1, 1)

    X_with_intercept = np.column_stack((np.ones(X.shape[0]), X, Z))

    model = sm.OLS(y, X_with_intercept)
    results = model.fit()

    return results

Z1 = data['gender_numeric']
Z2 = data['age_group_numeric']
Z3 = data['total_tweet_num']
X1 = data['_2023_kk_pro']
X2 = data['_2023_erdogan_pro']
X3 = data['_2023_erdogan_against']
X4 = data['_2023_kk_against']
X5 = data['_2023_emotion_ofke']
y = data['_2023_topic_elections_and_voting']

results = linear_regression_ols_multiple(np.column_stack((X1, X2, X3, X4, X5)), np.column_stack((Z1, Z2, Z3)), y)

print(results.summary())

# In[58]:


residuals = results.resid
fitted_values = results.fittedvalues

plt.figure(figsize=(8, 6))
sns.scatterplot(x=fitted_values, y=residuals, color='blue', alpha=0.5)

plt.axhline(y=0, color='red', linestyle='--')

plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values')

plt.show()
sns.scatterplot(x=results.fittedvalues, y=residuals, ax=axes[0, 0])
axes[0, 0].set_title('Residuals vs. Fitted Values')


# In[59]:

# Losing outliers through calculating Cook's distance

standardized_residuals = results.get_influence().resid_studentized_internal
leverage_values = results.get_influence().hat_matrix_diag
cook_distance = standardized_residuals ** 2 * leverage_values / (1 - leverage_values)

threshold = 4 / len(y)  # You can adjust the threshold based on your data size

outliers_cooks = np.where(cook_distance > threshold)[0]


# In[60]:


# Losing outliers
cleaned_data = data.drop(index=outliers_cooks)

Z1_cleaned = cleaned_data['gender_numeric']
Z2_cleaned = cleaned_data['age_group_numeric']
Z3_cleaned = cleaned_data['total_tweet_num']
X1_cleaned = cleaned_data['_2023_kk_pro']
X2_cleaned = cleaned_data['_2023_erdogan_pro']
X3_cleaned = cleaned_data['_2023_erdogan_against']
X4_cleaned = cleaned_data['_2023_kk_against']
X5_cleaned = cleaned_data['_2023_emotion_ofke']
y_cleaned = cleaned_data['_2023_topic_elections_and_voting']

results_cleaned = linear_regression_ols_multiple(np.column_stack((X1_cleaned, X2_cleaned, X3_cleaned, X4_cleaned, X5_cleaned)), np.column_stack((Z1_cleaned, Z2_cleaned, Z3_cleaned)), y_cleaned)

print(results_cleaned.summary())




# In[61]:

#calculating significance levels of coefficients

def print_coefficients_with_stars(results_cleaned):
    stars = ['***', '**', '*']
    for i in range(len(results_cleaned.params)):
        coef = results_cleaned.params[i]
        p_value = results_cleaned.pvalues[i]
        if p_value < 0.001:
            print(f'Coefficient {i}: {coef:.4f} {stars[0]}')
        elif p_value < 0.01:
            print(f'Coefficient {i}: {coef:.4f} {stars[1]}')
        elif p_value < 0.05:
            print(f'Coefficient {i}: {coef:.4f} {stars[2]}')
        else:
            print(f'Coefficient {i}: {coef:.4f}')

print_coefficients_with_stars(results_cleaned)

print(results_cleaned.summary())


# In[62]:
#calculating standardized coefficients

def standardize_coefficients(results):
    coefficients = results.params.values
    std_errors = results.bse.values
    
    standardized_coeffs = coefficients / std_errors
    
    return standardized_coeffs

def print_summary_with_standardized_coeffs(results, y_name):
    standardized_coeffs = standardize_coefficients(results)
    
    summary_table = {
        'Variable': ['Intercept', 'X1', 'X2', 'X3', 'X4','X5', 'Z1', 'Z2'],
        'Coefficient': results.params,
        'Std. Error': results.bse,
        't-value': results.tvalues,
        'Standardized Coefficient': standardized_coeffs
    }

    print(f"Summary for {y_name}")
    print(tabulate(summary_table, headers='keys', tablefmt='pretty'))
    print("\n")

print_summary_with_standardized_coeffs(results_cleaned, 'y_cleaned')


# In[14]:


# Residual analysis for cleaned data
cleaned_residuals = results_cleaned.resid
cleaned_fitted_values = results_cleaned.fittedvalues
plt.figure(figsize=(8, 6))
sns.scatterplot(x=cleaned_fitted_values, y=cleaned_residuals, color='blue', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values (Cleaned Data)')

plt.show()


# In[ ]:




