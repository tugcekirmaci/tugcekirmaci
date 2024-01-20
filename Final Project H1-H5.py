#!/usr/bin/env python
# coding: utf-8

# In[251]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate


# In[252]:


file_path = r'C:\Users\user\Desktop\Politus_database_200K_sample.csv'


# In[253]:


data = pd.read_csv(file_path)


# In[236]:


#how individuals' support for political figures (independent variable) predicts their emotional expressions (dependent variable).

#how these emotional expressions (now treated as an independent variable) predict their engagement in discussions related to elections and voting (another dependent variable).




# In[254]:


data['gender']


# In[255]:


gender_mapping = {'female': 0, 'male': 1}
data['gender_numeric'] = data['gender'].map(gender_mapping)
data['gender_numeric']


# In[256]:


data['age_group']


# In[257]:


age_group_map = {'<=18': 1, '19-29': 2, '30-39':3, '>=40': 4}
data['age_group_numeric'] = data['age_group'].map(age_group_map)
data['age_group_numeric']


# In[242]:


columns_for_pairplot = ['gender_numeric', 'age_group_numeric', 'total_tweet_num', '_2023_kk_pro', '_2023_erdogan_pro', '_2023_erdogan_against', '_2023_kk_against', '_2023_emotion_ofke', '_2023_emotion_korku','_2023_emotion_kaygi','_2023_emotion_uzuntu','_2023_emotion_umutsuzluk','_2023_topic_elections_and_voting']
sns.pairplot(data[columns_for_pairplot])
plt.show()

correlation_matrix = data[columns_for_pairplot].corr()
print(correlation_matrix)


# In[263]:


import numpy as np
import pandas as pd
import statsmodels.api as sm

def linear_regression_ols_multiple(X, Z, y):
    # Convert X to a NumPy array and check if it's a vector
    X = np.asarray(X)
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    # Convert Z to a NumPy array and check if it's a vector
    Z = np.asarray(Z)
    if len(Z.shape) == 1:
        Z = Z.reshape(-1, 1)

    # Add a column of ones to X for the intercept term
    X_with_intercept = np.column_stack((np.ones(X.shape[0]), X, Z))

    # Fit the OLS model
    model = sm.OLS(y, X_with_intercept)
    results = model.fit()

    return results


# Example data (assuming you have loaded 'data' DataFrame)
X1 = data['_2023_kk_pro']
X2 = data['_2023_erdogan_pro']
X3 = data['_2023_erdogan_against']
X4 = data['_2023_kk_against']
X5 = data['total_tweet_num']
Z1 = data['gender_numeric']
Z2 = data['age_group_numeric']
y = data['_2023_emotion_ofke']
y1 = data['_2023_emotion_korku']
y2 = data['_2023_emotion_kaygi']
y3 = data['_2023_emotion_uzuntu']
y4 = data['_2023_emotion_umutsuzluk']

# OLS regression
results = linear_regression_ols_multiple(
    np.column_stack((X1, X2, X3, X4, X5)),
    np.column_stack((Z1, Z2)),
    y
)
results1 = linear_regression_ols_multiple(
    np.column_stack((X1, X2, X3, X4, X5)),
    np.column_stack((Z1, Z2)),
    y1
)
results2 = linear_regression_ols_multiple(
    np.column_stack((X1, X2, X3, X4, X5)),
    np.column_stack((Z1, Z2)),
    y2
)
results3 = linear_regression_ols_multiple(
    np.column_stack((X1, X2, X3, X4, X5)),
    np.column_stack((Z1, Z2)),
    y3
)
results4 = linear_regression_ols_multiple(
    np.column_stack((X1, X2, X3, X4, X5)),
    np.column_stack((Z1, Z2)),
    y4
)

# Print coefficients
print("Intercept:", results.params[0])
print("Coefficient for X1:", results.params[1])
print("Coefficient for X2:", results.params[2])
print("Coefficient for X3:", results.params[3])
print("Coefficient for X4:", results.params[4])
print("Coefficient for X5:", results.params[5])
print("Coefficient for Z1:", results.params[6])
print("Coefficient for Z2:", results.params[7])

# Print summary statistics
print(results.summary())
print(results1.summary())
print(results2.summary())
print(results3.summary())
print(results4.summary())


# In[264]:


# Get residuals and fitted values
residuals = results.resid
fitted_values = results.fittedvalues

# Create a scatter plot of residuals against fitted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=fitted_values, y=residuals, color='blue', alpha=0.5)

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='red', linestyle='--')

# Set labels and title
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values')

# Show the plot
plt.show()



# In[265]:


# Extract standardized residuals and leverage values
standardized_residuals = results.get_influence().resid_studentized_internal
leverage_values = results.get_influence().hat_matrix_diag

# Calculate Cook's distance, a measure of influence
cook_distance = standardized_residuals ** 2 * leverage_values / (1 - leverage_values)

# Set a threshold for Cook's distance to identify outliers
threshold = 4 / len(y)  # You can adjust the threshold based on your data size

# Identify outliers based on Cook's distance
outliers_cooks = np.where(cook_distance > threshold)[0]


# In[266]:


# Exclude outliers based on Cook's distance
cleaned_data = data.drop(index=outliers_cooks)

# Example data (assuming you have loaded 'cleaned_data' DataFrame)
X1_cleaned = cleaned_data['_2023_kk_pro']
X2_cleaned = cleaned_data['_2023_erdogan_pro']
X3_cleaned = cleaned_data['_2023_erdogan_against']
X4_cleaned = cleaned_data['_2023_kk_against']
X5_cleaned = cleaned_data['total_tweet_num']
Z1_cleaned = cleaned_data['gender_numeric']
Z2_cleaned = cleaned_data['age_group_numeric']
y_cleaned =  cleaned_data['_2023_emotion_ofke']
y1_cleaned = cleaned_data['_2023_emotion_korku']
y2_cleaned = cleaned_data['_2023_emotion_kaygi']
y3_cleaned = cleaned_data['_2023_emotion_uzuntu']
y4_cleaned = cleaned_data['_2023_emotion_umutsuzluk']

# OLS regression with fixed effects for cleaned data
results_cleaned = linear_regression_ols_multiple(np.column_stack((X1_cleaned, X2_cleaned, X3_cleaned, X4_cleaned, X5_cleaned)), np.column_stack((Z1_cleaned, Z2_cleaned)), y_cleaned)
results1_cleaned = linear_regression_ols_multiple(np.column_stack((X1_cleaned, X2_cleaned, X3_cleaned, X4_cleaned, X5_cleaned)), np.column_stack((Z1_cleaned, Z2_cleaned)), y1_cleaned)
results2_cleaned = linear_regression_ols_multiple(np.column_stack((X1_cleaned, X2_cleaned, X3_cleaned, X4_cleaned, X5_cleaned)), np.column_stack((Z1_cleaned, Z2_cleaned)), y2_cleaned)
results3_cleaned = linear_regression_ols_multiple(np.column_stack((X1_cleaned, X2_cleaned, X3_cleaned, X4_cleaned, X5_cleaned)), np.column_stack((Z1_cleaned, Z2_cleaned)), y3_cleaned)
results4_cleaned = linear_regression_ols_multiple(np.column_stack((X1_cleaned, X2_cleaned, X3_cleaned, X4_cleaned, X5_cleaned)), np.column_stack((Z1_cleaned, Z2_cleaned)), y4_cleaned)



# Print coefficients for cleaned data
print("Intercept (Overall Impact):", combined_results.params[0])
print("Coefficient for X1:", combined_results.params[1])
print("Coefficient for X2:", combined_results.params[2])
print("Coefficient for X3:", combined_results.params[3])
print("Coefficient for X4:", combined_results.params[4])
print("Coefficient for X5:", combined_results.params[5])
print("Coefficient for Z1:", combined_results.params[6])
print("Coefficient for Z2:", combined_results.params[7])

# Print summary statistics for cleaned data
print(results_cleaned.summary())
print(results1_cleaned.summary())
print(results2_cleaned.summary())
print(results3_cleaned.summary())
print(results4_cleaned.summary())


# In[230]:


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

# Print summary statistics for cleaned data
print(results_cleaned.summary())


# In[224]:


def print_coefficients_with_stars(results1_cleaned):
    stars = ['***', '**', '*']
    for i in range(len(results1_cleaned.params)):
        coef = results1_cleaned.params[i]
        p_value = results1_cleaned.pvalues[i]
        if p_value < 0.001:
            print(f'Coefficient {i}: {coef:.4f} {stars[0]}')
        elif p_value < 0.01:
            print(f'Coefficient {i}: {coef:.4f} {stars[1]}')
        elif p_value < 0.05:
            print(f'Coefficient {i}: {coef:.4f} {stars[2]}')
        else:
            print(f'Coefficient {i}: {coef:.4f}')

print_coefficients_with_stars(results1_cleaned)

# Print summary statistics for cleaned data
print(results1_cleaned.summary())


# In[225]:


def print_coefficients_with_stars(results2_cleaned):
    stars = ['***', '**', '*']
    for i in range(len(results2_cleaned.params)):
        coef = results2_cleaned.params[i]
        p_value = results2_cleaned.pvalues[i]
        if p_value < 0.001:
            print(f'Coefficient {i}: {coef:.4f} {stars[0]}')
        elif p_value < 0.01:
            print(f'Coefficient {i}: {coef:.4f} {stars[1]}')
        elif p_value < 0.05:
            print(f'Coefficient {i}: {coef:.4f} {stars[2]}')
        else:
            print(f'Coefficient {i}: {coef:.4f}')

print_coefficients_with_stars(results2_cleaned)

# Print summary statistics for cleaned data
print(results2_cleaned.summary())


# In[226]:


def print_coefficients_with_stars(results3_cleaned):
    stars = ['***', '**', '*']
    for i in range(len(results3_cleaned.params)):
        coef = results3_cleaned.params[i]
        p_value = results3_cleaned.pvalues[i]
        if p_value < 0.001:
            print(f'Coefficient {i}: {coef:.4f} {stars[0]}')
        elif p_value < 0.01:
            print(f'Coefficient {i}: {coef:.4f} {stars[1]}')
        elif p_value < 0.05:
            print(f'Coefficient {i}: {coef:.4f} {stars[2]}')
        else:
            print(f'Coefficient {i}: {coef:.4f}')

print_coefficients_with_stars(results3_cleaned)

# Print summary statistics for cleaned data
print(results3_cleaned.summary())


# In[227]:


def print_coefficients_with_stars(results4_cleaned):
    stars = ['***', '**', '*']
    for i in range(len(results4_cleaned.params)):
        coef = results4_cleaned.params[i]
        p_value = results4_cleaned.pvalues[i]
        if p_value < 0.001:
            print(f'Coefficient {i}: {coef:.4f} {stars[0]}')
        elif p_value < 0.01:
            print(f'Coefficient {i}: {coef:.4f} {stars[1]}')
        elif p_value < 0.05:
            print(f'Coefficient {i}: {coef:.4f} {stars[2]}')
        else:
            print(f'Coefficient {i}: {coef:.4f}')

print_coefficients_with_stars(results4_cleaned)

# Print summary statistics for cleaned data
print(results4_cleaned.summary())


# In[267]:


def standardize_coefficients(results):
    # Extract coefficients and standard errors
    coefficients = results.params.values
    std_errors = results.bse.values
    
    # Standardize coefficients
    standardized_coeffs = coefficients / std_errors
    
    return standardized_coeffs

def print_summary_with_standardized_coeffs(results, y_name):
    # Standardize coefficients
    standardized_coeffs = standardize_coefficients(results)
    
    # Create a summary table
    summary_table = {
        'Variable': ['Intercept', 'X1', 'X2', 'X3', 'X4','X5', 'Z1', 'Z2'],
        'Coefficient': results.params,
        'Std. Error': results.bse,
        't-value': results.tvalues,
        'Standardized Coefficient': standardized_coeffs
    }

    # Print the summary table with the y_name
    print(f"Summary for {y_name}")
    print(tabulate(summary_table, headers='keys', tablefmt='pretty'))
    print("\n")

# Apply the function for each y value
print_summary_with_standardized_coeffs(results_cleaned, 'y_cleaned')
print_summary_with_standardized_coeffs(results1_cleaned, 'y1_cleaned')
print_summary_with_standardized_coeffs(results2_cleaned, 'y2_cleaned')
print_summary_with_standardized_coeffs(results3_cleaned, 'y3_cleaned')
print_summary_with_standardized_coeffs(results4_cleaned, 'y4_cleaned')


# In[ ]:





# In[ ]:





# In[ ]:




