import pandas as pd
import numpy as np

# Read in master dataframe
df = pd.read_csv('Master_df.csv', index_col = 0)

# Create a function to find a player's 'prime' years
def prime(name):
    # Create a sub-dataframe
    kershaw = df[df['Name'] == name]
    kershaw.reset_index(drop = True, inplace = True)
    # Create a column to store the percent change of 'performance' compared to previous year
    kershaw['delta'] = kershaw['performance'].pct_change()

    # Empty list to store the years in
    years = []
    for i in range(len(kershaw)):
        # Get the peak year
        if kershaw.loc[i, 'performance'] == kershaw['performance'].max():
            years.append(kershaw.loc[i, 'Age'])
            # Once peak year has been found start a loop to check all following years
            # Check if the subsequent year was > 10% decrease in performance or not, if not then append year to list
            for j in range(1, len(kershaw) - i):
                if kershaw.loc[i+j, 'delta'] > -0.1:
                    years.append(kershaw.loc[i+j, 'Age'])
                # Stop loop once a > 10% drop off in performance has been found
                elif kershaw.loc[i+j, 'delta'] <= -0.1:
                    break
    # Start the sentence string
    s = 'prime years: '
    # Append each year in the list to the sentence string
    for yr in years:
        s = s + str(yr) + ', '
    # Remove the ending comma and space
    s = s.rstrip(', ')
    # Return it as a value
    return s