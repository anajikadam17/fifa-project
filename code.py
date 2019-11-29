# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Code starts here

df = pd.read_csv(path)
df = df.loc[:,['Name', 'Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Value', 'Preferred Positions', 'Wage']]

# Code ends here


# --------------
# Code starts here

df['Value (M)'] = df['Value'].replace('[K, â‚¬, M, €]','', regex=True).astype(float)
df['Wage (M)'] = df['Wage'].replace('[K, â‚¬, M, €]','', regex=True).astype(float)

# df.drop(columns=['Unit', 'Unit2'], axis=1)
df['Position'] = df['Preferred Positions'].str.split(' ').str[0]
# Code ends here


# --------------
# Code starts here
import seaborn as sns
df['Position'].value_counts()
sns.countplot(x = 'Position', data = df)
value_distribution_values = df.nlargest(100, ['Wage (M)']) 

overall = df['Overall'].sort_values()
overall_value = df[['Value (M)', 'Overall']].groupby(['Overall'], as_index=False).mean()
df.plot(x='Overall', y='Value (M)')
# Code ends here


# --------------

p_list_1= ['GK', 'LB', 'CB', 'CB', 'RB', 'LM', 'CDM', 'RM', 'LW', 'ST', 'RW']

p_list_2 = ['GK', 'LWB', 'CB', 'RWB', 'LM', 'CDM', 'CAM', 'CM', 'RM', 'LW', 'RW']



    
# p_list_1 stats
df_copy = df.copy()
store = []
for i in p_list_1:
    store.append([i,
                    df_copy.loc[[df_copy[df_copy['Position'] == i]['Overall'].idxmax()]]['Name'].to_string(
                        index=False), df_copy[df_copy['Position'] == i]['Overall'].max()])
df_copy.drop(df_copy[df_copy['Position'] == i]['Overall'].idxmax(), inplace=True)
# return store
df1= pd.DataFrame(np.array(store).reshape(11, 3), columns=['Position', 'Player', 'Overall'])


# p_list_2 stats
df_copy = df.copy()
store = []
for i in p_list_2:
    store.append([i,
                    df_copy.loc[[df_copy[df_copy['Position'] == i]['Overall'].idxmax()]]['Name'].to_string(
                        index=False), df_copy[df_copy['Position'] == i]['Overall'].max()])
df_copy.drop(df_copy[df_copy['Position'] == i]['Overall'].idxmax(), inplace=True)
# return store
df2= pd.DataFrame(np.array(store).reshape(11, 3), columns=['Position', 'Player', 'Overall'])

if df1['Overall'].mean() > df2['Overall'].mean():
        print(df1)
        print(p_list_1)
else:
    print(df2)
    print(p_list_2)
        
    
    
    


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from math import sqrt
from sklearn.model_selection import train_test_split


# Code starts here
X = df.loc[:,['Overall', 'Potential' ,'Wage (M)']]
y = df['Value (M)']
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
# Code ends here


# --------------
from sklearn.preprocessing import PolynomialFeatures

# Code starts here
poly = PolynomialFeatures(3)

X_train_2 = poly.fit_transform(X_train)
model = LinearRegression()

model.fit(X_train_2, y_train)
X_test_2 = poly.fit_transform(X_test)
y_pred_2 = model.predict(X_test_2)
mae = mean_absolute_error(y_test, y_pred_2)
r2 = r2_score(y_test, y_pred_2)

# Code ends here


