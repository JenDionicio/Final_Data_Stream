import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


st.sidebar.header("Dashboard")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox('Select Page',['Introduction','Visualization','Prediction'])

df = pd.read_csv("transactions_dataset.csv")
tech_df = df.loc[df['sector'] == 'TECH']



if app_mode == "Introduction":

  st.title("Introduction")
  st.markdown("### Welcome to our ESG rankings Dashboard!")

  #st.image("veh.jpeg", use_column_width=True)

  st.markdown("#### Wondering how ESG rankings truly effect company investment & returns?")
  st.markdown("Our goal is explore investments relative to ESG Rankings & finding/creating a positive feedback loop ")
  st.markdown("##### Objectives")
  st.markdown("- Using other variables that contribute to investment over the years")
  st.markdown("- Points that can be made: ESG growth over the years; correlation w Investment & social pressures")
  st.markdown("- Does ESG ranking positivley or negatively effect investments? ")

  num = st.number_input('No. of Rows', 5, 10)

  head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))
  if head == 'Head':
    st.dataframe(df.head(num))
  else:
    st.dataframe(df.tail(num))

  st.text('(Rows,Columns)')
  st.write(df.shape)

  st.markdown("##### Key Variables")

  st.dataframe(df.describe())

  st.markdown("### Missing Values")
  st.markdown("Null or NaN values.")

  dfnull = df.isnull().sum()/len(df)*100
  totalmiss = dfnull.sum().round(2)
  st.write("Percentage of total missing values:",totalmiss)
  st.write(dfnull)
  if totalmiss <= 30:
    st.success("We have less then 30 percent of missing values, which is good. This provides us with more accurate data as the null values will not significantly affect the outcomes of our conclusions. And no bias will steer towards misleading results. ")
  else:
    st.warning("Poor data quality due to greater than 30 percent of missing value.")
    st.markdown(" > Theoretically, 25 to 30 percent is the maximum missing values are allowed, there's no hard and fast rule to decide this threshold. It can vary from problem to problem.")

  st.markdown("### Completeness")
  st.markdown(" The ratio of non-missing values to total records in dataset and how comprehensive the data is.")

  st.write("Total data length:", len(df))
  nonmissing = (df.notnull().sum().round(2))
  completeness= round(sum(nonmissing)/len(df),2)

  st.write("Completeness ratio:",completeness)
  st.write(nonmissing)
  if completeness >= 0.80:
    st.success("We have completeness ratio greater than 0.85, which is good. It shows that the vast majority of the data is available for us to use and analyze. ")
  else:
    st.success("Poor data quality due to low completeness ratio( less than 0.85).")

elif app_mode == "Visualization":
  st.title("Visualization")
  



  # DATA VISUALISATION
  tab1, tab2, tab3, tab4 = st.tabs(["SNS Plot", "Correlation Map", "Line Chart", "Pie Plot"])

  #SNS plot
  tab1.subheader("SNS plot")
  tech_df = tech_df.sample(n=10000)
  tab1.image('bigger_pairplot.png', use_column_width = True)

  tab1.subheader("Focus Variable Pair Plot")
  tab1.image('smaller_pairplot.png', use_column_width = True)


  # Heat Map
  tab2.subheader("Heat Map")

  # heat map code
  cols = ['ESG_ranking', 'Volatility_Buy',  'Sharpe Ratio', 'inflation','PS_ratio','NetProfitMargin_ratio', 'PB_ratio', 'roa_ratio', 'roe_ratio','EPS_ratio'] # possible essential columns
  corrMatrix = tech_df[cols].corr()
  fig = sns.heatmap(corrMatrix, annot=True, cmap='coolwarm', fmt='.2f')
  
  tab2.write("Heatmap Correlation")
  tab2.pyplot(fig)


  # highRank = tech_df.groupby(tech_df['ESG_ranking']> tech_df['ESG_ranking'].mean() )
  # highRank.get_group(1).describe()
  # highRank.get_group(0).describe()


  # # Line Chart
  # average_volatility = (highRank.get_group(1)['Volatility_Buy'] + highRank.get_group(1)['Volatility_sell']) / 2

  # # Convert the group to a DataFrame to ensure modifications are applied correctly
  # group_df = highRank.get_group(1).copy()
  
  # # Add the calculated average volatility as a new column
  # group_df['Average_Volatility'] = average_volatility
  
  # # Update the original DataFrame with the modified group
  # highRank.groups[1] = group_df

  # sns.lmplot(x='Average_Volatility', y='EPS_ratio', data=highRank.get_group(0))


                  

#   #Bar Graph
#   # User input for x-variable
#   columns = ['Region_Code', 'Gender', 'Vehicle_Age']
#   x_variable = tab2.selectbox("Select x-variable:", columns)
#   tab2.subheader(f"{x_variable} vs Price (INR)")
#   #data_by_variable = df.groupby(x_variable)['Annual_Premium'].mean()
#   #tab2.bar_chart(data_by_variable)

#   #Line Graph
#   tab3.subheader("Age vs Price")
#   #age_by_price = df.groupby('Age')['Annual_Premium'].mean()
#   #tab3.line_chart(age_by_price)

#   '''
#   tab4.subheader("Pie plot")
#   tab4.subheader("Response distribution by Vehicle Damage")
#   response_counts = df.groupby(['Vehicle_Damage', 'Response']).size().unstack(fill_value=0)
#   fig, ax = plt.subplots()
#   colors = ['#ff9999','#66b3ff']
#   damage_counts = response_counts.loc[1]
#   percentages = (damage_counts.values / damage_counts.sum()) * 100
#   labels = ['Yes', 'No']
#   ax.pie(percentages, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
#   ax.axis('equal')
#   tab4.pyplot(fig)

#   #Pie Plot2
#   tab4.subheader("Response Distribution by Not Previously Insured")
#   response_counts = df.groupby(['Previously_Insured', 'Response']).size().unstack(fill_value=0)
#   fig, ax = plt.subplots()
#   colors = ['#ff9999','#66b3ff']
#   prev_insurance_counts = response_counts.loc[0]
#   percentages = (prev_insurance_counts.values / prev_insurance_counts.sum()) * 100
#   labels = ['Yes', 'No']
#   ax.pie(percentages, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
#   ax.axis('equal')
#   tab4.pyplot(fig)


#   tab1, tab2, tab3, tab4 = st.tabs(["SNS Plot", "Bar Chart", "Line Chart", "Pie Plot"])

#   fig = sns.pairplot(df)
#   tab1.pyplot(fig)
#   '''

elif app_mode == "Prediction":
  st.title("Prediction")
  from sklearn.model_selection import train_test_split
  from sklearn.linear_model import LinearRegression
  from sklearn.metrics import mean_squared_error, r2_score
  import pandas as pd
  from sklearn.preprocessing import LabelEncoder
  import seaborn as sns
  import matplotlib.pyplot as plt
  
  # Assuming df is your DataFrame containing all variables
  # df = pd.read_csv("transactions_dataset.csv")
  #variables = df.columns
  cols = ['ESG_ranking', 'Volatility_Buy',  'Sharpe Ratio', 'inflation','PS_ratio','NetProfitMargin_ratio', 'PB_ratio', 'roa_ratio', 'roe_ratio','EPS_ratio'] # possible essential columns
  temp_df = df[cols]
  # Get list of all variable names
  label_encoder = LabelEncoder()
  for name in list(cols):
    temp_df[name] = label_encoder.fit_transform(temp_df[name])
  
  #for target_variable in variables
  # Select the target variable for prediction
  y = temp_df['NetProfitMargin_ratio']

  # Select predictors (all other variables except the target variable)
  X = temp_df.drop(columns=['NetProfitMargin_ratio'])

  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Fit linear regression model
  model = LinearRegression()
  model.fit(X_train, y_train)

  # Make predictions
  y_pred = model.predict(X_test)

  # Calculate accuracy metrics
  st.subheader('Actual vs. Predicted for NetProfitMargin_ratio')
  scatter_plot = sns.scatterplot(x='Actual', y='Predicted', data=results_df)
  scatter_plot.set_title('Actual vs. Predicted for NetProfitMargin_ratio')
  scatter_plot.set_xlabel('Actual')
  scatter_plot.set_ylabel('Predicted')
  sns.regplot(x='Actual', y='Predicted', data=results_df, scatter=False, color='red', ax=scatter_plot)
  st.pyplot()
  
  # Metrics
  mse = mean_squared_error(results_df['Actual'], results_df['Predicted'])
  r2 = r2_score(results_df['Actual'], results_df['Predicted'])
  
  st.write("Prediction of NetProfitMargin_ratio against all other variables:")
  st.write(f"Mean Squared Error: {mse}")
  st.write(f"R-squared: {r2}")


