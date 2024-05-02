import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
import mlflow
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import graphviz
import missingno as mno



st.sidebar.header("Dashboard")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox('Select Page',['Introduction','Visualization','Prediction'])

df = pd.read_csv("transactions_dataset.csv")
tech_df = df.loc[df['sector'] == 'TECH']


# - - - - - - - - - - - INTRODUCTION - - - - - - - - - - -

if app_mode == "Introduction":

  st.title("Introduction")
  st.markdown("### Welcome to our ESG rankings Dashboard!")

  st.image("ESG_image.png", use_column_width=True)


  st.markdown("## Environmental - Social - Governance")
  st.markdown("##### Does ESG rankings truly effect company investment & returns?")
  
  st.markdown("""
  ##### Objective:
  - Our goal is to explore a companies profit margin ratio relative to ESG Rankings to make a positive feedback loop
  """)
  
  st.markdown("##### Approach:")
  st.markdown("""
  1. Data Exploration
      - Shape, outliers, nulls
  2. Comprehensive Variable Analysis
      - Univariate Analysis
      - Bi-variate analysis
      - Multi-variate analysis
  3. Modelling
      - Build model that solves business problem 
  """)

  # - - - - - - - - - - - - - - - - - -

  st.markdown("<hr>", unsafe_allow_html=True)

  st.markdown("### About the Data Set")
  
  num = st.number_input('How many rows would you like to see?', 5, 10)

  head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))
  if head == 'Head':
    st.dataframe(df.head(num))
  else:
    st.dataframe(df.tail(num))

  st.text(f'This data frame has {df.shape[0]} Rows and {df.shape[1]} columns')

  
  st.markdown("\n\n##### About the Variables")
  st.dataframe(df.describe())

  st.markdown("\n\n### Missing Values")
  st.markdown("Are there any Null or NaN?")

  # Calculate percentage of missing values
  dfnull = tech_df.isnull().sum() / len(tech_df) * 100
  total_miss = dfnull.sum().round(2)
  
  # Display percentage of total missing values
  st.write("Percentage of total missing values:", total_miss, "%")
  
  # Create two columns layout
  col1, col2 = st.columns(2)
  
  # Display DataFrame with missing value percentages in the first column
  with col1:
      st.write("Percentage of Missing Values:")
      st.write(dfnull)
  
  # Display Missing Values Matrix in the second column
  with col2:
      st.write("Missing Values Matrix:")
      fig, ax = plt.subplots(figsize=(20, 6))
      mno.matrix(tech_df, ax=ax)
      st.pyplot(fig)
  
  if total_miss <= 30:
    st.success("This Data set is reliable to use with small amounts of missing values, thus yielding accurate data.")
  else:
    st.warning("Poor data quality due to greater than 30 percent of missing value.")
    st.markdown(" > Theoretically, 25 to 30 percent is the maximum missing values are allowed, there's no hard and fast rule to decide this threshold. It can vary from problem to problem.")

  # st.markdown("<hr>", unsafe_allow_html=True)
  # st.markdown("### Completeness")
  # st.markdown(" The ratio of non-missing values to total records in dataset and how comprehensive the data is.")

  # POSSIBLY DELETE
  # st.write("Total data length:", len(df))
  # nonmissing = (df.notnull().sum().round(2))
  # completeness= round(sum(nonmissing)/len(df),2)
  # st.write("Completeness ratio:",completeness)
  # st.write(nonmissing)
  # if completeness >= 0.80:
  #   st.success("We have completeness ratio greater than 0.85, which is good. It shows that the vast majority of the data is available for us to use and analyze. ")
  # else:
  #   st.success("Poor data quality due to low completeness ratio( less than 0.85).")


# - - - - - - - - - - - VISUALIZATION - - - - - - - - - - -

elif app_mode == "Visualization":
  st.title("Visualization")
  
  # DATA VISUALISATION
  tab1, tab2, tab3, tab4 = st.tabs(["Pair Plots", "Correlation Map", "Line Chart", "Pie Plot"])

  # DF defenition
  tech_df = tech_df.sample(n=10000)

  # - - - - - - - - - - - - - - -  TAB1
  # Define image paths and messages
  image_paths = ['bigger_pairplot.png', 'Annoted_bigger_sns.png', 'smaller_pairplot.png']
  messages = ["All variable pairplot", "Notable Relationships", "Focus Point Variables"]
  
  # Initialize index for the selected image and message
  selected_index = 0
  
  # Define a boolean variable to track whether the button has been clicked
  tab1.write(messages[selected_index])
  tab1.image(image_paths[selected_index], use_column_width=True)
  
  # Add a button to change the image and message
  if tab1.button("Next Image?"):
      # Increment the index to get the next image and message
      selected_index = (selected_index + 1) % len(image_paths)
      # Clear the previous image and message
      tab1.write("")
      # Display the next message
      tab1.write(messages[selected_index])
      # Display the next image
      tab1.image(image_paths[selected_index], use_column_width=True)
  
  # - - - - - - - - - - - - - - TAB 2
  # HEAT MAP
  tab2.title('Heatmap Correlation')
  
  # heat map code
  cols = ['ESG_ranking', 'Volatility_Buy',  'Sharpe Ratio', 'inflation','PS_ratio','NetProfitMargin_ratio', 'PB_ratio', 'roa_ratio', 'roe_ratio','EPS_ratio'] # possible essential columns
  corrMatrix = tech_df[cols].corr()
  
  fig2, ax = plt.subplots()
  sns.heatmap(corrMatrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
  
  # Display the plot within the Streamlit app
  tab2.pyplot(fig2)

  # - - - - - - - - - - - - - - TAB 3
  tab3.title('Differences of ESG Rankings')

  # Grouping based on condition
  high_rank = tech_df.groupby(tech_df['ESG_ranking'] > tech_df['ESG_ranking'].mean())

  # Get the group with ESG_ranking greater than the mean
  high_rank_group = high_rank.get_group(True)

  # Display summary statistics for the group
  tab3.subheader("Summary statistics for high ESG ranking group:")
  tab3.write(high_rank_group.describe())

  # Get the group with ESG_ranking less than or equal to the mean
  low_rank_group = high_rank.get_group(False)

  # Display summary statistics for the group
  tab3.subheader("Summary statistics for low ESG ranking group:")
  tab3.write(low_rank_group.describe())
  # - - - - - - - - - - - - - - TAB 3



# - - - - - - - - - - - PREDICTION - - - - - - - - - - -
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
  results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
  
  # Display the subheader
  st.subheader('Actual vs. Predicted for Net Profit Margin ratio')
  
  # Create a new Matplotlib figure and axis
  fig, ax = plt.subplots()
  
  # Scatter plot
  scatter_plot = sns.scatterplot(x='Actual', y='Predicted', data=results_df, ax=ax)
  scatter_plot.set_title('Actual vs. Predicted for NetProfitMargin_ratio')
  scatter_plot.set_xlabel('Actual')
  scatter_plot.set_ylabel('Predicted')

  # Regression line plot
  sns.regplot(x='Actual', y='Predicted', data=results_df, scatter=False, color='red', ax=ax)
  
  # Display the plot within the Streamlit app
  st.pyplot(fig)

# - - - - - - - - - - - - - - MLFLOW
 
