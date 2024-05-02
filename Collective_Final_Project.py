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
from sklearn.tree import export_graphviz



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

# - - - - - - - - - - - VISUALIZATION - - - - - - - - - - -

elif app_mode == "Visualization":
  data = {
    'ESG_ranking': tech_df['ESG_ranking'],
    'PS_ratio': tech_df['PS_ratio'],
    'PB_ratio': tech_df['PB_ratio'],
    'roa_ratio': tech_df['roa_ratio'],
  }
  
  df = pd.DataFrame(data)
  
  # Define weights for each metric
  weights = {
      'ESG_ranking': 0.3,
      'PS_ratio': 0.2,
      'PB_ratio': 0.3,
      'roa_ratio': 0.2
  }

  data = {
    'ESG_ranking': tech_df['ESG_ranking'],
    'PS_ratio': tech_df['PS_ratio'],
    'PB_ratio': tech_df['PB_ratio']
  }
  
  df = pd.DataFrame(data)
  
  # Create interaction terms
  tech_df['ESG_PS_interaction'] = tech_df['ESG_ranking'] * tech_df['PS_ratio']
  tech_df['ESG_PB_interaction'] = tech_df['ESG_ranking'] * tech_df['PB_ratio']
  tech_df['PS_PB_interaction'] = tech_df['PS_ratio'] * tech_df['PB_ratio']
  
  
  # Calculate the composite score
  tech_df['Composite_Score'] = sum(df[col] * weights[col] for col in weights)

  cols = ['ESG_ranking', 'Volatility_Buy',  'Sharpe Ratio', 'inflation','PS_ratio','NetProfitMargin_ratio', 'PB_ratio', 'roa_ratio', 'roe_ratio','EPS_ratio','Composite_Score',  'ESG_PS_interaction',  'ESG_PB_interaction',  'PS_PB_interaction' ] 

  
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

# - - - - - - - - - - - - - - DECISION TREE REGRESSOR
  # Define columns
  cols = ['ESG_ranking', 'Volatility_Buy',  'Sharpe Ratio', 'inflation', 'PS_ratio', 'NetProfitMargin_ratio',
          'PB_ratio', 'roa_ratio', 'roe_ratio', 'EPS_ratio']
  
  # Filter dataframe based on selected columns
  temp_df = tech_df[cols]
  
  # Split features and target variable
  X = temp_df.drop(["NetProfitMargin_ratio"], axis=1)
  y = temp_df["NetProfitMargin_ratio"]
  
  # Split dataset into training set and test set
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
  
  # Create Decision Tree Regressor object
  clf = DecisionTreeRegressor(max_depth=3)
  
  # Train Decision Tree Regressor
  clf.fit(X_train, y_train)
  
  # Predict the response for test dataset
  y_pred = clf.predict(X_test)
  
  # Calculate metrics
  mse = metrics.mean_squared_error(y_test, y_pred)
  r2_score = metrics.r2_score(y_test, y_pred)
  
  # Display MSE and R2 score
  st.write(f"MSE: {mse}")
  st.write(f"R2 Score: {r2_score}")
  
  # Plot decision tree
  st.graphviz_chart(export_graphviz(clf, out_file=None, feature_names=X.columns, filled=True, rounded=True))

  # - - - - - - - - - - - - - - - - - PYCARET

  data = {
    'Description': ['Session id', 'Target', 'Target type', 'Original data shape', 'Transformed data shape',
                    'Transformed train set shape', 'Transformed test set shape', 'Numeric features',
                    'Preprocess', 'Imputation type', 'Numeric imputation', 'Categorical imputation',
                    'Transform target', 'Transform target method', 'Fold Generator', 'Fold Number',
                    'CPU Jobs', 'Use GPU', 'Log Experiment', 'Experiment Name', 'USI'],
    'Value': [2557, 'NetProfitMargin_ratio', 'Regression', '(92401, 10)', '(92401, 10)', '(64680, 10)',
              '(27721, 10)', 9, True, 'simple', 'mean', 'mode', True, 'yeo-johnson', 'KFold', 10, -1,
              False, False, 'test1', '08d7']
  }
  
  df = pd.DataFrame(data)

  # Display DataFrame as a table
  st.table(df)

  
  # Create a DataFrame from the given data
  data = {
      'Model': ['knn', 'rf', 'et', 'lightgbm', 'xgboost', 'dt', 'gbr', 'ada', 'br', 'ridge',
                'lr', 'huber', 'en', 'lasso', 'llar', 'par', 'omp', 'dummy', 'lar'],
      'Algorithm': ['K Neighbors Regressor', 'Random Forest Regressor', 'Extra Trees Regressor',
                    'Light Gradient Boosting Machine', 'Extreme Gradient Boosting', 'Decision Tree Regressor',
                    'Gradient Boosting Regressor', 'AdaBoost Regressor', 'Bayesian Ridge', 'Ridge Regression',
                    'Linear Regression', 'Huber Regressor', 'Elastic Net', 'Lasso Regression',
                    'Lasso Least Angle Regression', 'Passive Aggressive Regressor', 'Orthogonal Matching Pursuit',
                    'Dummy Regressor', 'Least Angle Regression'],
      'MAE': [0.0000, 0.0000, 0.0000, 0.0055, 0.0003, 0.0000, 0.2143, 1.2493, 2.2450, 2.2451,
              2.2450, 2.1995, 2.3610, 2.3733, 2.3733, 3.0690, 6.3290, 8.3423, 8.7474],
      'MSE': [0.0000, 0.0000, 0.0000, 0.0002, 0.0000, 0.0000, 0.0777, 2.3647, 7.3785, 7.3784,
              7.3785, 8.0557, 9.1970, 9.4301, 9.4301, 16.9831, 68.2626, 108.6826, 147.4126],
      'RMSE': [0.0000, 0.0000, 0.0000, 0.0125, 0.0007, 0.0000, 0.2785, 1.5376, 2.7163, 2.7163,
               2.7163, 2.8372, 3.0326, 3.0708, 3.0708, 4.0527, 8.2619, 10.4250, 10.9345],
      'R2': [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9993, 0.9782, 0.9319, 0.9319,
             0.9319, 0.9257, 0.9152, 0.9130, 0.9130, 0.8435, 0.3705, -0.0023, -0.3576],
      'RMSLE': [0.0000, 0.0000, 0.0000, 0.0006, 0.0000, 0.0000, 0.0254, 0.1432, 0.2347, 0.2347,
                0.2347, 0.2184, 0.2081, 0.2166, 0.2165, 0.2905, 0.8095, 1.0236, 0.8220],
      'MAPE': [0.0000, 0.0000, 0.0000, 0.0006, 0.0000, 0.0000, 0.0309, 0.3354, 0.4365, 0.4367,
               0.4364, 0.4038, 0.4272, 0.4359, 0.4358, 0.6183, 3.0713, 6.3344, 2.9445],
      'TT (Sec)': [0.3600, 10.7310, 4.6500, 2.2730, 0.5930, 0.2650, 6.7620, 3.1140, 0.1550, 0.1480,
                    0.8520, 1.1060, 0.1560, 0.1560, 0.2480, 0.2530, 0.1470, 0.1440, 0.2080]
  }
  
  df = pd.DataFrame(data)
  
  # Display DataFrame as a table
  st.table(df)

  # - - - - - - - - - - - - - 
  from shapash.explainer.smart_explainer import SmartExplainer
  
  # Assuming `clf`, `X_test`, and `y_pred` are already defined
  xpl = SmartExplainer(clf)
  y_pred = pd.Series(y_pred)
  X_test = X_test.reset_index(drop=True)
  xpl.compile(x=X_test, y_pred=y_pred)
  fig = xpl.plot.features_importance()
  
  # Display the plot directly using Matplotlib's plt.show()
  plt.show(fig)
  st.pyplot(fig)


   
