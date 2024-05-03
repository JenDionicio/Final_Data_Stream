import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
import io  
import mlflow
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from matplotlib.backends.backend_agg import FigureCanvasAgg
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
  tech_df['Composite_Score'] = sum(tech_df[col] * weights[col] for col in weights)

  cols = ['ESG_ranking', 'Volatility_Buy',  'Sharpe Ratio', 'inflation','PS_ratio','NetProfitMargin_ratio', 'PB_ratio', 'roa_ratio', 'roe_ratio','EPS_ratio','Composite_Score',  'ESG_PS_interaction',  'ESG_PB_interaction',  'PS_PB_interaction' ] 

  # - - - - - - - - - - - - PAIRPLOT
  
  st.title("Visualization")
  
  # DATA VISUALISATION
  tab1, tab2, tab3 = st.tabs(["Pair Plots", "Correlation", "Feature Engineering"])

  # DF defenition
  tech_df = tech_df.sample(n=10000)

  # - - - - - - - - - - - - - - -  TAB1
  image_paths = ['bigger_pairplot.png', 'Annoted_bigger_sns.png', 'smaller_pairplot.png']
  messages = ["#### All variable pairplot", "#### Notable Relationships", "#### Focus Point Variables"]
  
  # Display the initial image and message
  tab1.title("PAIR PLOTS")
  tab1.write(messages[0])
  tab1.image(image_paths[0], use_column_width=True)
  
  button = tab1.button("Next Pair Plot")
  if button:
    tab1.write(messages[1])
    tab1.image(image_paths[1], use_column_width=True)
  button2 = tab1.button('Next Pair Plot ')
  if button2:
    tab1.write(messages[2])
    tab1.image(image_paths[2], use_column_width=True)

  var = tab1.button('Variables')
  if var:
    tab1.markdown("##### 'ESG_ranking', 'Volatility_Buy',  'Sharpe Ratio', 'inflation','PS_ratio','NetProfitMargin_ratio', 'PB_ratio', 'roa_ratio', 'roe_ratio','EPS_ratio'")


  
  # - - - - - - - - - - - - - - TAB 2

  tab2.title('Variable Correlation')
  tab2.markdown("##### ESG_ranking - Volatility_Buy - Sharpe Ratio inflation - PS_ratio NetProfitMargin_ratio PB_ratio - roa_ratio - roe_ratio - EPS_ratio")

  # HEAT MAP
  tab2.markdown('### Heatmap Correlation')
  
  # heat map code
  cols = ['ESG_ranking', 'Volatility_Buy',  'Sharpe Ratio', 'inflation','PS_ratio','NetProfitMargin_ratio', 'PB_ratio', 'roa_ratio', 'roe_ratio','EPS_ratio'] # possible essential columns
  corrMatrix = tech_df[cols].corr()
  
  fig2, ax = plt.subplots()
  sns.heatmap(corrMatrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
  
  # Display the plot within the Streamlit app
  tab2.pyplot(fig2)

  
  # -- DESCRIBE TABLES -- 
  tab2.markdown('Differences of ESG Rankings')

  # Grouping based on condition
  high_rank = tech_df.groupby(tech_df['ESG_ranking'] > tech_df['ESG_ranking'].mean())

  # Get the group with ESG_ranking greater than the mean
  high_rank_group = high_rank.get_group(True)

  # Display summary statistics for the group
  tab2.subheader("Summary statistics for high ESG ranking group:")
  tab2.write(high_rank_group.describe())

  # Get the group with ESG_ranking less than or equal to the mean
  low_rank_group = high_rank.get_group(False)

  # Display summary statistics for the group
  tab2.subheader("Summary statistics for low ESG ranking group:")
  tab2.write(low_rank_group.describe())

  # --  HISTOGRAMS --
  tab2.subheader('Histograms')
  
  # Create subplots
  fig, axes = plt.subplots(2, 2, figsize=(12, 8))
  
  # Plot histograms
  sns.histplot(tech_df['ESG_ranking'], kde=True, ax=axes[0, 0])
  axes[0, 0].set_title('Histogram of ESG Ranking')
  
  sns.histplot(tech_df['PS_ratio'], kde=True, ax=axes[0, 1])
  axes[0, 1].set_title('Histogram of PS Ratio')
  
  sns.histplot(tech_df['PB_ratio'], kde=True, ax=axes[1, 0])
  axes[1, 0].set_title('Histogram of PB Ratio')
  
  sns.histplot(tech_df['roa_ratio'], kde=True, ax=axes[1, 1])
  axes[1, 1].set_title('Histogram of ROA Ratio')
  
  # Adjust layout
  plt.tight_layout()
  
  # Display the plot in Streamlit
  tab2.pyplot(fig)

   # -- BAR PLOTS --
  fig, axes = plt.subplots(1, 4, figsize=(16, 8))
  
  # Plot bar charts
  sns.barplot(x='ESG_ranking', y='Volatility_sell', data=tech_df, ax=axes[0])
  axes[0].set_title('Average stock sell by Group')
  
  sns.barplot(x='ESG_ranking', y='expected_return (yearly)', data=tech_df, ax=axes[1])
  axes[1].set_title('Average returns by Group')
  
  sns.barplot(x='ESG_ranking', y='NetProfitMargin_ratio', data=tech_df, ax=axes[2])
  axes[2].set_title('Average profits by Group')
  
  sns.barplot(x='ESG_ranking', y='Volatility_Buy', data=tech_df, ax=axes[3])  # Swapped 'Volatility_Buy' with 'Volatility_sell'
  axes[3].set_title('Average stock buy by Group')
  
  # Adjust layout
  plt.tight_layout()
  
  # Display the plot in Streamlit
  tab2.pyplot(fig)

    # Bar Charts
  tab2.subheader('Bar Charts')
  
  # Create subplots
  fig, axes = plt.subplots(1, 4, figsize=(12, 6))
  
  # Plot bar charts
  sns.barplot(x='ESG_ranking', y='PS_ratio', data=tech_df, ax=axes[0])
  axes[0].set_title('Average PS Ratio by Group')
  
  sns.barplot(x='ESG_ranking', y='PB_ratio', data=tech_df, ax=axes[1])
  axes[1].set_title('Average PB Ratio by Group')
  
  sns.barplot(x='ESG_ranking', y='roa_ratio', data=tech_df, ax=axes[2])
  axes[2].set_title('Average ROA Ratio by Group')
  
  sns.barplot(x='ESG_ranking', y='Volatility_sell', data=tech_df, ax=axes[3])  # Swapped 'Volatility_Buy' with 'Volatility_sell'
  axes[3].set_title('Average stock sell by Group')
  
  # Adjust layout
  plt.tight_layout()
  
  # Display the plot in Streamlit
  tab2.pyplot(fig)

   # Box Plots
  tab2.subheader('Box Plots')
  
  # Create subplots
  fig, axes = plt.subplots(1, 4, figsize=(12, 6))
  
  # Plot box plots
  sns.boxplot(y='ESG_ranking', data=tech_df, ax=axes[0])
  axes[0].set_title('Box Plot of ESG Ranking')
  
  sns.boxplot(y='PS_ratio', data=tech_df, ax=axes[1])
  axes[1].set_title('Box Plot of PS Ratio')
  
  sns.boxplot(y='PB_ratio', data=tech_df, ax=axes[2])
  axes[2].set_title('Box Plot of PB Ratio')
  
  sns.boxplot(y='roa_ratio', data=tech_df, ax=axes[3])
  axes[3].set_title('Box Plot of ROA Ratio')
  
  # Adjust layout
  plt.tight_layout()
  
  # Display the plot in Streamlit
  tab2.pyplot(fig)
  
  
  # - - - - - - - - - - - - - - TAB 3
  tab3.title('Feature(Data) Engineering')
  tab3.markdown(  """  
  1. ESG Ranking: 
      - This metric reflects a company's ESG performance - better sustainability practices. 
  2.PS Ratio (Price-to-Sales Ratio):
      - This ratio compares a company's market capitalization to its total sales revenue - it indicates how much investors  are willing to pay for each dollar of sales generated by the company.
  3. PB Ratio (Price-to-Book Ratio): 
      - The PB ratio compares a company's market value to its book value - indicating how much investors are willing to pay for each dollar of assets.
  4. ROA Ratio (Return on Assets Ratio): 
      - This ratio measures a company's profitability relative to its total assets - it indicates how efficiently a company is generating profits from its assets. 
      
  Interaction Terms:
  
  - ESG-PS Interaction: 
      - The interaction between ESG ranking and PS ratio captures how a company's sustainability practices may influence its price-to-sales ratio. ESG-PB Interaction: This interaction captures how a company's ESG performance may impact its price-to-book ratio. It helps assess whether sustainability practices influence investors' perceptions of a company's value relative to its assets.
      
      - PS-PB Interaction: This interaction explores the relationship between price-to-sales and price-to-book ratios. It provides insights  into how investors weigh sales revenue and asset value when evaluating a company's stock. 
      - Composite Score:  The composite score combines the weighted contributions of ESG ranking, PS ratio, PB ratio, and possibly other metrics. It offers a holistic assessment of a company's overall performance and sustainability. A higher composite score indicates better overall performance based on the chosen metrics and weights.
      """  )

  # -- new table -- 
  tab3.write(tech_df)
  
# - - - - - - - - - - - PREDICTION - - - - - - - - - - -
elif app_mode == "Prediction":
  st.title("Predictions")
  
  cols = ['ESG_ranking', 'Volatility_Buy',  'Sharpe Ratio', 'inflation','PS_ratio','NetProfitMargin_ratio', 'PB_ratio', 'roa_ratio', 'roe_ratio','EPS_ratio'] # possible essential columns
  temp_df = df[cols]
  # Get list of all variable names
  label_encoder = LabelEncoder()
  for name in list(cols):
    temp_df[name] = label_encoder.fit_transform(temp_df[name])
  
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
  st.subheader('Actual vs. Predicted for Net Profit Margin ratio (Linear Regression)')
  
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
  st.subheader('Decision Tree Regressor')

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
  st.subheader('Pycaret Setup')

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


  st.subheader('Best Models - Pycaret/MLFlow')

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
  st.subheader('Feature Importance')
  st.image('newplot.png')
  


   
