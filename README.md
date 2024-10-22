
# OSRS-GE (Old School RuneScape Grand Exchange)

### Description
OSRS-GE is a data exploration project centered around the Grand Exchange of Old School RuneScape (OSRS). The project focuses on analyzing price and volume data to develop machine learning-driven trading strategies. The long-term vision is to create a comprehensive dashboard that provides market analytics similar to financial investment platforms, helping players make informed trading decisions.

This project showcases my skills in data science, machine learning, and financial analysis, with the ultimate goal of building a tool that optimizes trading strategies in RuneScape's Grand Exchange.

### Features
- **Data Analysis**: Exploration of price and volume data to identify trends and opportunities for profitable trades.
- **Machine Learning**: Developing machine learning models to generate trading insights for items in the Grand Exchange.
- **Future Dashboard**: Plans to develop a dashboard that presents relevant market data, technical analysis, and machine learning-driven trading signals.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/codyjcao/osrs-ge.git
   cd osrs-ge
   ```

2. **Install Python and Jupyter Notebook**:
   Ensure you have Python installed (preferably version 3.7 or later). Install Jupyter Notebook by running:
   ```bash
   pip install jupyterlab
   ```

3. **Install dependencies**:
   The project requires the following Python libraries for statistical computing and machine learning:
   ```bash
   pip install numpy pandas scikit-learn scipy
   ```

4. **Install OSRSBox**:
   Install the OSRSBox library, which provides tools specific to OSRS:
   ```bash
   pip install osrsbox
   ```

5. **Data Files**:
   Price and volume data for various items in the Grand Exchange are stored in CSV format and are regularly updated. These files can be found in the `/data` directory.

### Usage

The project includes a detailed Jupyter Notebook (`osrs_ML_trading_strategy.ipynb`) that focuses on developing a machine learning-based trading strategy for the Grand Exchange. The notebook covers the following steps:

1. **Data Loading**:
   - Item data, such as the "Dragonfire Shield" (item_id = 11284), is loaded from the Grand Exchange dataset. This data includes price and volume information with a time frequency of 6-hour intervals.
   - The notebook computes the **Volume Weighted Average Price (VWAP)**, which adjusts the price based on the traded volume. **Simple Returns (`simpRet_y`)** are calculated as the percentage change in VWAP between two consecutive periods to track short-term price movements.

2. **Feature Engineering**:
   - Technical indicators are computed to inform the trading strategy, including:
     - **RSI (Relative Strength Index)**: Identifies overbought or oversold market conditions.
     - **MACD (Moving Average Convergence Divergence)**: Detects changes in trend strength, direction, and momentum.
     - **Lagged Returns**: Captures the impact of historical performance on future prices.
   - These indicators enhance the model’s ability to predict price movements.

3. **Model Selection**:
   - **Augmented Dickey-Fuller (ADF) Test**: Used to check the stationarity of the return series (`simpRet_y`), ensuring that the model can work with stable data properties over time.
   - **ARIMA (AutoRegressive Integrated Moving Average)**: This time-series model is used to forecast price movements based on past data. The notebook uses an automated method to find the best model parameters.
   - **Linear Regression and Random Forest Classifiers**:
     - **Linear Regression**: Estimates the relationship between price returns and technical indicators (like RSI, MACD, and lagged returns). **Sample weighting** is applied to bias the model toward more recent data, ensuring that it prioritizes recent market trends.
     - **Random Forest Classifiers**: This model is used to classify whether the next price movement will be up or down. **Sample weighting** is similarly applied to emphasize recent data, helping the model predict short-term price movements more accurately.
   - **Sample Weighting**: The sample weighting technique prioritizes recent data over older observations. This ensures that the models focus more on the current market conditions, improving the overall prediction accuracy.

4. **Data Cleaning**:
   - The notebook includes handling for anomalies, such as the RuneScape server downtime on **2024-10-01**, to filter out bad data that could distort the model’s predictions.

By following these steps, the notebook develops a machine learning trading strategy that incorporates historical trends, technical indicators, and recent market conditions.

To run the notebook:
1. Ensure you have followed the installation instructions.
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open and run the notebook `osrs_ML_trading_strategy.ipynb` to explore the data, develop trading insights, and test machine learning models.

### Contributing
Contributions are welcome! If you would like to contribute, feel free to open an issue or submit a pull request.

Future updates will include a more formal contribution guide as the project develops.

### License
(To be decided)

### Future Plans
- Develop a dashboard with detailed visualizations, technical analysis tools, and real-time data.
- Expand machine learning models to cover more advanced trading strategies.
- Move data storage to a database for more efficient management and querying.
