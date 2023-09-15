

# Machine Learning Approaches to Short-Term Electric Vehicle Charging Demand Prediction: Applicability and Transferability
This is the repo of transfer learning for electric vehicles charging demand prediction, which collects charging demand of charging stations in 12 different cities and tests different prediction models.
## Charging session data
I have sourced and downloaded real-world EV charging session datasets from 12 cities across three continents. All sourced datasets have been stored in the `Data/ORIGINAL_PUBLIC_DATA` folder of this repository. The datasets above were processed to station-level charging demand time series data and further used for model development in this study. All the processed datasets used to reproduce the experiments of this paper have been stored in the `Data/PORCESSED_STATION` folder of this repository.
### China:
- **Hong Kong:** 
  - Source: [Hong Kong Electric EV Charging Data](https://sc.hkelectric.com/TuniS/www.hkelectric.com/zh/smart-power-services/ev-charging-solution/location-map) Retrieved on September 10, 2023.
### EU:
- **Perth, Dundee, Crieff, Aberfeldy, Pitlochry, Kinross, Auchterarder, Blairgowrie, Dunkeld:** 
  - Source: [Perth & Kinross EV Charging Data](https://data.pkc.gov.uk/dataset/ev-charging-data) Retrieved on September 10, 2023.
  - Source: [Dundee EV Charging Data](https://data.dundeecity.gov.uk/dataset/ev-charging-data) Retrieved on September 10, 2023.
### USA:
- **Palo Alto:**
  - Source: [Palo Alto EV Charging Station Usage (July 2011-Dec 2020)](https://data.cityofpaloalto.org/dataviews/257812/electric-vehicle-charging-station-usage-july-2011-dec-2020/) Retrieved on September 10, 2023.
- **Boulder:**
  - Source: [Boulder EV Charging Data](https://open-data.bouldercolorado.gov/datasets/95992b3938be4622b07f0b05eba95d4c_0/explore) Retrieved on September 10, 2023.

## Model Training and Testing
For details on model training and testing, please refer to the notebook [`models.ipynb`](https://github.com/WangSY618/Transfer-learning/blob/main/Transfer.ipynb) 
The script `nnpred.py` is utilized for model construction.The results of the model can be found [here](./Figure%202/model_results.csv).

### Implemented Machine Learning Models:
- **RF (Random Forest)**: An ensemble learning method that combines multiple decision trees to produce a more accurate prediction.
- **LASSO (Lasso Regression)**: A regression analysis method that performs both variable selection and regularization in order to enhance the prediction accuracy and interpretability of the model.
- **SGD (SGD Regression)**: Stochastic gradient descent regression, an iterative method for optimizing the objective functions commonly found in machine learning.
### Implemented Deep Learning Models:
- **MLP (Multilayer Perceptron)**: A class of feedforward artificial neural network.
- **LSTM (Long Short-Term Memory)**: A type of recurrent neural network capable of remembering past information and is particularly well-suited for sequence prediction problems.
- **GRU (Gated Recurrent Unit)**: Another type of recurrent neural network similar to LSTMs but with a different gating mechanism.
### Requirements:
To successfully run the provided code, ensure the following Python libraries are installed:
- **Data Handling and Operations**: 
  - `pandas`
  - `numpy`
- **Data Preprocessing**:
  - **Library**: `sklearn.preprocessing`
    - **Key Function/Class**: `StandardScaler`
- **Machine Learning/Deep Learning Algorithms**:
  - **Algorithms**:
    - `RandomForestRegressor`
    - `Lasso`
    - `SGDRegressor`
    - `MLPRegressor`
    - `LSTM`
    - `GRU`
  - **Deep Learning Library**:
    - `torch`
    - **Modules**:
      - `torch.nn`
      - `torch.utils.data`
- **Custom Modules and Utilities**: 
  - `nnpredmine` (a custom module)
    - Key functions: `nn_train`, `nn_pred`, `nn_finetune`
  - `utils`(a custom module)
    - Key functions: `MAPE`, `SMAPE`, `RMSE`, `MAE`
Most of these libraries can be installed using `pip`:"
```
pip install pandas numpy scikit-learn torch
```
**Note:** Both `nnpredmine` and `utils` are custom modules essential for the successful execution of the code. You can access and download them directly from the provided GitHub links:
- [`nnpredmine` module](https://github.com/WangSY618/Transfer-learning/blob/main/nnpred.py)
- [`utils` module](https://github.com/WangSY618/Transfer-learning/blob/main/utils.py)

## Visualization of Model Results
### Figures Overview:
- **Fig. 1**: Represents the charging statistics for the 12 study areas.
  - [View Notebook: Figure 1](https://github.com/WangSY618/Transfer-learning/blob/main/Figure/Figure%201/Figure%201.ipynb)
- **Fig. 2**: Illustrates the performance of various models.
  - [View Notebook: Figure 2](https://github.com/WangSY618/Transfer-learning/blob/main/Figure/Figure%202/Figure%202.ipynb)
- **Fig. 3**: Showcases the transferability of different models.
  - [View Notebook: Figure 3](https://github.com/WangSY618/Transfer-learning/blob/main/Figure/Figure%203/Figure%203.ipynb)
## Requirements:
To successfully run the provided code, ensure the following Python libraries are installed:
- **Data Handling and Operations**: 
  - `pandas`
  - `numpy`
- **Data Visualization**: 
  - `matplotlib`
  - `seaborn`
- **Machine Learning and Data Processing**: 
  - `sklearn`
    - Key modules: `KMeans`, `StandardScaler`, `make_blobs`, `metrics`
Install these libraries using `pip`:
```
pip install pandas numpy matplotlib seaborn scikit-learn mpl_toolkits
```
