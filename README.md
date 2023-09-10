

# Machine Learning Approaches to Short-Term Electric Vehicle Charging Demand Prediction: Applicability and Transferability
This is the repo of transfer learning for electric vehicles charging demand prediction, which collects charging demand of charging stations in 12 different cities and tests different prediction models.
## Charging session data
I have sourced and downloaded real-world EV charging session datasets from 12 cities across three continents. All sourced datasets have been stored in the `Data/ORIGINAL_PUBLIC_DATA` folder of this repository. The datasets above were processed to station-level charging demand time series data and further used for model development in this study. All the processed datasets used to reproduce the experiments of this paper have been stored in the `Data/PORCESSED_STATION` folder of this repository.
### China:
- **Hong Kong:** 
  - Source: [Hong Kong Electric EV Charging Data](https://sc.hkelectric.com/TuniS/www.hkelectric.com/zh/smart-power-services/ev-charging-solution/location-map) reserved on Sep 9-10, 2023.
### EU:
- **Perth, Dundee, Crieff, Aberfeldy, Pitlochry, Kinross, Auchterarder, Blairgowrie, Dunkeld:** 
  - Source: [Perth & Kinross EV Charging Data](https://data.pkc.gov.uk/dataset/ev-charging-data) reserved on Sep 9-10, 2023.
  - Source: [Dundee EV Charging Data](https://data.dundeecity.gov.uk/dataset/ev-charging-data) reserved on Sep 9-10, 2023.
### USA:
- **Palo Alto:**
  - Source: [Palo Alto EV Charging Station Usage (July 2011-Dec 2020)](https://data.cityofpaloalto.org/dataviews/257812/electric-vehicle-charging-station-usage-july-2011-dec-2020/) reserved on Sep 9-10, 2023.
- **Boulder:**
  - Source: [Boulder EV Charging Data](https://open-data.bouldercolorado.gov/datasets/39288b03f8d54b39848a2df9f1c5fca2_0/explore) reserved on Sep 9-10, 2023.

## Model Training and Testing
For details on model training and testing, please refer to the notebook [`Transfer.ipynb`](https://github.com/WangSY618/Transfer-learning/blob/main/Transfer.ipynb) 
After executing the training and testing process, results will be saved in the `./res` directory. 
The script `nnpred.py` is utilized for model construction.

### Implemented Machine Learning Models:
- **RF (Random Forest)**: An ensemble learning method that combines multiple decision trees to produce a more accurate prediction.
- **LASSO (Lasso Regression)**: A regression analysis method that performs both variable selection and regularization in order to enhance the prediction accuracy and interpretability of the model.
- **SGD (SGD Regression)**: Stochastic gradient descent regression, an iterative method for optimizing the objective functions commonly found in machine learning.
### Implemented Deep Learning Models:
- **MLP (Multilayer Perceptron)**: A class of feedforward artificial neural network.
- **LSTM (Long Short-Term Memory)**: A type of recurrent neural network capable of remembering past information and is particularly well-suited for sequence prediction problems.
- **GRU (Gated Recurrent Unit)**: Another type of recurrent neural network similar to LSTMs but with a different gating mechanism.

## Visualization of Model Results
### Figures Overview:

- **Fig. 1**: Represents the geographical locations and charging statistics for the 12 study areas.
  - [View Notebook: Figure 1](https://github.com/WangSY618/Transfer-learning/blob/main/Figure/Figure%201.ipynb)
- **Fig. 2**: Illustrates the performance of various models.
  - [View Notebook: Figure 2](https://github.com/WangSY618/Transfer-learning/blob/main/Figure/Figure%202.ipynb)
- **Fig. 3**: Showcases the transferability of different models.
  - [View Notebook: Figure 3](https://github.com/WangSY618/Transfer-learning/blob/main/Figure/Figure%203.ipynb)
## Requirements:

(You can list down the requirements here, if any.)

---

这样的格式应该使内容更加清晰并易于理解。如果有其他需要添加或修改的内容，请告诉我。

##模型结果可视化
###图1，for Fig.1 | 12 Study Areas: Geographical Locations and Charging Statistics. "Figure/Figure1.ipynb"(https://github.com/WangSY618/Transfer-learning/blob/main/Figure/Figure%201.ipynb)
###图2，for Fig.2 | Model Performance."Figure/Figure2.ipynb"(https://github.com/WangSY618/Transfer-learning/blob/main/Figure/Figure%202.ipynb)
###图3，for Fig.3 | Model Transferability "Figure/Figure3.ipynb"(https://github.com/WangSY618/Transfer-learning/blob/main/Figure/Figure%203.ipynb)
## Requirements

```
torch
sklearn
numpy 
pandas
```

### File structure

```
|-- Data    # original data
|   `-- STATION
|       |-- A1-PERTH.csv
|       |-- A10-Aberfeldy.csv
|       |-- A11-Dunkeld.csv
|       |-- A12-Blairgowrie.csv
|       |-- A2-XIANGGANG.csv
|       |-- A3-PALO.csv
|       |-- A4-DUNDEE.csv
|       |-- A5-BOULDER.csv
|       |-- A6-Auchterarder.csv
|       |-- A7-Kinross.csv
|       |-- A8-Pitlochry.csv
|       `-- A9-Crieff.csv
|-- Figure    # model results visualization
|   `-- Figure 1.ipynb   
|   `-- Figure 2.ipynb
|   `-- Figure 3.ipynb
|-- res    # save result
|   `-- station
|       |-- {method}_pred_{pred_len}_{metric}.csv
|       |-- ...
|       `-- ...
|-- station_transfer.ipynb  # main notebook
|-- nnpred.py               # training and testing for neural network models
`-- utils.py                # evaluation utils
```


### Training and testing

Implemented machine learning models:
+ `RF`: Random forest
+ `LASSO`: Lasso regression 
+ `SGD`: SGD regression

Implemented deep learning models:
+ `MLP`: Multilayer perceptron
+ `LSTM`: Long short-term memory
+ `GRU`: Gate recurrent unit

Please refer to `Transfer.ipynb` for training and testing details.
The results will be save in `.\res`.
