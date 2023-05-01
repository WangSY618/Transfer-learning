

# transfer learning for EV charging demand prediction

This is the repo of transfer learning for electric vehicles charging demand prediction, which collects charging demand of charging stations in 12 different cities and tests different prediction models.


### Requirements

```
torch
sklearn
pmdarima
numpy 
pandas
```

`pmdarima` can be installed via `pip install pmdarima`

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
+ `MLP`: Multilayer perceptron
+ `LASSO`: Lasso regression 
+ `SGD`: SGD regression

Implemented deep learning models:
+ `LSTM`: Long short-term memory
+ `GRU`: Gate recurrent unit

Please refer to `station_transfer.ipynb` for training and testing details.
The results will be save in `.\res`.