

# Machine Learning Approaches to Short-Term Electric Vehicle Charging Demand Prediction: Applicability and Transferability

This is the repo of transfer learning for electric vehicles charging demand prediction, which collects charging demand of charging stations in 12 different cities and tests different prediction models.

## Charging session data
We used public real-world EV charging session datasets collected from 12 cities: one in China (i.e., Hong Kong), nine in the EU (i.e., Perth, Dundee, Crieff, Aberfeldy, Pitlochry, Kinross, Auchterarder, Blairgowrie, Dunkeld) and two in the USA (i.e., Palo Alto, and Boulder).<br />
The charging session data of Hong Kong were obtained from https://sc.hkelectric.com/TuniS/www.hkelectric.com/zh/smart-power-services/ev-charging-solution/location-map. <br />
The charging session datasets in Kinross, Dunkeld, Aberfeldy, Perth, Blairgowrie, Pitlochry, Auchterarder, and Crieff were available at https://data.pkc.gov.uk/dataset/ev-charging-data. <br />
The dataset for Dundee can be found at https://data.dundeecity.gov.uk/dataset/ev-charging-data. <br />
The dataset for PALO can be obtained from https://data.cityofpaloalto.org/dataviews/257812/electric-vehicle-charging-station-usage-july-2011-dec-2020/. <br />
The dataset for Boulder can be found at https://open-data.bouldercolorado.gov/datasets/39288b03f8d54b39848a2df9f1c5fca2_0/explore. <br />


### Requirements

```
torch
sklearn
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
