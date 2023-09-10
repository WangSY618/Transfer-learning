

# Machine Learning Approaches to Short-Term Electric Vehicle Charging Demand Prediction: Applicability and Transferability
This is the repo of transfer learning for electric vehicles charging demand prediction, which collects charging demand of charging stations in 12 different cities and tests different prediction models.
## Charging session data
I have sourced and downloaded public, real-world EV charging session datasets from 12 cities spanning three different continents. All datasets have been stored in the `Data/original` folder of this repository. 
### China:
- **Hong Kong:** 
  - Source: [Hong Kong Electric EV Charging Data](https://sc.hkelectric.com/TuniS/www.hkelectric.com/zh/smart-power-services/ev-charging-solution/location-map)
### EU:
- **Perth, Dundee, Crieff, Aberfeldy, Pitlochry, Kinross, Auchterarder, Blairgowrie, Dunkeld:** 
  - Source: [Perth & Kinross EV Charging Data](https://data.pkc.gov.uk/dataset/ev-charging-data)
  - Source: [Dundee EV Charging Data](https://data.dundeecity.gov.uk/dataset/ev-charging-data)
### USA:
- **Palo Alto:**
  - Source: [Palo Alto EV Charging Station Usage (July 2011-Dec 2020)](https://data.cityofpaloalto.org/dataviews/257812/electric-vehicle-charging-station-usage-july-2011-dec-2020/)
- **Boulder:**
  - Source: [Boulder EV Charging Data](https://open-data.bouldercolorado.gov/datasets/39288b03f8d54b39848a2df9f1c5fca2_0/explore)
The datasets above were processed to station-level charging demand time series data and further used for model development in this study. All the processed datasets used to reproduce the experiments of this paper have been stored in the `Data/original` folder of this repository.

### Requirements

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
