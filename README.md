# Spatio-temporal DeepKriging Pytorch 

Pytorch implementation of Space-Time DeepKriging. 

### 🌐 Applications

Due to size and privacy constraints, full real-world datasets cannot be uploaded to this repository.

#### 📦 Sample Data Provided:

* Precipitation : `datasets/dataset-10DAvg-sample.npy`

The precipitation data is spatially interpolated using the Space-Time DeepKriging (STDK) model.

To generate model for interpolation, run:

```bash
python src/python_scripts/precipitation_interpolation/create_embedding.py
python src/python_scripts/precipitation_interpolation/ST_interpolation.py
```

⚠ Further scripts require the original precipitation dataset, which cannot be shared publicly.

Additional preprocessing utilities:

* `src/python_scripts/precipitation_interpolation/data_preprocessing.py` (Preprocess the given precipitation data)
* `src/python_scripts/precipitation_interpolation/create_data_for_forecasting.py` (Create data for ConvLSTM model)

---
