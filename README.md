# Spatio-temporal DeepKriging Pytorch 

Pytorch implementation of Space-Time DeepKriging. 

---

## 📦 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/pratiknag/Spatio-temporalDeepKriging-Pytorch.git
cd Spatio-temporalDeepKriging-Pytorch
````

### 2. Create and Activate a Virtual Environment

Ensure `virtualenv` is installed:

```bash
python3 -m pip install virtualenv
```

Create and activate a virtual environment:

```bash
python3 -m virtualenv env
source env/bin/activate  # For Linux/macOS
```

### 3. Install Dependencies

Install the required packages using:

```bash
pip install -r requirements.txt
```


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
