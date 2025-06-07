# Stroke Prognosis Prediction System

This repository implements a multimodal deep learning system for predicting functional outcomes in ischemic stroke patients, based on the research paper:

**"Predicting stroke outcome: A case for multimodal deep learning methods with tabular and CT Perfusion data"**

## Features

- **Multimodal Fusion**: Combines tabular clinical data and CT perfusion imaging
- **DAFT Architecture**: Dynamic Affine Feature Map Transform for effective modality fusion
- **Lightweight Models**: Under 100K parameters for efficient deployment
- **Interpretability**: Feature importance analysis and transformation visualization
- **End-to-End Pipeline**: From data preprocessing to model deployment

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stroke-prognosis.git
   cd stroke-prognosis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

1. Organize your data in the following structure:
   ```
   data/
   ├── train.csv
   ├── val.csv
   ├── test.csv
   └── ct_scans/
       ├── train/
           ├── patient_001.npy
           ├── patient_002.npy
           └── ...
       ├── val/
       └── test/
   ```

2. CSV files should contain:
   - Patient ID
   - Clinical features (see `configs/base_config.py` for required columns)
   - mRS score (0-6)
   - Binary label (0: mRS 0-2, 1: mRS 3-6)

## Training Models

To train a specific model:
```bash
python scripts/train.py --model DAFT
```

Available models:
- `TabNet`: Tabular data only
- `ResNet`: CT perfusion images only
- `DAFT`: Multimodal fusion with DAFT
- `LateFusion`: Late fusion baseline
- `HybridFusion`: Hybrid fusion baseline

## Model Export

Export a trained model for deployment:
```bash
python scripts/export_model.py \
  --model DAFT \
  --checkpoint results/best_model_DAFT.pth \
  --export_dir deployed_models
```

## API Deployment

1. Build Docker image:
   ```bash
   docker build -t stroke-prognosis-api .
   ```

2. Run container:
   ```bash
   docker run -p 8000:8000 -v $(pwd)/models:/app/models stroke-prognosis-api
   ```

3. Use the API:
   ```bash
   curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "tabular_data={\"age\": 65, ...}" \
     -F "ct_scan=@path/to/ct_scan.npy"
   ```

## Results Interpretation

- **Global Feature Importance**: `results/global_importance.png`
- **Sample-specific Importance**: `results/local_importance.png`
- **DAFT Transformation**: `results/daft_transformation.png`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.