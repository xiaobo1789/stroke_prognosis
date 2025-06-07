# ISLES 2022: 多中心脑卒中病变分割数据集  
*A multi-center MRI stroke lesion segmentation dataset for ISLES 2022 challenge*  


## **数据集概览**  
本数据集包含 **400例多中心、多厂商MRI脑卒中病例**，用于急性至亚急性缺血性脑卒中病变分割算法的开发与评估。数据分为：  
- **训练集**：250例（公开可用），包含FLAIR和DWI序列及专家标注掩码；  
- **测试集**：150例（仅用于模型验证，不公开）。  

### **数据内容**  
1. **影像模态**：  
   - FLAIR（液体衰减反转恢复序列）；  
   - DWI（扩散加权成像，含b=1000 trace图及ADC图）。  
2. **文件格式**：  
   - 影像：NIfTI格式（.nii.gz），符合BIDS规范；  
   - 标注：NIfTI格式掩码，像素值`1`代表病变区域。  
3. **关键特征**：  
   - 病变体积范围：0.003-477.264 ml（Table 2）；  
   - 包含后循环缺血、多发栓塞等复杂病例（图1）。  


## **使用许可**  
本数据集采用 **[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)** 许可。使用时需：  
1. 在成果中引用以下文献：  
   ```bibtex
   @misc{hernandez2022isles,
     title={ISLES 2022: A multi-center magnetic resonance imaging stroke lesion segmentation dataset},
     author={Hernandez Petzsche, Moritz Roman and others},
     year={2022},
     eprint={2206.06694},
     archivePrefix={arXiv},
     primaryClass={cs.CV}
   }


### **三、配套文件清单**
| 文件名          | 用途说明                                                                 |  
|-----------------|--------------------------------------------------------------------------|  
| `LICENSE`       | 明确CC BY-SA 4.0许可条款（可从[Creative Commons](https://creativecommons.org/licenses/by-sa/4.0/legalcode)获取） |  
| `.gitattributes`| 配置Git LFS跟踪规则（若存在大文件）：`*.nii.gz filter=lfs diff=lfs merge=lfs -text` |  
| `requirements.txt` | 列出数据读取依赖（如nibabel、numpy）：`nibabel>=3.2.1, numpy>=1.21.0` |  


### **四、上传后验证**  
1. 检查文件完整性：确保训练集文件路径与README描述一致；  
2. 测试下载流程：通过Git或LFS克隆仓库，验证NIfTI文件可正常读取；  
3. 监控配额使用：在GitHub仓库设置中查看LFS存储消耗，避免超量。  

通过以上步骤，可确保数据集合规、高效地上传至GitHub，并便于全球研究者复现和扩展您的工作。
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