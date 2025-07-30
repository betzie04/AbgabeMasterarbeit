# On Homotopy of Similar Neural Networks using Non-linear Transformations

Implementation code for the Master Thesis "On Homotopy of Similar Neural Networks using Non-linear Transformations"

This repository contains experimental implementations for analyzing equivalence between language encoders through intrinsic and extrinsic similarity measures, realized using nonlinear neural networks.

## 🚀 Quick Start

### Installation
This project runs on Python 3.10.12. Install dependencies using `requirements.txt`:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### Docker Setup (Optional)
```bash
docker build -t similarity_models:latest .
docker run --gpus "device=1" -d -v ~/code-submission:/similarity_models -v /home/code_submission/data:/similarity_models/data -it similarity_models bash
```

## 📁 Project Structure

```
code-submission/
    ├── get-embeddings.py                          # Embedding generation
    training
    ├── train_alternating_multi.py                 # Alternating training
    ├── train-classifier.py                        # Linear probe training
    ├── train_intrinsic_multi.py                   # Multi-seed intrinsic  
    analysis
    ├── check_orig_train_accuracies.py            # Accuracy validation
    ├── create_comparison_table_lipschitz.py       # Lipschitz constant 
    ├── performance_similarity.py                  # Performance similarity
    setup
    ├── requirements.txt                           # Python dependencies
    ├── Dockerfile                                 # Container setup
```

## 🔬 Experiments Overview

This repository implements four main types of experiments:

**Note**: Configure all output directories in the scripts or use command-line arguments to specify paths.

### 1. Embedding Generation & Preprocessing
- **`get-embeddings.py`**: Generates training embeddings from 25 different MultiBERT models, electra and roberta
- **Output**: Saves embeddings to `data/nonlin/models/{model_name}/` or `data/lin/models/{model_name}/`
- **Path Configuration**: Use `--root_path` argument to specify base directory (default: `data/nonlin/`)

### 2. Similarity Measurements

#### Intrinsic Similarity
- **`train_intrinsic_multi.py`**: Trains intrinsic transformations and computes intrinsic similarity between encoder representations
- **Purpose**: Measures relationships between nonlinear transformed encoder g and h 
- **Output**: Saves trained models to `data/{nonlin|lin}/intrinsic_models{activation}/{model1}_vs_{model2}/`
- **Path Configuration**: Use `--root_path` argument or modify `ROOT_PATH` variable in script
- **Activation Configuration**: Use `--activation` argument to specify activation function (e.g., `_3_ReLu`, `_4_Sigmoid`)

#### Performance Similarity  
- **`train-classifier.py`**: Trains task-specific transformations using neural networks, e.g. \psi(g)
- **`performance_similarity.py`**: Evaluates similarity through downstream task performance and creates agreement plots
- **Output**: 
  - Performance metrics: `data/results/plots/{activation}_HeatMap_Spearman_Accuracy_{task}_subplots.png`
  - Bar plots: `data/results/plots/{activation}_Accuracy_F1_Bar_Plots_clean.png`
  - Results saved to `--output_dir` (default: `data/`)
- **Activation Configuration**: Use `--activation` argument to specify activation function for file naming

### 3. Extrinsic Similarity
- **`train_alternating_multi.py`**: Implements alternating optimization approaches for extrinsic similarity analysis, e.g, train ψ and φ such that sup inf ||ψ(h) - φ(g)||_∞
- **Output**: Saves models and results to `data/nonlin/models{activation}/{model1}_vs_{model2}/`
- **Path Configuration**: Configure using `--root_path` argument
- **Activation Configuration**: Use `--activation` argument to specify activation function for model organization

### 4. Similarity Analysis
- **`check_orig_train_accuracies.py`**: Validates original training accuracies and checks computed values
- **`create_comparison_table_lipschitz.py`**: Analyzes various results including Lipschitz constants, intrinsic vs extrinsic similarity comparisons, heatmaps, and Spearman correlation tables
- **Output**: 
  - Lipschitz analysis: `data/results_max_loss/{task}_intrinsic_lipschitz_ranges.png`
  - Comparison plots: `data/results_max_loss/subplots_intrinsic_vs_extrinsic_by_lr_{task}.png`
  - Filtered constants: `Filtered_lipschitz_constants_task_{task}.png`

## 🛠 Usage Examples

### Generate Embeddings
```bash
# Default path (data/nonlin/)
python get-embeddings.py

# Custom path
python get-embeddings.py --root_path "your/custom/path/"

# Specify cache directory
python get-embeddings.py --cache_dir "data/cache"
```

### Train Neural Network Classifiers
```bash
# Default activation
python train-classifier.py

# Specify activation function
python train-classifier.py --activation "_3_ReLu"
```

### Run Intrinsic Similarity Analysis
```bash
# Non-linear transformations with default activation
python train_intrinsic_multi.py --root_path "data/nonlin/"

# Linear transformations
python train_intrinsic_multi.py --root_path "data/lin/" --linear

# Specify activation function
python train_intrinsic_multi.py --activation "_3_ReLu" --root_path "data/nonlin/"

# Different activation options
python train_intrinsic_multi.py --activation "_4_Sigmoid" --root_path "data/nonlin/"
```

### Analyze Performance Similarity
```bash
# Default output directory
python performance_similarity.py

# Custom output directory
python performance_similarity.py --output_dir "your/results/path/"

# Specify activation function for analysis
python performance_similarity.py --activation "_3_ReLu_save_logits_andProb"
```

### Run Alternating Training
```bash
# Default activation
python train_alternating_multi.py --root_path "data/nonlin/"

# Specify activation function
python train_alternating_multi.py --root_path "data/nonlin/" --activation "_3_ReLu"

# Different activation with linear transformations
python train_alternating_multi.py --root_path "data/lin/" --activation "_4_Sigmoid" --linear
```

### Generate Analysis Reports
```bash
# Create Lipschitz constant analysis and comparison tables
python create_comparison_table_lipschitz.py

# Check training accuracies
python check_orig_train_accuracies.py
```

## 🔧 Path Configuration

### Required Directory Setup
Before running the scripts, ensure the following directories exist or will be created:

1. **Base Data Directory**: `data/` (or your custom path)
2. **Model Cache**: `data/cache/` (for Hugging Face models)
3. **Output Directories**: Created automatically by scripts

### Path Arguments
Most scripts accept these path arguments:
- `--root_path`: Base directory for data storage (default: `data/nonlin/`)
- `--cache_dir`: Directory for cached models (default: `data/cache/`)
- `--output_dir`: Directory for analysis results (default: `data/`)
- `--activation`: Activation function for model naming (e.g., `_3_ReLu`, `_4_Sigmoid`)

### Results Organization
Results are automatically organized into specialized directories:
- **Intrinsic Results**: `data/results/intrinsic/` - JSON files with training metrics and similarity measures
- **Extrinsic Results**: `data/results/extrinsic/` - Performance metrics and comparison data
- **Visualizations**: `data/results/plots/` - Generated plots, heatmaps, and charts
- **Analysis Output**: `data/results_max_loss/` - Lipschitz analysis and max loss comparisons



## 📋 Dependencies

Key dependencies include:
- **PyTorch**: Neural network training and tensor operations
- **Transformers**: Hugging Face model implementations  
- **Datasets**: Data loading and preprocessing
- **Geoopt**: Riemannian optimization
- **Scikit-learn**: Machine learning utilities
- **Matplotlib/Seaborn**: Data visualization
- **NumPy/Pandas**: Numerical computing and data manipulation

See `requirements.txt` for the complete list with specific versions.

## 🎯 Key Features

- **Multi-model Analysis**: Works with 25 different MultiBERT variants
- **Task Diversity**: Supports SST-2 (sentiment) and MRPC (paraphrase) tasks
- **Parallel Processing**: Optimized for multi-core execution
- **Comprehensive Metrics**: Implements multiple similarity measures
- **Reproducible**: Fixed seeds and deterministic operations



## 🔧 Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended for faster training
- **Memory**: At least 16GB RAM for processing large embedding matrices
- **Storage**: Approximately 10GB for generated embeddings and model outputs

## 🔍 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in training scripts
2. **Missing embeddings**: Run `get-embeddings.py` first before other scripts
3. **Import errors**: Ensure all dependencies are installed via `requirements.txt`
4. **Path errors**: Verify that `--root_path` points to the correct directory
5. **Permission errors**: Ensure write permissions for output directories

### Performance Tips

- Utilize parallel processing where available (most scripts support multiprocessing)
- Monitor GPU memory usage during training
- Consider using smaller batch sizes for memory-constrained systems

## 🐳 Docker Setup

### Build and Run Container
```bash
docker build -t similarity_models:latest .
docker run --gpus "device=1" -d -v ~/code-submission:/similarity_models -v /home/code_submission/data:/similarity_models/data -it similarity_models bash
```


## License

[MIT](https://choosealicense.com/licenses/mit/)
