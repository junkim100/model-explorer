# Model Explorer

Model Explorer is an interactive CLI tool for exploring the architecture of Hugging Face transformer models. It provides a user-friendly interface to navigate through the layers, blocks, and parameters of complex language models.

## Features

- Interactive exploration of Hugging Face transformer models
- Tree-based visualization of model architecture
- Detailed information about layers, tensors, and parameters
- Support for any Hugging Face model

## Requirements

- Linux-based operating system
- Conda package manager
- CUDA-capable GPU (recommended for large models)

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/model-explorer.git
cd model-explorer
```

2. Create a Conda environment:
```
conda create -n model-explorer python=3.9
conda activate model-explorer
```

3. Install the required packages:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

## Usage

To run the Model Explorer, use the following command:
```python model_explorer.py "model_name"```


Replace `"model_name"` with the Hugging Face model you want to explore. For example:
```python model_explorer.py "meta-llama/Llama-2-7b-hf"```


Navigate through the model using arrow keys and Enter. Press 'q' to quit the application.

## Environment Variables

If you need to use a Hugging Face API token, create a `.env` file in the project root with the following content:
```HF_TOKEN=your_token_here```


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
