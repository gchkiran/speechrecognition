# Speech Recognition with Librispeech Dataset

This project implements a speech recognition system using the Librispeech dataset. It includes modules for data preprocessing, model training, and inference.

## Project Structure

```
speechrecognition/
│
├── demo/
│   ├── demo.py                  # Script to run the demo
│   ├── templates/demo.html      # HTML template for demo UI
│
├── neuralnet/
│   ├── dataset.py               # Dataset preparation and loading
│   ├── model.py                 # Neural network model architecture
│   ├── train.py                 # Model training logic
│   ├── scorer.py                # Evaluation metrics and scoring
│   ├── utils.py                 # Helper functions
│   ├── optimize_graph.py        # Model optimization and graph utilities
│
├── scripts/
│   ├── commonvoice_create_jsons.py   # Script for dataset processing
│   ├── mimic_create_jsons.py         # Another dataset processing utility
│
├── decoder.py                  # Decoding utilities for speech recognition
├── engine.py                   # Core logic for recognition
└── __init__.py                 # Initialization file for the module
```

## Prerequisites

- Python 3.7 or higher
- Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Usage

1. **Dataset Preparation**:
   - Ensure the Librispeech dataset is downloaded and properly organized.
   - Use the scripts in `scripts/` to preprocess and create JSONs for training.

2. **Training the Model**:
   - Run the `train.py` script in the `neuralnet/` directory:
     ```bash
     python neuralnet/train.py
     ```

3. **Running the Demo**:
   - Execute the demo script to test the system with a web-based interface:
     ```bash
     python demo/demo.py
     ```

4. **Model Evaluation**:
   - Use the `scorer.py` script to evaluate performance on the test set:
     ```bash
     python neuralnet/scorer.py
     ```

## Contributing

- Code contributions are welcome. Please ensure that your code is well-documented and follows the project’s coding conventions.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---