Requires Python (recommended to use virtual environment e.g., venv / virtualenv) <br/>
Citation of dataset below  <br/>
 <br/>
Training saves a model in /models/ at every epoch to be used as checkpoints, can stop ctrl+c and start training back from any epoch that you choose by  <br/>
 <br/>
### 1. Clone Repository  <br/>
git clone https://github.com/mnothman/imagecolorizer.git  <br/>
 <br/>
 <br/>
### 2. Setup Environment (optional)  <br/>
python -m venv venv  <br/>
source venv/bin/activate  # On Windows: venv\Scripts\activate <br/>
 <br/>
pip install -r requirements.txt  <br/>
 <br/>
### 3. Download Dataset  <br/>
Download dataset from: https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization <br/>
 <br/>
Extract zip to: imagecolorizer/data/raw/archive/  <br/>
 <br/>
Final path should be: imagecolorizer/data/raw/archive/landscapeImages/ <br/>
                                                            ├─ color <br/>
                                                            └─ gray <br/>
 <br/>
Or can also just change the 4 paths, named: gray_path and color_path <br/>
(current hard coded routes of data in project is imagecolorizer/data/raw/archive/landscapeImages) <br/>
 <br/>
 <br/>
### 4. Train Model  <br/>
The training can take a long time for all 50 epochs, so I included a 24MB .keras file in models/ dir if you want to skip this step <br/>
 <br/>
Models save at every checkpoint as well, can resume from checkpoint  <br/>
 <br/>
To start from epoch 1, need to start at 1 if:  <br/>
(a) change model architecture, altering input data (e.g, changing from 256x256 to 128x128), add input channels, switching gray/rgb  <br/>
(b) change in output dimensions (add/removal of output classes for classification), switching classification to regression vice versa <br/>
(c) switching loss function: new loss function or adding new terms  <br/>
(d) dataset changes (adding substantial new data, switching dataset entirely, altering labels) <br/>
(e) changes in optimizer (e.g., Adam to SGD, and adjusting parameters) <br/>
... <br/>
 <br/>
Start fresh at 1: <br/>
python src/train.py <br/>
 <br/>
 <br/>
to start at checkpoint: <br/>
python src/train.py models/colorization_model_epoch_#ofEpochToResumeFrom.keras <br/>
e.g.  <br/>
python src/train.py models/colorization_model_epoch_05.keras <br/>
 <br/>
 <br/>
### 5. Evaluate Model  <br/>
Evaluating the model downloads 10 images in /outputs/predictions/ <br/>
Displays the inputted grayscale image on the left, the middle is the model predictions of the image being colored, while the right image is the actual colored image <br/>
 <br/>
![groundtruth1](https://github.com/user-attachments/assets/9d630a4b-a6c2-4012-888b-15e6f7e4fd9f) <br/> <br/>
 <br/>
![groundtruth2](https://github.com/user-attachments/assets/1e0c30f6-4edf-4988-90e7-e282fa9b5a8c) <br/> <br/>
 <br/>
![groundtruth3](https://github.com/user-attachments/assets/0e098715-0df2-4fbc-a0a4-8d69aa383363) <br/> <br/>
 <br/>
 <br/>
Evaluate.py evaluates the most recent model in models/ directory, unless explicitly states (similar to training) <br/>
 <br/>
Evaluates most recent model (in this example 21):  <br/>
python src/evaluate.py <br/>
 <br/>
Evaluates specified model (in this example 1): <br/>
python src/evaluate.py models/colorization_model_epoch_01.keras <br/>
 <br/>
 <br/>
 <br/>
### 6. Run Project  <br/>
 <br/>
 <br/>
cd api <br/>
python app.py <br/>
 <br/>
 <br/>
 <br/>
testing: <br/>
 <br/>
python tests/generate_mock_data.py <br/>
 <br/>
 <br/>
PYTHONPATH=src python -m unittest tests.test_data_loader <br/>
 <br/>
 <br/>
Kaggle dataset used: https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization <br/>
 <br/>
``` <br/>
imagecolorizer <br/>
├─ README.md <br/>
├─ api <br/>
│  ├─ app.py <br/>
│  ├─ static <br/>
│  └─ templates <br/>
│     └─ app.html <br/>
├─ data <br/>
│  ├─ processed <br/>
│  └─ raw <br/>
│     └─ archive <br/>
│        └─ landscapeImages <br/>
│           ├─ color <br/>
│           └─ gray   <br/>
├─ models <br/>
│  └─ colorization_model.keras <br/>
├─ outputs <br/>
│  ├─ logs <br/>
│  ├─ predictions <br/>
│  └─ visualizations <br/>
├─ requirements.txt <br/>
├─ src <br/>
│  ├─ data_loader.py <br/>
│  ├─ evaluate.py <br/>
│  ├─ model.py <br/>
│  ├─ train.py <br/>
│  └─ utils.py <br/>
├─ tests <br/>
│  ├─ test_data_loader.py <br/>
│  ├─ test_endpoints.py <br/>
│  └─ test_model.py <br/>
└─  <br/>