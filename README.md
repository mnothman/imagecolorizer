![GrayImageDog1](https://github.com/user-attachments/assets/dde3129f-404e-42c7-a926-3654defbd960)

![GrayImageDog2](https://github.com/user-attachments/assets/6943300b-9536-4336-8b5c-1297fdaec6e9)

![ColoredImage1](https://github.com/user-attachments/assets/1a08d90f-f0f5-4d41-94ce-f1820c1bb5e4)

![ColoredImage2](https://github.com/user-attachments/assets/103ce0dd-3722-4cfb-bb1a-06c7b98cc070)

![GrayImage1](https://github.com/user-attachments/assets/5e1127ef-317d-455a-9b50-db199af96dd3)

![GrayImage2](https://github.com/user-attachments/assets/29ace671-a278-47c5-a708-3af86601da17)

Requires Python (recommended to use virtual environment e.g., venv / virtualenv) <br/>
Citation of dataset below  <br/>
 <br/>
Training saves a model in /models/ at every epoch to be used as checkpoints, can stop running training with `ctrl+c` and start training back from any epoch that you choose (read training section for more) <br/>


 <br/>

### 1. Clone Repository  <br/>
```bash
git clone https://github.com/mnothman/imagecolorizer.git  
```
 <br/>

### 2. Setup Environment (optional)  <br/>
Open virtual environment and start it
```bash
python -m venv venv  
```
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate 

```


Install requirements:
```bash
pip install -r requirements.txt  

```
 <br/>

### 3. Download Dataset  <br/>
Download dataset from: https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization <br/>
 <br/>
Extract zip to: imagecolorizer/data/raw/archive/  <br/>
 <br/>
Final path should be: 
imagecolorizer/data/raw/archive/landscapeImages/ <br/>
                                                       -----------------------------------------------------------------------     ├── color/ <br/>
                                           -----------------------------------------------------------------------                 └── gray/  <br/>
 <br/>
 <br/>
Or can also just change the 4 paths with variable names: 'gray_path' and 'color_path' <br/><br/>
Current hard coded routes of data in project is: <br/>
imagecolorizer/data/raw/archive/landscapeImages <br/>
 <br/>


### 4. Train Model  <br/>
The training can take a long time for all 50 epochs to finish training, so I included two different 24MB .keras file in models/ dir if you want to skip this step (included 01 and 21 to test between the 2 different models at different training stages- to test see below in evaluate)<br/>
 <br/>
Models keras files save at every checkpoint as well <br/>
Possible to resume from checkpoint <br/>
 <br/>
To start from epoch 1 (fresh training), need to start at 1 if:  <br/>
(a) change model architecture, altering input data (e.g, changing from 256x256 to 128x128), add input channels, switching gray/rgb  <br/>
(b) change in output dimensions (add/removal of output classes for classification), switching classification to regression vice versa <br/>
(c) switching loss function: new loss function or adding new terms  <br/>
(d) dataset changes (adding substantial new data, switching dataset entirely, altering labels) <br/>
(e) changes in optimizer (e.g., Adam to SGD, and adjusting parameters) <br/>
... <br/>
 <br/>
Start fresh at 1 (will overwrite old keras saves in models/ dir): <br/>
```bash
python src/train.py
```
 <br/>
 <br/>
To start at checkpoint: <br/>

```bash
python src/train.py models/colorization_model_epoch_#ofEpochToResumeFrom.keras
```
Example of usage:
```bash
python src/train.py models/colorization_model_epoch_21.keras
```
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
Evaluate.py evaluates the most recent model in models/ directory (default is 21), unless explicitly stated (similar to training) <br/>
 <br/>
Evaluates most recent model (in this example 21):  <br/>
```bash
python src/evaluate.py 
```
 <br/>
Evaluates specified model (in this example 1): <br/>

```bash
python src/evaluate.py models/colorization_model_epoch_01.keras
```
 <br/>

### 6. Run Project  <br/>

```bash
cd api
python app.py 
```
 <br/>

### 7. Using Project <br/> <br/>

http://127.0.0.1:5000/ <br/><br/>

Use the colorize button to turn a gray image into predicted color using the trained <br/> model, and vice versa use the gray scale button to turn a colored image to gray scale.<br/>

Examples: <br/><br/>

![ColoredImage1](https://github.com/user-attachments/assets/1a08d90f-f0f5-4d41-94ce-f1820c1bb5e4)

![ColoredImage2](https://github.com/user-attachments/assets/103ce0dd-3722-4cfb-bb1a-06c7b98cc070)

![GrayImageDog1](https://github.com/user-attachments/assets/dde3129f-404e-42c7-a926-3654defbd960)

![GrayImageDog2](https://github.com/user-attachments/assets/6943300b-9536-4336-8b5c-1297fdaec6e9)

![GrayImage1](https://github.com/user-attachments/assets/5e1127ef-317d-455a-9b50-db199af96dd3)

![GrayImage2](https://github.com/user-attachments/assets/29ace671-a278-47c5-a708-3af86601da17)


### 8. Testing  <br/>

```bash
python tests/generate_mock_data.py
```
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