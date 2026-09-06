# Crop Disease Classifier 

This is a machine learning project that I made to identify diseases in tomato leaves from images.

The project uses a CNN (Convolutional Neural Network) trained on tomato leaf images. I also made a simple Streamlit web app where an image can be uploaded and the model predicts the disease.

## What it can detect

The model can classify 10 types of tomato leaves:

- Bacterial Spot
- Early Blight
- Healthy
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites Two-spotted Spider Mite
- Target Spot
- Tomato Mosaic Virus
- Tomato Yellow Leaf Curl Virus

## Technologies Used

- Python
- TensorFlow / Keras
- CNN
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Jupyter Notebook

## How I Built It

1. Collected the tomato leaf dataset
2. Created training and validation datasets
3. Resized the images to 224 × 224
4. Applied normalization and data augmentation
5. Built a CNN model
6. Trained the model
7. Evaluated the model using accuracy, classification report and confusion matrix
8. Created a prediction function
9. Built a Streamlit web app

## Model

The CNN contains:

- 3 convolutional layers
- Max pooling layers
- Dense layer
- Dropout
- Softmax output layer

Input image size: **224 × 224**

## Result

The model achieved around **64% validation accuracy** on the current dataset.

The application also shows the top 3 predictions with their confidence values.

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```
## Dataset

I used the Tomato Leaf Disease dataset from Hugging Face.

## Note

This project was made for learning and demonstration purposes. The prediction should not be treated as a professional agricultural diagnosis.

## Future Improvements

- Train with more images
- Improve model accuracy
- Try transfer learning
- Add more crop diseases
- Add treatment information
