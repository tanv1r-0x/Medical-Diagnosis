##### Chest X-Ray Diagnosis System (Normal vs Pneumonia)

Project structure

medical_image_diagnosis/
│
├── dataset/
│   ├── train/
│   │   ├── NORMAL/*.jpeg
│   │   └── PNEUMONIA/*.jpeg
│   └── val/
|       ├── NORMAL/*.jpeg
│       └── PNEUMONIA/*.jpeg
│
├── model/
│   ├── medical_model.h5     (created after training)
│   └── labels.txt            (created after training, plain text - no JSON)
│
├── train.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md


## How it works
1. **Image upload** — user uploads a chest X-ray via the Streamlit UI
2. **Preprocessing** — image resized to 224x224, normalized with
   MobileNetV2's `preprocess_input`
3. **Model** — MobileNetV2 pretrained on ImageNet (frozen base) +
   a small trainable classification head (Dense → Dropout → sigmoid)
4. **Prediction** — outputs `Normal` or `Pneumonia` with a confidence %
5. **Result screen** — Streamlit shows the image, prediction, and a
   confidence progress bar




pip install -r requirements.txt


python train.py


streamlit run app.py


# Future Scope
- Add more classes (e.g. Normal / Bacterial Pneumonia / Viral Pneumonia)
- Try a different backbone (ResNet50, EfficientNetB0) and compare accuracy
- Add a Grad-CAM heatmap to show which region of the X-ray influenced
  the prediction (great for explainability)
- Fine-tune the last few MobileNetV2 layers instead of keeping the
  whole base frozen, for higher accuracy
- Log every prediction with a timestamp to a CSV file for an audit trail
