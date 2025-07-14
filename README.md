# 🖼️ Image Captioning and Text Generation using CNN, RNN & Django

This project is a web-based **Image Captioning and Text Generation** system built using the **Django framework**. It leverages **Convolutional Neural Networks (CNNs)** to extract visual features from images and **Recurrent Neural Networks (RNNs)** (LSTMs) to generate meaningful textual captions.

---

## 🚀 Features

- Upload an image through the web interface
- CNN extracts features from the image
- RNN (LSTM) generates a descriptive caption
- Real-time caption generation on the frontend
- Built with Django for easy web deployment

---

## 🧠 Technologies Used

- **Frontend**: HTML5, CSS3, Bootstrap (optional)
- **Backend**: Django (Python)
- **Machine Learning**: 
  - CNN (e.g., InceptionV3 or ResNet) for image feature extraction
  - RNN (LSTM) for sequence generation
- **Libraries**:
  - TensorFlow / Keras
  - NumPy, Matplotlib, Pillow
  - NLTK or spaCy for NLP preprocessing

---

## 📂 Dataset

Compatible with:
- [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- [MS COCO](https://cocodataset.org/#home)

---

## 📸 How It Works

1. User uploads an image on the web interface.
2. CNN model extracts visual features.
3. Features are passed to an LSTM model to generate the caption.
4. The caption is displayed back to the user on the page.

---

## 🛠️ Setup Instructions

```bash
git clone https://github.com/your-username/Image-Captioning-Django.git
cd Image-Captioning-Django

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Django server
python manage.py runserver
