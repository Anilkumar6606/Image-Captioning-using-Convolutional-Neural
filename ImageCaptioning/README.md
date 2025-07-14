# Image Captioning using Convolutional Neural Networks

## Setup Instructions

1. Clone the repository.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Ensure you have the model and tokenizer files in `media/models/`:
   - `model_9.h5`
   - `tokenizer.p`
4. Run migrations:
   ```
   python manage.py migrate
   ```
5. Start the development server:
   ```
   python manage.py runserver
   ```
6. Open your browser at `http://127.0.0.1:8000/`

## Notes
- Only image files up to 5MB are accepted for upload.
- This project is for educational/demo purposes. **Do not use in production as-is!**
- Change the `SECRET_KEY`, set `DEBUG = False`, and configure `ALLOWED_HOSTS` before deploying.
- For best results, use the provided model and tokenizer files. 