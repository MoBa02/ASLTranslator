# Real‑Time Arabic Sign Language Translation

An intelligent, deep learning–based Flask application that translates continuous **Saudi Sign Language (SSL)** from video input into grammatically correct spoken Arabic in real-time.

##  Overview

This project bridges the communication gap by translating sign language directly into spoken and written Arabic. It leverages **MediaPipe** for holistic keypoint extraction (pose + hands) and a **BiLSTM (Bidirectional Long Short-Term Memory)** neural network to recognize sentence-level gestures. The output is further refined by a Large Language Model (LLM) to ensure natural phrasing and is vocalized using Text-to-Speech (TTS).

### Key Features
* **Real-time Translation:** Live inference directly from a webcam feed.
* **Sentence-Level Recognition:** Trained on the **ArabSign dataset**, supporting 50 distinct SSL sentences.
* **Advanced Tracking:** Uses MediaPipe Holistic to track body pose and hand movements simultaneously.
* **Deep Learning:** Custom BiLSTM classifier built with PyTorch.
* **LLM Refinement:** Integration with Groq LLM API to convert raw predictions into smooth, grammatical Arabic text.
* **Text-to-Speech:** Generates spoken Arabic audio using gTTS.

## 🛠️ Tech Stack

* **Backend:** Python, Flask, Flask-SocketIO
* **Computer Vision:** OpenCV, MediaPipe
* **Machine Learning:** PyTorch, NumPy, Pandas
* **APIs & Utilities:** Groq LLM API, gTTS (Google Text-to-Speech)

## 📂 Project Structure

├── app.py              # Main Flask application, routes, and Socket.IO events
├── inference.py        # Data preprocessing, BiLSTM model definition, and prediction logic
├── create_test_videos.py # Utility script for testing
├── ArabSignModel.pth   # Trained model weights
├── 01_test.csv         # Label mappings for the dataset
├── templates/
│   └── index.html      # Frontend Web UI
├── .env                # Environment variables (API keys)
└── .gitignore

## ⚙️ Setup & Installation

### 1\. Clone the Repository

git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
cd your-repo-name

### 2\. Create a Virtual Environment (Recommended)

# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

### 3\. Install Dependencies

pip install flask flask-socketio opencv-python mediapipe numpy pandas torch gtts groq python-dotenv

### 4\. Configure Environment Variables

Create a `.env` file in the project root directory to store your sensitive keys:

GROQ_API_KEY=your_groq_api_key_here
FLASK_SECRET_KEY=your_flask_secret_key_here


##  Running the Application

1.  Start the Flask server:

    python app.py

2.  Open your web browser and navigate to:

    [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 🧠 How It Works

1.  **Input:** Video frames are captured via webcam or video file.
2.  **Extraction:** MediaPipe extracts 3D landmarks for the face, pose, and hands.
3.  **Processing:** Keypoints are normalized and sequenced.
4.  **Prediction:** The PyTorch BiLSTM model processes the sequence to predict the specific sign/sentence.
5.  **Refinement:** The raw prediction strings are sent to the Groq LLM to generate natural Arabic sentences.
6.  **Output:** The final text is displayed on the UI and converted to speech.

## 📚 Acknowledgments & Citations

This project utilizes the **ArabSign** dataset for training and evaluation. We gratefully acknowledge the authors for making this resource available to the research community.

If you use this project or the underlying model, please cite the original ArabSign paper:

> **ArabSign: A Multi-modality Dataset and Benchmark for Continuous Arabic Sign Language Recognition** > *Hamzah Luqman* > *2023 IEEE 17th International Conference on Automatic Face and Gesture Recognition (FG)*

**BibTeX Citation:**
@INPROCEEDINGS{luqmanArabsign2023,
  author={Luqman, Hamzah},
  title={ArabSign: A Multi-modality Dataset and Benchmark for Continuous Arabic Sign Language Recognition},
  booktitle={2023 IEEE 17th International Conference on Automatic Face and Gesture Recognition (FG)},
  year={2023},
  pages={1-8},
  doi={10.1109/FG57933.2023.10042720}
}
## 🤝 Contributing

Contributions are welcome\! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

