# SmarTECycle ♻️

**SmarTECycle** is an AI-powered smart waste management system that classifies recyclable items from images and manages pickup requests efficiently.  

---

## Features
- **AI Waste Classification:** Upload an image to detect recyclable type.  
- **Pickup Management:** Schedule and track pickup requests.  
- **REST API:** Flask backend with endpoints for inference and pickups.  
- **Frontend Interface:** Simple UI to interact with the system.  

---

## API Endpoints
- **POST /infer:** Upload image to classify
- **GET /pickups:** Retrieve pickup requests
- **POST /pickups:** Create new pickup request 

---

## Tech Stack
- **Python 3 | Flask | TensorFlow/Keras | PIL | NumPy**
- **HTML | CSS | JavaScript**

---

## License

- If you want, I can also **add a small section at the top with badges for Python, Flask, and TensorFlow** so it looks more professional on GitHub. Do you want me to do that?


---
## Installation & Setup
```bash
git clone https://github.com/<your-username>/smartecycle.git
cd smartecycle/backend
python -m venv venv
source venv/Scripts/activate   # Windows
pip install -r requirements.txt
python app.py

Backend runs at: http://127.0.0.1:5000

Open frontend: frontend/index.html
