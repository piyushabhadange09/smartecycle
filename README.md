# SmarTECycle ♻️

**SmarTECycle** is an AI-powered smart waste management system that classifies recyclable items from images and manages pickup requests efficiently.  

---

## Features
- **AI Waste Classification:** Upload an image to detect recyclable type.  
- **Pickup Management:** Schedule and track pickup requests.  
- **REST API:** Flask backend with endpoints for inference and pickups.  
- **Frontend Interface:** Simple UI to interact with the system.  

---

## Installation & Setup
```bash
git clone https://github.com/<your-username>/smartecycle.git
cd smartecycle/backend
python -m venv venv
source venv/Scripts/activate   # Windows
pip install -r requirements.txt
python app.py
