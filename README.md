# Women Clothing E-Commerce — Simple Backend + Frontend

This repository now includes a minimal Flask backend and a tiny frontend to explore the dataset interactively.

Files added
- `backend/app.py` — Flask app that loads the CSV in `Data/` and exposes API endpoints.
- `frontend/index.html` — Simple single-page dashboard (Plotly) that fetches the API.
- `run_backend.bat` — Windows helper script to start the backend.

API endpoints
- `GET /api/info` — basic dataset info (rows, columns, column names)
- `GET /api/top-products?n=10` — top N products by review count (default n=10)
- `GET /api/age-distribution` — ages and counts
- `GET /api/rating-by-product?n=20` — average rating and counts per product

Quick start (Windows)
1. Create a virtual environment and activate it (recommended).
2. Install requirements:

```
pip install -r requirements.txt
```

3. Start the backend (from repository root):

```
run_backend.bat
```

4. Open the frontend in a browser:

 - Open `frontend/index.html` in your browser, or visit `http://127.0.0.1:5000/` if you started the Flask server (the server will serve the frontend index automatically).

Notes
- The backend attempts to load dataset from `Data/Womens Clothing E-Commerce Reviews_Dataset.csv`. If that file name differs, `backend/app.py` also tries `Womens Clothing E-Commerce Reviews.csv` (matching notebook naming).
- CORS is enabled so the frontend can be opened directly from file:// or served by the backend.

If you want, I can also:
- Wire the frontend to a nicer UI or add more charts.
- Add unit tests for the Flask endpoints.

# Women-s-E-Commerce-Clothing-Data-Analysis-Project

SDAIA T5 Data Science Bootcamp EDA Project
* [Project Proposal](https://github.com/Mashael2030/Diabetes-Health-Indicators-Classfication/blob/main/Documentation/Project%20Proposal.md)
* [Final Report](https://github.com/Mashael2030/Diabetes-Health-Indicators-Classfication/blob/main/Documentation/Description%20writeup.md)
* [Source Code](https://github.com/Mashael2030/Diabetes-Health-Indicators-Classfication/blob/main/Code/Diabetes_Health_Indicators_Classification_Model_update%20(1).ipynb)
* [Presentation Slides](https://github.com/Mashael2030/Diabetes-Health-Indicators-Classfication/blob/main/Documentation/Diabetes%20presentation.pdf)


 


