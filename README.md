# 🚨 Crime Hotspot Detection in Los Angeles (2020–2024)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange" />
</p>

## 📌 Project Overview

Crime Hotspot Detection is a data analysis and machine learning project that identifies high-crime concentration areas in Los Angeles using historical LAPD crime data from 2020 to 2024.

The project uses Exploratory Data Analysis, geographic clustering, and machine learning techniques to understand crime patterns and visualize historical hotspots through an interactive Streamlit dashboard.

## 🎯 Objectives

- Analyze historical crime patterns in Los Angeles.
- Identify areas with higher crime concentrations.
- Study crime trends by year, category, time, and day.
- Apply clustering techniques to geographic crime data.
- Classify geographic grid cells based on historical crime volume.
- Visualize findings using an interactive dashboard.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Plotly
- Scikit-learn
- DBSCAN
- K-Means Clustering
- Random Forest Classifier
- Streamlit
- PyDeck

## 📊 Project Workflow

1. Data Collection
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis
4. Feature Engineering
5. Geographic Clustering using DBSCAN
6. Grid-Based Hotspot Identification
7. Random Forest Classification
8. Model Evaluation
9. Streamlit Dashboard Development

## 🤖 Machine Learning Algorithms

**DBSCAN:** Identifies geographically concentrated groups of crime incidents.

**K-Means:** Groups crime locations based on geographic coordinates.

**Random Forest Classifier:** Classifies geographic grid cells based on historical crime volume.

## 🚀 Application Preview

### Dashboard Overview

<p align="center">
  <img src="screenshots/dashboard1.png" alt="Dashboard Overview" width="900"/>
</p>

### Crime Analysis and Hotspot Visualization

<p align="center">
  <img src="screenshots/dashboard2.png" alt="Crime Analysis Dashboard" width="900"/>
</p>

## ✨ Dashboard Features

- Total crime incident statistics.
- Year-wise crime trends.
- Crime category analysis.
- Hourly and weekday crime patterns.
- LAPD area-wise analysis.
- Geographic hotspot visualization.
- Machine learning model evaluation.
- Feature importance analysis.
- Downloadable aggregated hotspot results.

## 📂 Project Structure

```text
crime_hotspot_detection/
│
├── app.py
├── Crime_Hotspot_Detection_Los_Angeles.ipynb
├── requirements.txt
├── README.md
├── .gitignore
│
└── screenshots/
    ├── dashboard1.png
    └── dashboard2.png
```

## ⚙️ Installation and Execution

**1. Clone the repository**

```bash
git clone https://github.com/shreyashende2005-afk/crime_hotspot_detection.git
```

**2. Navigate to the project directory**

```bash
cd crime_hotspot_detection
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the Streamlit application**

```bash
streamlit run app.py
```

**5. Upload the dataset**

Upload the required LAPD crime CSV file through the application's dataset uploader.

## 📁 Dataset Information

The project uses Los Angeles Police Department (LAPD) Crime Data from 2020 to 2024.

The dataset includes crime dates, crime descriptions, LAPD areas, victim information, and geographic coordinates.

Note: The original dataset is not included in this repository to protect data privacy. It must be supplied separately through the application.

## ⚠️ Limitations

- Analysis is based on historical crime records.
- Hotspots represent historical crime concentrations, not guaranteed future crime locations.
- Results depend on dataset quality and completeness.
- The model is not a real-time crime prediction system.

## 🔮 Future Enhancements

- Integrate updated crime datasets.
- Improve geographic hotspot detection.
- Add time-series forecasting.
- Enhance interactive maps and visualizations.

## 👩‍💻 Author

**Shreya Shende**

GitHub: [shreyashende2005-afk](https://github.com/shreyashende2005-afk)

---

<p align="center">⭐ Thank you for visiting this project!</p>