# 🚨 Crime Hotspot Detection in Los Angeles (2020–2024)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Completed-success" />
</p>

## 📌 Project Overview

Crime Hotspot Detection is a data analysis and machine learning project that identifies high-crime concentration areas in Los Angeles using historical crime data from 2020 to 2024.

The project applies Exploratory Data Analysis (EDA), geographic clustering, and machine learning techniques to understand crime patterns, identify high-volume locations, and present meaningful insights through an interactive Streamlit dashboard.

## 🎯 Objectives

- Analyze historical crime patterns in Los Angeles.
- Identify areas with higher crime concentrations.
- Understand crime trends by year, time, day, and category.
- Apply clustering techniques to geographic crime data.
- Classify geographic grid cells based on historical crime volume.
- Visualize crime patterns using an interactive dashboard.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas & NumPy | Data cleaning and manipulation |
| Matplotlib & Seaborn | Exploratory data visualization |
| Plotly | Interactive charts |
| Scikit-learn | Machine learning |
| DBSCAN | Geographic clustering |
| K-Means | Clustering analysis |
| Streamlit | Interactive web application |
| PyDeck | Geographic visualization |

## 📊 Project Workflow

1. Data Collection
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Geographic Clustering using DBSCAN
6. Grid-Based Crime Hotspot Identification
7. Random Forest Classification
8. Model Evaluation
9. Interactive Dashboard Development using Streamlit

## 🤖 Machine Learning Algorithms

### 1. DBSCAN
Used to identify geographically concentrated groups of crime incidents.

### 2. K-Means Clustering
Used to group crime locations based on geographic coordinates.

### 3. Random Forest Classifier
Used to classify geographic grid cells into high-volume and other categories based on historical crime patterns.

## 📈 Streamlit Dashboard

The interactive dashboard provides:

- Total crime incident statistics.
- Year-wise crime trends.
- Most frequent crime categories.
- Hourly and weekday crime patterns.
- LAPD area-wise crime analysis.
- Geographic crime hotspot visualization.
- Machine learning model performance.
- Feature importance analysis.
- Downloadable aggregated hotspot results.

### 🚀 Application Preview

## 🚀 Application Preview

### 🏠 Dashboard Overview

<p align="center">
  <img src="screenshots/dashboard1.png" alt="Crime Dashboard Overview" width="900"/>
</p>

### 📊 Crime Analysis and Hotspot Visualization

<p align="center">
  <img src="screenshots/dashboard2.png" alt="Crime Analysis Dashboard" width="900"/>
</p>

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
    └── dashboard.png
```

## ⚙️ Installation and Execution

### Step 1: Clone the repository

```bash
git clone https://github.com/shreyashende2005-afk/crime_hotspot_detection.git
```

### Step 2: Navigate to the project directory

```bash
cd crime_hotspot_detection
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit application

```bash
streamlit run app.py
```

### Step 5: Upload the dataset

Use the dataset uploader available in the application to load the required LAPD crime CSV file.

## 📁 Dataset Information

The project uses the Los Angeles Police Department (LAPD) Crime Data from 2020 to 2024.

The dataset contains information such as:
- Date and time of occurrence
- Crime description and category
- LAPD area and area name
- Victim age
- Geographic latitude and longitude

**Note:** The original dataset is not included in this repository. It is kept private and must be supplied locally through the application.

## ⚠️ Limitations

- The analysis is based on historical crime records.
- Identified hotspots represent historical crime concentrations, not guaranteed future crime locations.
- Results depend on dataset quality and completeness.
- The model classifies historical high-volume grid cells; it is not a real-time crime prediction system.

## 🔮 Future Enhancements

- Integrate updated crime datasets.
- Improve hotspot detection techniques.
- Add advanced time-series forecasting.
- Enhance geographic visualizations.
- Develop more detailed crime trend analysis.

## 👩‍💻 Author

**Shreya Shende**

GitHub: [shreyashende2005-afk](https://github.com/shreyashende2005-afk)

---

<p align="center">
  ⭐ If you find this project interesting, consider giving it a star!
</p>
