# Youtube-influencer-predictor.

## 1. Project Goal

This project aims to build a machine learning model to predict the view count of videos from a curated list of 17 Malayalam tech YouTubers.

2. Files in this Repository

This repository contains all the work for the Week 1 submission:

1. final_5k+_data_collector_(user's_list).py

What it is: The Python script used to collect all the data from the YouTube Data API v3.

How it works: It uses the list of 17 channel IDs provided and automatically fetches video statistics for all videos on those channels. It is designed to be run multiple times, as it will "resume" and skip channels it has already collected to respect the daily API quota.

2. malayalam_youtube_tech_data_final.csv

What it is: The final, raw dataset collected for this project.

Status: Contains 16,806 unique video records from the 17 tech channels.

3. Data_Cleaning_and_Analysis.ipynb

What it is: The Jupyter Notebook for the Week 1 submission.

What it does:

Loads the malayalam_youtube_tech_data_final.csv dataset into a pandas DataFrame.

Performs initial data cleaning (checks for missing values and removes duplicates).

Performs initial data preprocessing (checks data types, gets statistical summaries).

Performs Feature Engineering by creating three new, powerful "ratio" columns (like_ratio, comment_ratio, discussion_ratio) to improve model accuracy.

