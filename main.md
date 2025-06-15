# Context
Forecasting foundation models are powerful AI models trained on large, diverse time series data to predict future values across many domains. Like ChatGPT for language, these models can be adapted to different forecasting tasks with little or no additional training. They learn general patterns such as trends and seasonality from massive datasets. This makes them useful in areas like finance, energy, healthcare, and retail. They're helping to make accurate forecasting faster, easier, and more accessible.

Some of the most popular forecasting foundation models include TimeGPT, PatchTST, and TSMixer, known for their strong generalization across time series tasks. Among them, Time-MoE stands out by using a Mixture of Experts approach, where specialized sub-models handle different temporal patterns. This makes Time-MoE especially effective for complex, diverse, or irregular time series data.


# Forecasting Website Views Using Foundation Models

## Objective

Use the forecasting foundation model **Time-MoE** to forecast the daily views of a blogging website using a small dataset. Assess model accuracy, inference speed, and scalability across different forecast horizons.

---

## Dataset

You will be provided with a time series dataset containing:

* **day** the day of interest.
* **target** the total day views the target variable that re. 
* **published** a flag of the days when a new article is published
* **is_holiday** a flag of holidays.

The data is available in the file `data.csv`

---

## Tasks & Instructions

### Task 1: Implement and Fine-Tune a Time-MoE Model

* Preprocess the dataset for multivariate time series forecasting.
* Use the open-source **Time-MoE** model to implement a forecasting model for the given dataset.

**Deliverable**: Clean code that predicts the number of published articles per each day.

---

### Task 2: Optimize Inference Time on CPU

* Focus on deploying the Time-MoE model to run on **CPU only** (no GPU).
* Measure the inference time per iteration for a forecast horizon of 1 day.
* Apply optimizations in order to keep the inference time less than 100 ms/iter.

**Deliverable**: Report showing inference time before and after optimization on a validation set.

---

### Task 3: Evaluate Model Performance

* Propose at least two metrics to evaluate the predictive performance of a forecasting model on the given dataset.


**Deliverable**: Table of metric values + paragraph justifying metric choices.

---

### Task 4: Horizon-Based Comparison

* Compare model predictive performance, and the inference time across different forecast horizons:

  * Short-term: **1 day**
  * Medium-term: **7 days**
  * Long-term: **30 days**

**Deliverable**: Visualizations comparing predictive performance and inference time across horizons.

---

### Task 5: Model Size-Based Comparison

Time-MoE comes in different sizes, the objective is to find out the effect of the size on the predictive performance and the inference time. 

* Compare model predictive performance, and the inference time across different Time-MoE versions:

  * Base
  * Large
  * Ultra

**Deliverable**: Visualizations comparing predictive performance and inference time across model sizes.

---

### Task 6: Streaming Implementation

* What would you do if the data was coming in a stream (one data-point per day), the forecasting model is required to predict the total views the next day (each day). How would your implementation change in this case?

**Deliverable**: Comparative analysis between Time-MoE and TimeGPT on the previous criteria.

---

### Task 7 (Bonus): Forecast Using TimeGPT

* Use a public API or pretrained model (e.g., **Nixtla's TimeGPT**).
* Apply to the same dataset.
* Compare with Time-MoE on:

  * Accuracy
  * Inference time
  * Ease of use

**Deliverable**: Comparative analysis between Time-MoE and TimeGPT on the previous criteria.

