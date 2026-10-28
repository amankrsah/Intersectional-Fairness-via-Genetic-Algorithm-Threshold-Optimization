# Intersectional Fairness via Genetic Algorithm Threshold Optimization

A full-stack, real-time machine learning platform designed to audit and mitigate intersectional bias in predictive models.

Automated decision systems in high-stakes domains frequently exhibit measurable disparities across demographic subgroups. Existing fairness mitigation methods typically address a single sensitive attribute in isolation, leaving intersectional subgroups defined by the conjunction of multiple characteristics systematically under-protected. This project implements a post-processing framework that partitions predictions by intersectional demographic membership and uses a real-valued genetic algorithm (GA) to optimize a distinct classification threshold per partition.

---

##  Live Demo

Experience the Intersectional Fairness Engine in action:
**[Launch the Live App](https://stu-attendance.nitrr.ac.in/fairness/)**

---

##  Key Features

* **Model-Agnostic Post-Processing:** The framework is model-agnostic and compatible with any calibrated probability estimator (e.g., Logistic Regression). It requires no retraining and no architectural changes to the base model.
* **Intersectional Optimization:** Simultaneously optimizes a composite fitness function that penalizes deviations from Disparate Impact (DI), Statistical Parity Difference (SPD), Average Odds Difference (AOD), and Equal Opportunity Difference (EOD) across two sensitive attributes.
* **Live Telemetry:** Streams optimization progress, fairness metrics, and fitness penalties in real-time via WebSockets.
* **Local Explainability:** Integrates SHAP and LIME for instance-level auditing.
* **Regulatory Compliance:** Drives Disparate Impact into the legally recognized four-fifths rule compliant band of [0.80, 1.25] across various datasets and attributes, with minor loss of accuracy.

---

##  Mathematical Foundation & Evaluation Metrics

The optimization engine utilizes the DEAP framework to implement the GA. The objective is to find the threshold vector $\theta^*$ that minimizes a composite fairness-accuracy fitness function over the training partition.

### The Composite Fitness Function
The scalar fitness function aggregates three penalty components:

$$
F(\theta) = \lambda_1 \cdot P_{\text{parity}}(\theta) + \lambda_2 \cdot P_{\text{odds}}(\theta) + \lambda_3 \cdot P_{\text{acc}}(\theta)
$$

Where the individual penalties are defined as:

* **Parity Penalty:** $P_{\text{parity}} = |1 - DI_1| + |SPD_1| + |1 - DI_2| + |SPD_2|$
* **Odds Penalty:** $P_{\text{odds}} = |AOD_1| + |EOD_1| + |AOD_2| + |EOD_2|$
* **Accuracy Penalty:** $P_{\text{acc}} = 1 - \text{Accuracy}$

### Evaluation Metrics & Formulas
Assuming $Y \in \{0,1\}$ is the ground-truth label, $\hat{Y} \in \{0,1\}$ is the predicted label, and $S \in \{0,1\}$ is the sensitive attribute ($S=0$: unprivileged group, $S=1$: privileged group):

* **Disparate Impact (DI):** Measures the ratio of favorable outcomes between the unprivileged and privileged groups. Target is 1.0; compliant is $\geq 0.80$.
  * Formula: $DI = P(\hat{Y}=1 | S=0) / P(\hat{Y}=1 | S=1)$
* **Statistical Parity Difference (SPD):** Measures the absolute difference in favorable outcome rates. Target is 0.0.
  * Formula: $SPD = P(\hat{Y}=1 | S=0) - P(\hat{Y}=1 | S=1)$
* **Equal Opportunity Difference (EOD):** Measures the difference in True Positive Rates (TPR). Target is 0.0.
  * Formula: $EOD = TPR(S=0) - TPR(S=1)$
* **Average Odds Difference (AOD):** Measures the average difference in both False Positive Rates (FPR) and True Positive Rates (TPR). Target is 0.0.
  * Formula: $AOD = \frac{1}{2} [(FPR(S=0) - FPR(S=1)) + (TPR(S=0) - TPR(S=1))]$

**Constraint Weights:** The default fitness function utilizes graduated penalty weights ($\lambda_1=1000, \lambda_2=5, \lambda_3=1$). The large weight on $P_{\text{parity}}$ prioritizes four-fifths rule DI compliance, followed by odds equalization, and then accuracy preservation.

---

##  Datasets Used for Validation

The framework was empirically validated across five benchmark datasets spanning finance, criminal justice, and healthcare:

1. **Adult Income Dataset** [🔗 UCI ML Repository](https://archive.ics.uci.edu/dataset/2/adult)
   * **Description:** ~48,842 instances from the 1994 U.S. Census.
   * **Target:** Income >$50K.
   * **Sensitive Attributes:** Sex (Female=0) and Race (non-White=0).
2. **German Credit Dataset** [🔗 UCI ML Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
   * **Description:** 1,000 loan applicants, binary credit risk.
   * **Target:** Credit risk (Good/Bad).
   * **Sensitive Attributes:** Sex (Female=0, inferred from personal status field) and Age (binarized at 25 years, younger=0).
3. **COMPAS Dataset** [🔗 ProPublica GitHub](https://github.com/propublica/compas-analysis)
   * **Description:** ~5,278 Broward County defendants after standard filtering criteria.
   * **Target:** Two-year recidivism.
   * **Sensitive Attributes:** Sex and Race (African-American=0).
4. **Cleveland Heart Disease Dataset** [🔗 UCI ML Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)
   * **Description:** 303 patients; presence of heart disease.
   * **Target:** Presence of heart disease (binarized from a five-level severity score).
   * **Sensitive Attributes:** Sex (Female=0) and Age (binarized at 55 years).
5. **Diabetic Readmission Dataset** [🔗 UCI ML Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
   * **Description:** ~101,766 hospital encounters.
   * **Target:** Readmission within 30 days.
   * **Sensitive Attributes:** Sex and Race (non-Caucasian=0).

---

##  Client-Server Architecture

This application employs a decoupled, asynchronous architecture to handle computationally heavy Genetic Algorithm workloads without blocking the UI.

### 1. Data Ingestion (REST)
* **Endpoint:** `POST /api/analyze-columns`
* **Flow:** The React client mounts a CSV file via `FormData`. FastAPI reads the file into a Pandas DataFrame, stores it in a global dictionary (`STORAGE`) with a unique `file_id`, and returns the column headers.

### 2. Optimization Engine (WebSocket)
* **Endpoint:** `WS /ws/train`
* **Flow:**
    * The client establishes a full-duplex WebSocket connection and sends a JSON configuration payload (selected features, model type, GA parameters).
    * FastAPI trains the base classifier and evaluates initial fairness metrics, emitting a `baseline` payload to the client.
    * The DEAP evolutionary loop begins. At the end of every generation, the server calculates the fitness cost and fairness metrics, yielding execution to the event loop (`asyncio.sleep(0.01)`) to broadcast a `progress` payload.
    * React receives the stream and continuously updates the Recharts instances without full-page re-renders.
    * Upon completion, the server stores the optimized thresholds and emits a `complete` status containing an `export_id`.

### 3. Auditing & Export (REST)
* **Endpoints:** `POST /api/explain-shap`, `POST /api/explain-lime`
* **Flow:** Utilizing the `export_id`, the client requests instance-level explanations. The server instantiates the respective explainer against the cached test set and returns local feature impacts, which React renders as bidirectional bar charts.
* **Endpoints:** `GET /api/download-model`, `GET /api/report-image`
* **Flow:** The client triggers file downloads for the finalized JSON threshold weights and the server-rendered Matplotlib PNG summary.

---

##  Technology Stack

**Frontend (Client)**
* **React:** UI framework.
* **Material-UI (MUI):** Component library for responsive layout and dark/light modes.
* **Recharts:** Rendering live-streaming telemetry and horizontal LIME/SHAP waterfall charts.

**Backend (Server)**
* **FastAPI:** High-performance asynchronous REST and WebSocket API.
* **Scikit-Learn:** Base classifier training and data preprocessing.
* **DEAP:** Genetic Algorithm execution and evolutionary operators.
* **SHAP & LIME:** Model explainability.
* **Pandas & NumPy:** Data manipulation and matrix operations.
* **Matplotlib:** Server-side generation of summary PNG reports.

---

##  Installation & Setup

### Prerequisites
* Node.js v16+
* Python 3.9+

### Backend Setup
Navigate to the backend directory, create a virtual environment, and install dependencies.

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install fastapi uvicorn pandas numpy scikit-learn deap shap lime matplotlib pydantic python-multipart
