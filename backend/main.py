import asyncio
import json
import io
import math
import time
from typing import Dict, Any
import numpy as np
import pandas as pd

# Add matplotlib for backend image generation
import matplotlib
matplotlib.use('Agg') # Use Agg backend to prevent GUI thread crashing on servers
import matplotlib.pyplot as plt

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.websockets import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, average_precision_score
from deap import base, creator, tools, algorithms

# Import Explainability Libraries
import shap
import lime
import lime.lime_tabular

app = FastAPI(title="Intersectional Fairness GA Platform")

# Allow CORS for React frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory storage for active datasets and trained thresholds
STORAGE: Dict[str, Any] = {}

class ExplainRequest(BaseModel):
    instance_index: int = 0  # Which row from the test set to explain

def calc_fairness_metrics(y_true, y_pred, sensitive_features):
    """Calculates DI, SPD, AOD, and EOD."""
    unpriv_mask = (sensitive_features == 0)
    priv_mask = (sensitive_features == 1)
    
    unpriv_pos_rate = y_pred[unpriv_mask].mean() if unpriv_mask.sum() > 0 else 0
    priv_pos_rate = y_pred[priv_mask].mean() if priv_mask.sum() > 0 else 0
    
    spd = unpriv_pos_rate - priv_pos_rate
    di = unpriv_pos_rate / priv_pos_rate if priv_pos_rate > 0 else 0.0
    
    # Calculate True Positive Rate and False Positive Rate for AOD and EOD
    def get_rates(y_t, y_p):
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        if cm.shape == (2, 2): 
            tn, fp, fn, tp = cm.ravel()
        else: 
            tn, fp, fn, tp = 0, 0, 0, 0 
            
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return tpr, fpr
        
    tpr_un, fpr_un = get_rates(y_true[unpriv_mask], y_pred[unpriv_mask])
    tpr_pr, fpr_pr = get_rates(y_true[priv_mask], y_pred[priv_mask])
    
    eod = tpr_un - tpr_pr
    aod = 0.5 * ((fpr_un - fpr_pr) + (tpr_un - tpr_pr))
    
    return di, spd, aod, eod

@app.post("/api/analyze-columns")
async def analyze_columns(file: UploadFile = File(...)):
    """Reads uploaded CSV and returns structural metadata."""
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        file_id = str(time.time_ns())
        
        # Keep dataset in cache briefly for reference
        STORAGE[file_id] = df
        
        return {
            "file_id": file_id,
            "columns": df.columns.tolist(),
            "row_count": len(df)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {str(e)}")


@app.websocket("/ws/train")
async def ws_train(websocket: WebSocket):
    await websocket.accept()
    try:
        # 1. Receive configuration object from client
        config_data = await websocket.receive_text()
        config = json.loads(config_data)
        
        file_id = config.get("file_id")
        target_col = config.get("target_col")
        sa1_col = config.get("sa1_col")
        sa2_col = config.get("sa2_col")
        
        pop_size = int(config.get("pop_size", 100))
        generations = int(config.get("generations", 20))
        cx_prob = float(config.get("cx_prob", 0.7))
        mut_prob = float(config.get("mut_prob", 0.2))
        model_type = config.get("model_type", "logistic_regression")
        
        if file_id not in STORAGE:
            raise ValueError("Dataset context expired or not found. Please re-upload the CSV.")

        df = STORAGE[file_id].copy()

        await websocket.send_json({"status": "log", "message": "Validating dataset integrity and selected columns..."})

        for col in [target_col, sa1_col, sa2_col]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' is missing from the dataset.")

        initial_len = len(df)
        df = df.dropna(subset=[target_col, sa1_col, sa2_col])
        if len(df) < initial_len:
            await websocket.send_json({"status": "log", "message": f"Dropped {initial_len - len(df)} rows containing missing target or protected attributes."})

        if len(df) < 50:
            raise ValueError("Dataset is too small after dropping missing values.")

        def safe_binarize(series):
            if series.nunique() > 2:
                majority_class = series.mode()[0]
                return (series == majority_class).astype(int)
            else:
                return series.astype('category').cat.codes

        df['target'] = safe_binarize(df[target_col])
        df['sa1'] = safe_binarize(df[sa1_col])
        df['sa2'] = safe_binarize(df[sa2_col])
        df['demo_group'] = (df['sa1'] * 2) + df['sa2']
        
        drop_cols = [target_col, sa1_col, sa2_col, 'sa1', 'sa2', 'demo_group']
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [c for c in categorical_cols if c not in drop_cols]
        
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dummy_na=False)
        feature_cols = [col for col in df_encoded.columns if col not in drop_cols]

        X = df_encoded[feature_cols].fillna(0).values 
        y = df_encoded['target'].values
        s1 = df_encoded['sa1'].values
        s2 = df_encoded['sa2'].values
        demo = df_encoded['demo_group'].values
        
        X_train, X_test, y_train, y_test, s1_tr, s1_te, s2_tr, s2_te, demo_tr, demo_te = train_test_split(
            X, y, s1, s2, demo, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        await websocket.send_json({"status": "log", "message": f"Initializing {model_type.replace('_', ' ').title()}..."})
        
        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_type == "decision_tree":
            model = DecisionTreeClassifier(random_state=42)
        else:
            model = LogisticRegression(max_iter=1500, random_state=42, solver='lbfgs')

        model.fit(X_train_scaled, y_train)
        
        train_probs = model.predict_proba(X_train_scaled)[:, 1]
        test_probs = model.predict_proba(X_test_scaled)[:, 1]
        
        y_pred_base = (test_probs >= 0.5).astype(int)
        
        base_acc = accuracy_score(y_test, y_pred_base)
        base_auroc = roc_auc_score(y_test, test_probs)
        base_auprc = average_precision_score(y_test, test_probs)
        
        b_di1, b_spd1, b_aod1, b_eod1 = calc_fairness_metrics(y_test, y_pred_base, s1_te)
        b_di2, b_spd2, b_aod2, b_eod2 = calc_fairness_metrics(y_test, y_pred_base, s2_te)
        
        await websocket.send_json({
            "status": "baseline",
            "metrics": {
                "acc": float(base_acc), "auroc": float(base_auroc), "auprc": float(base_auprc),
                "di1": float(b_di1), "spd1": float(b_spd1), "aod1": float(b_aod1), "eod1": float(b_eod1),
                "di2": float(b_di2), "spd2": float(b_spd2), "aod2": float(b_aod2), "eod2": float(b_eod2)
            }
        })
        
        await websocket.send_json({"status": "log", "message": "Initializing DEAP Genetic Framework..."})
        
        if hasattr(creator, "FitnessMin"): del creator.FitnessMin
        if hasattr(creator, "Individual"): del creator.Individual
            
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
            
        toolbox = base.Toolbox()
        toolbox.register("attr_float", np.random.uniform, 0.001, 0.999)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=4)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def eval_fitness(individual):
            thresholds_array = np.array([individual[g] for g in demo_tr])
            y_pred = (train_probs >= thresholds_array).astype(int)
            acc = accuracy_score(y_train, y_pred)
            di_1, spd_1, _, _ = calc_fairness_metrics(y_train, y_pred, s1_tr)
            di_2, spd_2, _, _ = calc_fairness_metrics(y_train, y_pred, s2_tr)
            
            parity_error = (abs(1.0 - di_1) + abs(spd_1) + abs(1.0 - di_2) + abs(spd_2)) * 1000.0
            accuracy_error = (1.0 - acc) * 1.0
            return (parity_error + accuracy_error,)
            
        toolbox.register("evaluate", eval_fitness)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0.5, sigma=0.2, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        pop = toolbox.population(n=pop_size)
        start_time = time.time()
        
        for gen in range(1, generations + 1):
            offspring = algorithms.varAnd(pop, toolbox, cxpb=cx_prob, mutpb=mut_prob)
            
            for ind in offspring:
                for i in range(len(ind)): ind[i] = max(0.001, min(ind[i], 0.999))
                    
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                
            pop = toolbox.select(offspring, k=len(pop))
            best_ind = tools.selBest(pop, k=1)[0]
            current_min_fitness = best_ind.fitness.values[0]
            
            elapsed = time.time() - start_time
            avg_gen_time = elapsed / gen
            eta_seconds = math.ceil(avg_gen_time * (generations - gen))
            
            test_thresh_arr = np.array([best_ind[g] for g in demo_te])
            y_pred_ga = (test_probs >= test_thresh_arr).astype(int)
            
            ga_acc = accuracy_score(y_test, y_pred_ga)
            ga_di1, ga_spd1, ga_aod1, ga_eod1 = calc_fairness_metrics(y_test, y_pred_ga, s1_te)
            ga_di2, ga_spd2, ga_aod2, ga_eod2 = calc_fairness_metrics(y_test, y_pred_ga, s2_te)
            
            await websocket.send_json({
                "status": "progress", "generation": gen, "total_generations": generations, "eta": f"{eta_seconds}s",
                "message": f"Gen {gen}/{generations} completed. Best Objective Cost: {current_min_fitness:.4f}",
                "metrics": {
                    "acc": float(ga_acc), "auroc": float(base_auroc), "auprc": float(base_auprc),
                    "di1": float(ga_di1), "spd1": float(ga_spd1), "aod1": float(ga_aod1), "eod1": float(ga_eod1),
                    "di2": float(ga_di2), "spd2": float(ga_spd2), "aod2": float(ga_aod2), "eod2": float(ga_eod2),
                    "fitness": float(current_min_fitness)
                }
            })
            await asyncio.sleep(0.01)

        final_best = tools.selBest(pop, k=1)[0]
        
        if hasattr(model, 'coef_'):
            extracted_weights = model.coef_.tolist()
            extracted_intercept = model.intercept_.tolist()
        elif hasattr(model, 'feature_importances_'):
            extracted_weights = model.feature_importances_.tolist()
            extracted_intercept = "N/A (Tree-based Model)"
        else:
            extracted_weights, extracted_intercept = "N/A", "N/A"

        model_payload = {
            "model_architecture": model_type,
            "intercept": extracted_intercept,
            "weights_or_importances": extracted_weights,
            "optimal_thresholds": list(final_best),
            "feature_names": feature_cols
        }
        
        export_id = f"model_{int(time.time())}"
        
        # Store all data needed for LIME and SHAP
        STORAGE[export_id] = {
            "model_payload": model_payload,
            "report_data": {
                "base_acc": base_acc, "ga_acc": ga_acc,
                "b_di1": b_di1, "g_di1": ga_di1,
                "b_di2": b_di2, "g_di2": ga_di2,
                "sa1_col": sa1_col, "sa2_col": sa2_col
            },
            "explain_data": {
                "model": model,
                "X_train": X_train_scaled,
                "X_test": X_test_scaled,
                "feature_names": feature_cols
            }
        }
        
        await websocket.send_json({"status": "complete", "export_id": export_id, "message": "Optimization Engine process finalized successfully."})
        
    except ValueError as ve:
        await websocket.send_json({"status": "error", "message": str(ve)})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"status": "error", "message": f"Critical Engine Error: {str(e)}"})
    finally:
        try: await websocket.close()
        except Exception: pass

# =====================================================================
# SHAP EXPLAINABILITY ENDPOINT
# =====================================================================
@app.post("/api/explain-shap/{export_id}")
async def explain_shap(export_id: str, req: ExplainRequest):
    """Returns global/local SHAP values for a specific instance in the test set."""
    if export_id not in STORAGE:
        raise HTTPException(status_code=404, detail="Model configurations not found or expired.")
    
    data = STORAGE[export_id]["explain_data"]
    model = data["model"]
    X_train = data["X_train"]
    X_test = data["X_test"]
    feature_names = data["feature_names"]
    
    idx = req.instance_index
    if idx < 0 or idx >= len(X_test):
        raise HTTPException(status_code=400, detail="Invalid instance index. Out of bounds.")
    
    # shap.Explainer auto-selects the best algorithm (Tree or Linear).
    # We pass the first 100 training rows as background data to calculate base values quickly.
    explainer = shap.Explainer(model, X_train[:100])
    shap_values = explainer(X_test[idx:idx+1])
    
    # Depending on the model (binary classification), shap might return an array of shape (1, features, 2)
    if len(shap_values.shape) == 3:
        impacts = shap_values.values[0, :, 1].tolist() # Impacts for the positive class
    else:
        impacts = shap_values.values[0].tolist()
        
    return {
        "instance_index": idx,
        "feature_names": feature_names,
        "impacts": impacts
    }

# =====================================================================
# LIME EXPLAINABILITY ENDPOINT
# =====================================================================
@app.post("/api/explain-lime/{export_id}")
async def explain_lime(export_id: str, req: ExplainRequest):
    """Returns local LIME rules for a specific instance in the test set."""
    if export_id not in STORAGE:
        raise HTTPException(status_code=404, detail="Model configurations not found or expired.")
        
    data = STORAGE[export_id]["explain_data"]
    model = data["model"]
    X_train = data["X_train"]
    X_test = data["X_test"]
    feature_names = data["feature_names"]
    
    idx = req.instance_index
    if idx < 0 or idx >= len(X_test):
        raise HTTPException(status_code=400, detail="Invalid instance index. Out of bounds.")
        
    # Initialize the explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=['Outcome 0', 'Outcome 1'],
        discretize_continuous=True
    )
    
    # Generate explanation for the specific row
    exp = explainer.explain_instance(X_test[idx], model.predict_proba, num_features=5)
    
    # Returns a list of tuples: e.g., [('age > 45', -0.15), ('income <= 50k', 0.23)]
    return {
        "instance_index": idx,
        "explanation": exp.as_list() 
    }

@app.get("/api/download-model/{export_id}")
async def download_model(export_id: str):
    if export_id not in STORAGE: raise HTTPException(404, "Not found")
    headers = {'Content-Disposition': f'attachment; filename="{export_id}_optimized_weights.json"'}
    return JSONResponse(content=STORAGE[export_id]["model_payload"], headers=headers)

@app.get("/api/report-image/{export_id}")
async def generate_report_image(export_id: str):
    if export_id not in STORAGE or "report_data" not in STORAGE[export_id]: raise HTTPException(404, "Not found")
    data = STORAGE[export_id]["report_data"]
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ['Accuracy', f'DI ({data["sa1_col"]})', f'DI ({data["sa2_col"]})']
    base_scores = [data["base_acc"], data["b_di1"], data["b_di2"]]
    ga_scores = [data["ga_acc"], data["g_di1"], data["g_di2"]]
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, base_scores, width, label='Baseline Model', color='#555555')
    ax.bar(x + width/2, ga_scores, width, label='GA Optimized Model', color='#2e7d32')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1, label='Perfect Parity (1.0)')
    ax.set_ylabel('Scores'); ax.set_title('Post-Processing Intersectional Fairness Improvements')
    ax.set_xticks(x); ax.set_xticklabels(metrics); ax.legend(); plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0); plt.close(fig)
    return StreamingResponse(buf, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)