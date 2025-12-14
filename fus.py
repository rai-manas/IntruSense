# fusion_minimal.py
import os, glob, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



warnings.filterwarnings("ignore")

# ========= CONFIG =========
MODALITY_FOLDERS = {
    "Network":   r"./Network",
    "Linux":     r"./Linux",
    "Windows":   r"./Windows",
    "Telemetry": r"./Telemetry",
}
RESULTS_DIR = Path("./results_min")
RESULTS_DIR.mkdir(exist_ok=True)
LABEL_CANDIDATES = {"label","labels","type","attack","class","target","isanomaly","anomaly","malicious","benign","outcome","y"}

ROW_CAP = 300_000         # per modality cap; raise/lower if needed
MIN_PER_CLASS = 50        # ensure both classes loaded
TEST_SIZE = 0.30
RANDOM_STATE = 42

# ========= HELPERS =========
def find_label_in_header(cols):
    norm = {c: c.strip().lower().replace(" ", "").replace("\uFEFF","") for c in cols}
    for orig, low in norm.items():
        if low in LABEL_CANDIDATES:
            return orig
    return None

def plot_modality_models_metrics(results_dir, modality, out_dir):
    """
    results_dir / modality / <ModelName> / metrics_summary.csv
    Example: results_dir="results_fusion", modality="Network"
    """
    modality_dir = Path(results_dir) / modality
    rows = []
    for model_name in ["LR", "LinSVM", "RF"]:
        csv_path = modality_dir / model_name / "metrics_summary.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            row = df.iloc[0].to_dict()
            row["model"] = model_name
            rows.append(row)

    if not rows:
        return  # nothing to plot

    df_all = pd.DataFrame(rows)

    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    for metric in metrics:
        plt.figure(figsize=(4, 3))
        sns.barplot(x="model", y=metric, data=df_all)
        plt.ylim(0, 1.0)
        plt.title(f"{modality} – {metric}")
        plt.ylabel(metric)
        plt.xlabel("Model")
        plt.tight_layout()
        out_path = Path(out_dir) / f"{modality}_{metric}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()

def plot_fusion_vs_modalities(results_dir, modalities, out_dir):
    """
    Compare best macro-F1 of each modality vs fusion macro-F1.
    """
    rows = []

    # best per modality
    for modality in modalities:
        modality_dir = Path(results_dir) / modality
        best_f1 = None
        best_model = None
        for model_name in ["LR", "LinSVM", "RF"]:
            csv_path = modality_dir / model_name / "metrics_summary.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                f1 = df["f1_macro"].iloc[0]
                if (best_f1 is None) or (f1 > best_f1):
                    best_f1 = f1
                    best_model = model_name
        if best_f1 is not None:
            rows.append({"name": f"{modality} ({best_model})", "f1_macro": best_f1})

    # fusion row
    fusion_csv = Path(results_dir) / "Fusion" / "metrics_summary.csv"
    if fusion_csv.exists():
        df_f = pd.read_csv(fusion_csv)
        rows.append({"name": "Fusion", "f1_macro": df_f["f1_macro"].iloc[0]})

    if not rows:
        return

    df_plot = pd.DataFrame(rows)

    plt.figure(figsize=(6, 3))
    sns.barplot(x="name", y="f1_macro", data=df_plot)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Macro F1-score")
    plt.title("Best per Modality vs Fusion")
    plt.tight_layout()
    out_path = Path(out_dir) / "fusion_vs_modalities_f1_macro.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


BIN_MAP = {
    "normal":"Benign","benign":"Benign","0":"Benign","False":"Benign","false":"Benign",
    "attack":"Attack","malicious":"Attack","1":"Attack","True":"Attack","true":"Attack","anomaly":"Attack"
}
def unify_label_series(y):
    y = pd.Series(y).astype(str).str.strip()
    y_low = y.str.lower()
    out = y.copy()
    mask = y_low.isin(BIN_MAP.keys())
    out.loc[mask] = y_low.loc[mask].map(BIN_MAP)
    return out

def class_aware_stream(folder, row_cap=ROW_CAP, min_per_class=MIN_PER_CLASS):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs in {folder}")

    # probe first file
    probe = pd.read_csv(files[0], nrows=2000, low_memory=False)
    y_col_probe = find_label_in_header(probe.columns.tolist())
    if y_col_probe is None:
        # heuristic fallback
        small = probe
        cands = [c for c in small.columns if small[c].nunique(dropna=False) <= 10]
        y_col_probe = cands[0] if cands else small.columns[-1]

    total = 0
    parts = []
    counts = {}

    def upd(yser):
        vc = yser.value_counts(dropna=False)
        for k,v in vc.items():
            counts[k] = counts.get(k,0) + int(v)

    chunksize = 100_000
    for f in files:
        hdr = pd.read_csv(f, nrows=0, low_memory=False).columns.tolist()
        y_col = y_col_probe if y_col_probe in hdr else find_label_in_header(hdr)
        if y_col is None:
            # last fallback per file
            try:
                samp = pd.read_csv(f, nrows=5000, low_memory=False)
            except Exception:
                continue
            cands = [c for c in samp.columns if samp[c].nunique(dropna=False) <= 10]
            y_col = cands[0] if cands else None
        if y_col is None:
            continue

        for chunk in pd.read_csv(f, low_memory=False, chunksize=chunksize):
            if y_col not in chunk.columns:
                continue
            chunk = chunk.dropna(axis=1, how="all")
            chunk["__FILE__"] = os.path.basename(f)

            y_chunk = unify_label_series(chunk[y_col])
            X_chunk = chunk.drop(columns=[y_col])

            parts.append(pd.concat([X_chunk, y_chunk.rename("__LBL__")], axis=1))
            upd(y_chunk)
            total += len(chunk)

            have_two = len(counts) >= 2
            both_ok = have_two and all(v >= min_per_class for v in counts.values())
            hit_cap = total >= row_cap
            if both_ok and hit_cap:
                break
        if (len(counts) >= 2 and all(v >= min_per_class for v in counts.values()) and total >= row_cap):
            break

    if not parts:
        raise RuntimeError(f"No rows loaded from {folder}")

    df = pd.concat(parts, ignore_index=True).dropna(axis=1, how="all")

    # separate back
    y = df["__LBL__"].astype(str)
    groups = df["__FILE__"].astype(str) if "__FILE__" in df.columns else pd.Series(["_nofile_"]*len(df))
    X = df.drop(columns=["__LBL__","__FILE__"], errors="ignore")

    # ↓ type handling outside pipeline (simple, robust)
    # try numeric coercion
    for c in X.columns:
        if X[c].dtype.kind in "biufc":
            continue
        # try to coerce; if few numeric parses, leave as object
        coerced = pd.to_numeric(X[c], errors="coerce")
        num_ratio = coerced.notna().mean()
        if num_ratio > 0.9:
            X[c] = coerced
        else:
            X[c] = X[c].astype(str).fillna("__NA__")

    # drop >50%-unique categorical columns to avoid OHE blow-ups
    obj_cols = X.select_dtypes(include=["object"]).columns
    if len(obj_cols):
        n = len(X)
        high_card = [c for c in obj_cols if X[c].nunique(dropna=False)/max(n,1) > 0.50]
        if high_card:
            X = X.drop(columns=high_card)

    print(f"[{os.path.basename(folder)}] class_counts={y.value_counts().to_dict()}  "
          f"rows={len(X)}  num={X.select_dtypes(include=[np.number]).shape[1]}  "
          f"cat={X.select_dtypes(exclude=[np.number]).shape[1]}")
    return X, y, groups

def grouped_split(X, y, groups, test_size=TEST_SIZE, seed=RANDOM_STATE):
    if pd.Series(groups).nunique() <= 1:
        # fall back to random split but keep class presence
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        idx_tr, idx_te = next(sss.split(X, y))
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        idx_tr, idx_te = next(gss.split(X, y, groups))
    return (X.iloc[idx_tr], X.iloc[idx_te], y.iloc[idx_tr], y.iloc[idx_te])

def build_preproc(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=False)),
    ])

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.int8, min_frequency=0.01)
    except TypeError:
        # older sklearn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.int8)

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe),
    ])

    return ColumnTransformer(
        [("num", num_pipe, num_cols),
         ("cat", cat_pipe, cat_cols)],
        sparse_threshold=1.0
    )

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

def train_and_select_models(pre, Xtr, ytr, Xte, yte, outroot: Path, try_rf=True):
    results = {}

    # ---- LR (always) ----
    lr = Pipeline([("pre", pre),
                   ("clf", LogisticRegression(max_iter=200, class_weight="balanced",
                                             solver="saga", n_jobs=-1))])
    outdir = outroot / "LR"; outdir.mkdir(parents=True, exist_ok=True)
    lr.fit(Xtr, ytr)
    yhat = lr.predict(Xte)
    proba = lr.predict_proba(Xte)
    rep = classification_report(yte, yhat, digits=4, output_dict=True)
    cm = confusion_matrix(yte, yhat, labels=np.unique(yte))
    with open(outdir/"report.json","w") as f: json.dump(rep, f, indent=2)
    pd.DataFrame(cm, index=np.unique(yte), columns=np.unique(yte)).to_csv(outdir/"confusion_matrix.csv")
    pd.DataFrame([{
        "accuracy": rep.get("accuracy", None),
        "precision_macro": rep["macro avg"]["precision"],
        "recall_macro": rep["macro avg"]["recall"],
        "f1_macro": rep["macro avg"]["f1-score"],
        "precision_weighted": rep["weighted avg"]["precision"],
        "recall_weighted": rep["weighted avg"]["recall"],
        "f1_weighted": rep["weighted avg"]["f1-score"],
    }]).to_csv(outdir/"metrics_summary.csv", index=False)
    np.save(outdir/"proba.npy", proba); pd.Series(yte).to_csv(outdir/"y_true.csv", index=False)
    results["LR"] = (rep["macro avg"]["f1-score"], proba)

    # ---- Linear SVM + calibration (probabilities) ----
    lin = Pipeline([("pre", pre),
                    ("clf", CalibratedClassifierCV(LinearSVC(random_state=RANDOM_STATE),
                                                   method="sigmoid", cv=3))])
    outdir = outroot / "LinSVM"; outdir.mkdir(parents=True, exist_ok=True)
    lin.fit(Xtr, ytr)
    yhat = lin.predict(Xte)
    proba = lin.predict_proba(Xte)
    rep = classification_report(yte, yhat, digits=4, output_dict=True)
    cm = confusion_matrix(yte, yhat, labels=np.unique(yte))
    with open(outdir/"report.json","w") as f: json.dump(rep, f, indent=2)
    pd.DataFrame(cm, index=np.unique(yte), columns=np.unique(yte)).to_csv(outdir/"confusion_matrix.csv")
    pd.DataFrame([{
        "accuracy": rep.get("accuracy", None),
        "precision_macro": rep["macro avg"]["precision"],
        "recall_macro": rep["macro avg"]["recall"],
        "f1_macro": rep["macro avg"]["f1-score"],
        "precision_weighted": rep["weighted avg"]["precision"],
        "recall_weighted": rep["weighted avg"]["recall"],
        "f1_weighted": rep["weighted avg"]["f1-score"],
    }]).to_csv(outdir/"metrics_summary.csv", index=False)
    np.save(outdir/"proba.npy", proba); pd.Series(yte).to_csv(outdir/"y_true.csv", index=False)
    results["LinSVM"] = (rep["macro avg"]["f1-score"], proba)

    # ---- Random Forest (optional; densify safely) ----
    if try_rf:
        outdir = outroot / "RF"; outdir.mkdir(parents=True, exist_ok=True)
        try:
            # Build a *separate* preprocessor that outputs dense only for RF
            num_cols = pre.transformers_[0][2]
            cat_cols = pre.transformers_[1][2]
            num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                                 ("scale", StandardScaler(with_mean=True))])  # dense
            try:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.01)
            except TypeError:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                 ("ohe", ohe)])
            pre_dense = ColumnTransformer(
                [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
                sparse_threshold=0.0
            )

            rf = Pipeline([("pre", pre_dense),
                           ("clf", RandomForestClassifier(
                               n_estimators=300, max_depth=None,
                               class_weight="balanced_subsample",
                               n_jobs=-1, random_state=RANDOM_STATE))])
            rf.fit(Xtr, ytr)
            yhat = rf.predict(Xte)
            # RF has predict_proba
            proba = rf.predict_proba(Xte)
            rep = classification_report(yte, yhat, digits=4, output_dict=True)
            cm = confusion_matrix(yte, yhat, labels=np.unique(yte))
            with open(outdir/"report.json","w") as f: json.dump(rep, f, indent=2)
            pd.DataFrame(cm, index=np.unique(yte), columns=np.unique(yte)).to_csv(outdir/"confusion_matrix.csv")
            pd.DataFrame([{
                "accuracy": rep.get("accuracy", None),
                "precision_macro": rep["macro avg"]["precision"],
                "recall_macro": rep["macro avg"]["recall"],
                "f1_macro": rep["macro avg"]["f1-score"],
                "precision_weighted": rep["weighted avg"]["precision"],
                "recall_weighted": rep["weighted avg"]["recall"],
                "f1_weighted": rep["weighted avg"]["f1-score"],
            }]).to_csv(outdir/"metrics_summary.csv", index=False)
            np.save(outdir/"proba.npy", proba); pd.Series(yte).to_csv(outdir/"y_true.csv", index=False)
            results["RF"] = (rep["macro avg"]["f1-score"], proba)
        except Exception as e:
            # If densifying explodes RAM or any error occurs, just skip RF
            with open(outdir/"SKIPPED.txt","w") as f: f.write(str(e))

    # pick best by macro-F1
    best_name = max(results.items(), key=lambda kv: kv[1][0])[0]
    best_proba = results[best_name][1]
    return best_name, best_proba


def train_lr(pre, Xtr, ytr, Xte, yte, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    clf = Pipeline([("pre", pre),
                    ("lr", LogisticRegression(max_iter=200, class_weight="balanced",
                                             solver="saga", n_jobs=-1))])
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)
    proba = clf.predict_proba(Xte)

    rep = classification_report(yte, yhat, digits=4, output_dict=True)
    with open(outdir/"report.json","w") as f:
        json.dump(rep, f, indent=2)

    # also save a simple CSV summary of key metrics
    summary = {
    "accuracy": rep.get("accuracy", None),
    "precision_macro": rep["macro avg"]["precision"],
    "recall_macro": rep["macro avg"]["recall"],
    "f1_macro": rep["macro avg"]["f1-score"],
    "precision_weighted": rep["weighted avg"]["precision"],
    "recall_weighted": rep["weighted avg"]["recall"],
    "f1_weighted": rep["weighted avg"]["f1-score"],
    }
    pd.DataFrame([summary]).to_csv(outdir / "metrics_summary.csv", index=False)

    # after: with open(RESULTS_DIR/"Fusion"/"report.json","w") as f: json.dump(...)

# also save a compact CSV summary for Fusion


    cm = confusion_matrix(yte, yhat, labels=np.unique(yte))
    pd.DataFrame(cm, index=np.unique(yte), columns=np.unique(yte)).to_csv(outdir/"confusion_matrix.csv", index=True)

    np.save(outdir/"proba.npy", proba)
    pd.Series(yte).to_csv(outdir/"y_true.csv", index=False)

    return rep, proba, np.unique(yte)

def plot_confusion_matrix(cm_df, title, out_path):
    """
    cm_df: pandas DataFrame with index = true labels, columns = predicted labels.
    out_path: where to save the PNG.
    """
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ========= MAIN =========
all_probs = {}   # modality -> (proba, classes, y_true)
for modality, folder in MODALITY_FOLDERS.items():
    print(f"\n=== {modality} ===")
    X, y, groups = class_aware_stream(folder, ROW_CAP, MIN_PER_CLASS)

    # group split (no leakage)
    Xtr, Xte, ytr, yte = grouped_split(X, y, groups, test_size=TEST_SIZE, seed=RANDOM_STATE)

    pre = build_preproc(X)
    outroot = RESULTS_DIR / modality
    best_name, proba = train_and_select_models(pre, Xtr, ytr, Xte, yte, outroot, try_rf=True)

    classes = np.unique(yte)
    all_probs[modality] = (proba, classes, yte.reset_index(drop=True), best_name)
    print(f"[{modality}] best={best_name}  proba_shape={proba.shape}")


# ========= LATE FUSION: simple average =========
modalities = list(all_probs.keys())

# log which model won per modality
print("Fusing winners:", {m: all_probs[m][3] for m in modalities})

# align lengths by truncating to the smallest test fold
minlen = min(len(all_probs[m][2]) for m in modalities)

# take y_true from the first modality and truncate to minlen
y_true = all_probs[modalities[0]][2][:minlen].reset_index(drop=True)

# stack probabilities from each modality (truncated to minlen)
prob_list = []
for m in modalities:
    proba, classes, y_, best_name = all_probs[m]
    prob_list.append(proba[:minlen])

avg_proba = np.mean(prob_list, axis=0)

# map argmax to labels (use sorted unique labels from y_true)
labels = np.array(sorted(np.unique(y_true)))
y_pred = labels[avg_proba.argmax(axis=1)]

from sklearn.metrics import classification_report, confusion_matrix
import json

fused_rep = classification_report(y_true, y_pred, digits=4, output_dict=True)

# ===== Save Fusion results =====
fusion_dir = RESULTS_DIR / "Fusion"
fusion_dir.mkdir(parents=True, exist_ok=True)

# full JSON report
with open(fusion_dir / "report.json", "w") as f:
    json.dump(fused_rep, f, indent=2)

# concise metrics CSV
summary_fused = {
    "accuracy": fused_rep.get("accuracy", None),
    "precision_macro": fused_rep["macro avg"]["precision"],
    "recall_macro": fused_rep["macro avg"]["recall"],
    "f1_macro": fused_rep["macro avg"]["f1-score"],
    "precision_weighted": fused_rep["weighted avg"]["precision"],
    "recall_weighted": fused_rep["weighted avg"]["recall"],
    "f1_weighted": fused_rep["weighted avg"]["f1-score"],
}
pd.DataFrame([summary_fused]).to_csv(fusion_dir / "metrics_summary.csv", index=False)

# confusion matrix
pd.DataFrame(confusion_matrix(y_true, y_pred, labels=np.unique(y_true)),
             index=np.unique(y_true), columns=np.unique(y_true))\
  .to_csv(fusion_dir / "confusion_matrix.csv")

print("\n=== FUSION (avg) ===")
print(json.dumps(fused_rep["macro avg"], indent=2))

# === PLOTTING SECTION ===
RESULTS_DIR = Path("results_fusion")   # change if you used different name
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

modalities = ["Network", "Linux", "Windows", "Telemetry"]

# 1) Confusion matrices for each modality + fusion
for modality in modalities:
    cm_path = RESULTS_DIR / modality / "best" / "confusion_matrix.csv"
    # if you don't have "best" subfolder, adapt path:
    # e.g. RESULTS_DIR / modality / "LinSVM" / "confusion_matrix.csv"
    if cm_path.exists():
        cm_df = pd.read_csv(cm_path, index_col=0)
        plot_confusion_matrix(
            cm_df,
            title=f"{modality} – Confusion Matrix",
            out_path=PLOTS_DIR / f"{modality}_confusion_matrix.png",
        )

# Fusion confusion matrix
fusion_cm_path = RESULTS_DIR / "Fusion" / "confusion_matrix.csv"
if fusion_cm_path.exists():
    cm_df_f = pd.read_csv(fusion_cm_path, index_col=0)
    plot_confusion_matrix(
        cm_df_f,
        title="Fusion – Confusion Matrix",
        out_path=PLOTS_DIR / "Fusion_confusion_matrix.png",
    )

# 2) Per-modality metrics per model
for modality in modalities:
    plot_modality_models_metrics(RESULTS_DIR, modality, PLOTS_DIR)

# 3) Fusion vs best per modality
plot_fusion_vs_modalities(RESULTS_DIR, modalities, PLOTS_DIR)

# 4) Class distribution (if you expose load_modality and MODALITY_FOLDERS)
try:
    from fus import load_modality  # or just use the local function directly
except ImportError:
    pass  # if load_modality is in this same file, just call it directly

try:
    MODALITY_FOLDERS  # dict like {"Network": "./Network", ...}
    plot_class_distribution(load_modality, MODALITY_FOLDERS, row_cap=300000, out_dir=PLOTS_DIR)
except Exception as e:
    print("[WARN] Could not plot class distribution:", e)



