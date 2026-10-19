"""Simulate wafers, measure them, fit the combined score and evaluate on held-out images.

    python run_experiment.py              # main run (seed 0) with figures + 10-seed repeat
    python run_experiment.py --seeds 1    # main run only

For each seed: 480 training wafers (240 good / 240 faulty) fit the score, and a
separate set of 240 test wafers (120 / 120), drawn with a different seed, is used
only for evaluation.
"""
import argparse
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from wafer.measure import FEATURE_NAMES, measure
from wafer.simulate import generate_dataset

N_TRAIN, N_TEST = 480, 240
OUT = "results"


def run(seed):
    tr_imgs, y_tr, _ = generate_dataset(N_TRAIN // 2, N_TRAIN // 2, seed)
    te_imgs, y_te, te_tags = generate_dataset(N_TEST // 2, N_TEST // 2, seed + 10_000)
    F_tr = np.array([measure(i) for i in tr_imgs])
    F_te = np.array([measure(i) for i in te_imgs])

    # One score per wafer: a weighted sum of the standardised measures (logistic
    # regression fitted on the training wafers only). score > 0 means "faulty".
    scaler = StandardScaler().fit(F_tr)
    clf = LogisticRegression().fit(scaler.transform(F_tr), y_tr)
    score = clf.decision_function(scaler.transform(F_te))
    pred = (score > 0).astype(int)
    tags = np.array(te_tags)

    res = {
        "seed": seed,
        "accuracy": float(accuracy_score(y_te, pred)),
        "roc_auc": float(roc_auc_score(y_te, score)),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),  # rows: actual good/faulty
        "false_positive_rate": float(pred[y_te == 0].mean()),
        "recall_by_fault": {
            "bend only": float(pred[tags == "bend"].mean()),
            "width only": float(pred[tags == "width"].mean()),
            "corner only": float(pred[tags == "corner"].mean()),
            "two or three faults": float(pred[np.char.find(tags.astype(str), "+") >= 0].mean()),
        },
        "single_measure_auc": {n: float(roc_auc_score(y_te, F_te[:, j])) for j, n in enumerate(FEATURE_NAMES)},
        "score_weights_on_standardised_measures": dict(zip(FEATURE_NAMES, map(float, clf.coef_[0]))),
    }
    return res, dict(te_imgs=te_imgs, y_te=y_te, tags=tags, score=score, pred=pred, F_te=F_te)


def bootstrap_ci(y, score, pred, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    accs, aucs = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        accs.append(accuracy_score(y[idx], pred[idx]))
        aucs.append(roc_auc_score(y[idx], score[idx]))
    return [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))], \
           [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def make_figures(res, d):
    os.makedirs(OUT, exist_ok=True)
    imgs, y, tags, score, pred, F = d["te_imgs"], d["y_te"], d["tags"], d["score"], d["pred"], d["F_te"]

    # 1. Example wafers
    picks = [("good", "Good")] + [(t, f"{t.capitalize()} fault") for t in ("bend", "width", "corner")]
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.3))
    for a, (t, title) in zip(ax, picks):
        i = int(np.nonzero(tags == t)[0][0])
        a.imshow(imgs[i], cmap="gray", vmin=0, vmax=255); a.set_title(title); a.axis("off")
    fig.tight_layout(); fig.savefig(f"{OUT}/sample_wafers.png", dpi=130); plt.close(fig)

    # 2. Pipeline stages for a typical good and a typical faulty wafer (median score of each class)
    def median_index(label):
        idx = np.nonzero(y == label)[0]
        return int(idx[np.argsort(score[idx])[len(idx) // 2]])
    rows = [median_index(0), median_index(1)]
    fig, ax = plt.subplots(2, 4, figsize=(14, 8.2))
    captions = []
    for r, i in enumerate(rows):
        feats, dbg = measure(imgs[i], return_debug=True)
        overlay = cv2.cvtColor(dbg["edges"], cv2.COLOR_GRAY2RGB)
        for rho, theta, _ in dbg["lines"]:
            p = np.array([rho * np.cos(theta), rho * np.sin(theta)])
            q = np.array([-np.sin(theta), np.cos(theta)]) * 400
            cv2.line(overlay, tuple(map(int, p - q)), tuple(map(int, p + q)), (255, 70, 70), 1)
        for c in dbg["corners"]:
            cv2.circle(overlay, tuple(map(int, c)), 6, (70, 230, 70), 1)
        panels = [(imgs[i], "Input"), (dbg["mask"], "Otsu threshold (straightened)"),
                  (dbg["edges"], "Canny edges"), (overlay, "Hough lines and corners")]
        for a, (im, title) in zip(ax[r], panels):
            a.imshow(im, cmap="gray"); a.set_title(title, fontsize=11); a.axis("off")
        captions.append(f"{'Typical good wafer' if r == 0 else f'Typical faulty wafer ({tags[i]})'}: straightness {feats[0]:.2f} px, "
                        f"corner offset {feats[1]:.2f} px, width variation {feats[2]:.3f}, score {score[i]:.2f}")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=5.0)
    for r, text in enumerate(captions):
        fig.text(0.01, ax[r][0].get_position().y1 + 0.045, text, fontsize=11, weight="bold")
    fig.savefig(f"{OUT}/pipeline_example.png", dpi=120); plt.close(fig)

    # 3. ROC curves: combined score and each measure alone
    plt.figure(figsize=(5.2, 5))
    fpr, tpr, _ = roc_curve(y, score)
    plt.plot(fpr, tpr, lw=2.2, label=f"Combined score (AUC {res['roc_auc']:.3f})")
    for j, n in enumerate(FEATURE_NAMES):
        f, t, _ = roc_curve(y, F[:, j])
        plt.plot(f, t, lw=1, ls="--", label=f"{n} (AUC {res['single_measure_auc'][n]:.3f})")
    plt.plot([0, 1], [0, 1], "k:", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curve, 240 held-out test wafers"); plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout(); plt.savefig(f"{OUT}/roc_curve.png", dpi=130); plt.close()

    # 4. Confusion matrix
    cm = np.array(res["confusion_matrix"])
    plt.figure(figsize=(4.2, 3.8)); plt.imshow(cm, cmap="Blues")
    plt.xticks([0, 1], ["good", "faulty"]); plt.yticks([0, 1], ["good", "faulty"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion matrix (test set)")
    plt.tight_layout(); plt.savefig(f"{OUT}/confusion_matrix.png", dpi=130); plt.close()

    # 5. Score distribution
    plt.figure(figsize=(6, 3.8))
    bins = np.linspace(score.min(), score.max(), 35)
    plt.hist(score[y == 0], bins, alpha=0.65, label="good")
    plt.hist(score[y == 1], bins, alpha=0.65, label="faulty")
    plt.axvline(0, color="k", ls="--", lw=1, label="threshold")
    plt.xlabel("Combined wafer score (higher = worse)"); plt.ylabel("Wafers"); plt.legend()
    plt.title("Score distribution (test set)"); plt.tight_layout()
    plt.savefig(f"{OUT}/score_distribution.png", dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10, help="number of seeds to repeat over (seed 0 is the main run)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    main_res, main_data = run(0)
    main_res["accuracy_95ci"], main_res["roc_auc_95ci"] = bootstrap_ci(
        main_data["y_te"], main_data["score"], main_data["pred"])
    make_figures(main_res, main_data)

    summary = {"main_run": main_res}
    if args.seeds > 1:
        runs = [main_res] + [run(s)[0] for s in range(1, args.seeds)]
        acc = np.array([r["accuracy"] for r in runs]); auc = np.array([r["roc_auc"] for r in runs])
        summary["repeat"] = {"n_seeds": args.seeds, "accuracy_mean": float(acc.mean()), "accuracy_std": float(acc.std(ddof=1)),
                             "roc_auc_mean": float(auc.mean()), "roc_auc_std": float(auc.std(ddof=1)),
                             "accuracy_per_seed": acc.round(4).tolist(), "roc_auc_per_seed": auc.round(4).tolist()}
    with open(f"{OUT}/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    r = main_res
    print(f"Main run (seed 0): accuracy {r['accuracy']:.3f} (95% CI {r['accuracy_95ci'][0]:.3f}-{r['accuracy_95ci'][1]:.3f}), "
          f"ROC AUC {r['roc_auc']:.3f} (95% CI {r['roc_auc_95ci'][0]:.3f}-{r['roc_auc_95ci'][1]:.3f})")
    print("Confusion matrix [[TN FP] [FN TP]]:", r["confusion_matrix"], "| false positive rate:", round(r["false_positive_rate"], 3))
    print("Recall by fault:", {k: round(v, 3) for k, v in r["recall_by_fault"].items()})
    print("Single-measure AUC:", {k: round(v, 3) for k, v in r["single_measure_auc"].items()})
    print("Score weights:", {k: round(v, 2) for k, v in r["score_weights_on_standardised_measures"].items()})
    if "repeat" in summary:
        p = summary["repeat"]
        print(f"Over {p['n_seeds']} seeds: accuracy {p['accuracy_mean']:.3f} +/- {p['accuracy_std']:.3f}, "
              f"ROC AUC {p['roc_auc_mean']:.3f} +/- {p['roc_auc_std']:.3f}")


if __name__ == "__main__":
    main()
