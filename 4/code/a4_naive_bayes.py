"""
SYDE 572 - Assignment 4: Text Classification with Naive Bayes

All Naive Bayes code below is written from scratch (no scikit-learn).
scikit-learn is used only in the optional "Train a classifier using
scikit-learn" task, and to cross-check the Gaussian NB on iris.

Run:  python a4_naive_bayes.py
Outputs: figures/*.pdf and results/results.json (+ console log)
"""
from codecs import open
from collections import Counter, defaultdict
import csv
import json
import math
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})

RESULTS = {}


# ---------------------------------------------------------------------------
# Reading the data
# ---------------------------------------------------------------------------
def read_documents(doc_file, label_col=1):
    """label_col=1 -> sentiment (pos/neg), label_col=0 -> topic."""
    docs, labels = [], []
    with open(doc_file, encoding="utf-8") as f:
        for line in f:
            words = line.strip().split()
            docs.append(words[3:])
            labels.append(words[label_col])
    return docs, labels


def read_all(doc_file):
    docs, sent, topic = [], [], []
    with open(doc_file, encoding="utf-8") as f:
        for line in f:
            words = line.strip().split()
            docs.append(words[3:])
            topic.append(words[0])
            sent.append(words[1])
    return docs, sent, topic


# ---------------------------------------------------------------------------
# Naive Bayes (multinomial, Laplace / additive smoothing)
# ---------------------------------------------------------------------------
def train_nb(documents, labels, alpha=1.0):
    """
    Estimate the parameters of a multinomial Naive Bayes model.

      P(y)       = N_y / N                              (class prior, MLE)
      P(w | y)   = (c(w, y) + alpha) / (N_y + alpha*|V|) (additive smoothing)

    |V| counts every word seen in training plus one slot for unseen words,
    so P(. | y) is a proper distribution and unseen words get
    alpha / (N_y + alpha*|V|).  Everything is stored as log probabilities.
    """
    classes = sorted(set(labels))
    class_counts = Counter(labels)
    word_counts = {c: Counter() for c in classes}
    for doc, y in zip(documents, labels):
        word_counts[y].update(doc)

    vocab = set()
    for c in classes:
        vocab.update(word_counts[c])
    V = len(vocab) + 1  # +1 for the "unknown word" type

    n_docs = len(documents)
    log_prior, log_lik, log_unk, n_tokens = {}, {}, {}, {}
    for c in classes:
        total = sum(word_counts[c].values())
        denom = math.log(total + alpha * V)
        log_prior[c] = math.log(class_counts[c] / n_docs)
        log_lik[c] = {w: math.log(word_counts[c][w] + alpha) - denom for w in vocab}
        log_unk[c] = math.log(alpha) - denom
        n_tokens[c] = total

    return {"classes": classes, "log_prior": log_prior, "log_lik": log_lik,
            "log_unk": log_unk, "word_counts": word_counts, "vocab": vocab,
            "alpha": alpha, "n_tokens": n_tokens}


def score_doc_label(document, label, model):
    """log P(document, label) = log P(label) + sum_i log P(w_i | label)."""
    ll = model["log_lik"][label]
    unk = model["log_unk"][label]
    return model["log_prior"][label] + sum(ll.get(w, unk) for w in document)


def classify_nb(document, model):
    return max(model["classes"], key=lambda c: score_doc_label(document, c, model))


def classify_documents(docs, model):
    return [classify_nb(d, model) for d in docs]


def accuracy(true_labels, guessed_labels):
    true_labels, guessed_labels = list(true_labels), list(guessed_labels)
    return sum(t == g for t, g in zip(true_labels, guessed_labels)) / len(true_labels)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def acc_ci(acc, n, z=1.96):
    """95% normal-approximation (Wald) interval for a binomial proportion."""
    se = math.sqrt(acc * (1 - acc) / n)
    return acc - z * se, acc + z * se


def mcnemar(true, pred_a, pred_b):
    """Exact McNemar test: are classifiers A and B equally accurate?"""
    b = sum((a == t) and (p != t) for t, a, p in zip(true, pred_a, pred_b))  # A right, B wrong
    c = sum((a != t) and (p == t) for t, a, p in zip(true, pred_a, pred_b))  # A wrong, B right
    p = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    return b, c, p


def summarize(name, true, pred):
    acc = accuracy(true, pred)
    lo, hi = acc_ci(acc, len(true))
    print(f"  {name:<38s} acc = {acc:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    return {"acc": acc, "ci": [lo, hi], "n": len(true)}


def pick_alpha(train_docs, train_labels, grid, dev_frac=0.1):
    """Pick the smoothing parameter on a held-out dev slice of the TRAINING data."""
    k = int((1 - dev_frac) * len(train_docs))
    tr_d, tr_l, dv_d, dv_l = train_docs[:k], train_labels[:k], train_docs[k:], train_labels[k:]
    scores = {}
    for a in grid:
        m = train_nb(tr_d, tr_l, alpha=a)
        scores[a] = accuracy(dv_l, classify_documents(dv_d, m))
    best = max(grid, key=lambda a: (scores[a], -abs(math.log(a))))
    return best, scores


# ---------------------------------------------------------------------------
# Feature transformations (optional task: improve the features)
# ---------------------------------------------------------------------------
NEGATIONS = {"not", "no", "never", "n't", "nothing", "nobody", "none", "nor",
             "neither", "nowhere", "cannot", "without", "hardly"}
CLAUSE_END = {".", ",", "!", "?", ";", ":", "(", ")", "--", "-"}


def mark_negation(doc):
    """Pang et al. (2002): prefix NOT_ to every token after a negation word
    up to the next punctuation mark."""
    out, neg = [], False
    for w in doc:
        if w in CLAUSE_END:
            neg = False
            out.append(w)
        elif w in NEGATIONS:
            neg = True
            out.append(w)
        else:
            out.append("NOT_" + w if neg else w)
    return out


def add_bigrams(doc):
    return doc + [a + "__" + b for a, b in zip(doc, doc[1:])]


# ===========================================================================
def main():
    t0 = time.time()
    all_docs, all_sent, all_topic = read_all(os.path.join(DATA, "all_sentiment_shuffled.txt"))
    N = len(all_docs)
    split_point = int(0.80 * N)
    train_docs, eval_docs = all_docs[:split_point], all_docs[split_point:]
    train_labels, eval_labels = all_sent[:split_point], all_sent[split_point:]
    train_topic, eval_topic = all_topic[:split_point], all_topic[split_point:]

    lens = [len(d) for d in all_docs]
    RESULTS["data"] = {
        "n_docs": N, "n_train": len(train_docs), "n_eval": len(eval_docs),
        "train_label_counts": dict(Counter(train_labels)),
        "eval_label_counts": dict(Counter(eval_labels)),
        "topic_counts": dict(Counter(all_topic)),
        "mean_len": float(np.mean(lens)), "median_len": float(np.median(lens)),
        "min_len": int(min(lens)), "max_len": int(max(lens)),
        "n_tokens": int(sum(lens)),
    }
    print("Data:", json.dumps(RESULTS["data"], indent=1))

    # -------------------------------------------------------------- sanity checks
    print("\n=== Sanity checks (alpha = 1) ===")
    m1 = train_nb(train_docs, train_labels, alpha=1.0)
    RESULTS["vocab_size"] = len(m1["vocab"])
    RESULTS["class_tokens"] = m1["n_tokens"]
    RESULTS["priors"] = {c: math.exp(v) for c, v in m1["log_prior"].items()}
    sanity = {}
    for doc in (["great"], ["bad"], ["a", "top-quality", "performance"]):
        key = " ".join(doc)
        sanity[key] = {}
        for c in ("pos", "neg"):
            lp = score_doc_label(doc, c, m1)
            sanity[key][c] = {"logp": lp, "p": math.exp(lp)}
            print(f"  P({c}, {doc}) = exp({lp:.4f}) = {math.exp(lp):.3e}")
    RESULTS["sanity"] = sanity
    small = [["great"], ["bad"], ["i", "love", "it"], ["waste", "of", "money"],
             ["not", "good"], ["highly", "recommended"], ["boring"],
             ["a", "top-quality", "performance"], ["it", "broke", "after", "a", "week"]]
    RESULTS["small_docs"] = {" ".join(d): classify_nb(d, m1) for d in small}
    for d in small:
        print(f"  classify_nb({' '.join(d)!r}) -> {classify_nb(d, m1)}")

    # ---------------------------------------------------------- alpha selection
    print("\n=== Smoothing parameter (dev = last 10% of training) ===")
    grid = [0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    best_alpha, dev_scores = pick_alpha(train_docs, train_labels, grid)
    for a in grid:
        print(f"  alpha = {a:<6} dev acc = {dev_scores[a]:.4f}")
    print("  best alpha:", best_alpha)
    RESULTS["alpha_dev"] = {"grid": grid, "dev_acc": [dev_scores[a] for a in grid],
                            "best": best_alpha}
    # test accuracy as a function of alpha (for the report only, not for selection)
    test_by_alpha = []
    for a in grid:
        m = train_nb(train_docs, train_labels, alpha=a)
        test_by_alpha.append(accuracy(eval_labels, classify_documents(eval_docs, m)))
    RESULTS["alpha_test"] = test_by_alpha

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.semilogx(grid, [dev_scores[a] for a in grid], "o-", label="dev (10% of train)")
    ax.semilogx(grid, test_by_alpha, "s--", label="evaluation set")
    ax.axvline(best_alpha, color="grey", lw=0.8, ls=":")
    ax.set_xlabel(r"smoothing parameter $\alpha$")
    ax.set_ylabel("accuracy")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(FIG, "alpha_sweep.pdf"))
    plt.close(fig)

    # --------------------------------------------------------------- evaluation
    print("\n=== Evaluation on the 20% test set ===")
    model = train_nb(train_docs, train_labels, alpha=best_alpha)
    pred = classify_documents(eval_docs, model)
    RESULTS["main"] = summarize(f"NB (alpha={best_alpha})", eval_labels, pred)
    RESULTS["main"]["alpha"] = best_alpha
    pred_a1 = classify_documents(eval_docs, m1)
    RESULTS["main_alpha1"] = summarize("NB (alpha=1)", eval_labels, pred_a1)
    maj = Counter(train_labels).most_common(1)[0][0]
    RESULTS["majority"] = summarize(f"majority baseline ({maj})", eval_labels, [maj] * len(eval_labels))
    # confusion matrix
    cm = {t: {p: 0 for p in ("pos", "neg")} for t in ("pos", "neg")}
    for t, p in zip(eval_labels, pred):
        cm[t][p] += 1
    RESULTS["confusion"] = cm
    print("  confusion (rows=true):", cm)
    # per-topic accuracy on the eval set
    per_topic = {}
    for tp in sorted(set(eval_topic)):
        idx = [i for i, x in enumerate(eval_topic) if x == tp]
        per_topic[tp] = {"acc": accuracy([eval_labels[i] for i in idx], [pred[i] for i in idx]),
                         "n": len(idx)}
    RESULTS["per_topic_acc"] = per_topic
    print("  per-topic accuracy:", per_topic)

    # ----------------------------------------------------------- error analysis
    print("\n=== Error analysis ===")
    errors = []
    for i, (d, t, p) in enumerate(zip(eval_docs, eval_labels, pred)):
        if t != p:
            margin = score_doc_label(d, p, model) - score_doc_label(d, t, model)
            errors.append((i, len(d), margin, t, p))
    RESULTS["n_errors"] = len(errors)
    short = sorted([e for e in errors if e[1] <= 40], key=lambda e: -e[2])[:10]
    err_list = []
    for i, L, margin, t, p in short:
        d = eval_docs[i]
        # per-word contribution to log P(p|d)/P(t|d)
        contrib = Counter()
        for w in d:
            contrib[w] += (model["log_lik"][p].get(w, model["log_unk"][p])
                           - model["log_lik"][t].get(w, model["log_unk"][t]))
        top = [(w, round(v, 2)) for w, v in contrib.most_common(5)]
        err_list.append({"idx": split_point + i, "topic": eval_topic[i], "true": t, "pred": p,
                         "len": L, "log_odds": margin,
                         "p_wrong": 1 / (1 + math.exp(-margin)),
                         "text": " ".join(d), "top_words": top})
        print(f"  [{eval_topic[i]}] true={t} pred={p} len={L} log-odds={margin:.2f}")
        print("     ", " ".join(d))
        print("      strongest words:", top)
    RESULTS["errors"] = err_list
    # margin distribution: correct vs. incorrect
    margins_c, margins_e = [], []
    for d, t, p in zip(eval_docs, eval_labels, pred):
        s = abs(score_doc_label(d, "pos", model) - score_doc_label(d, "neg", model))
        (margins_c if t == p else margins_e).append(s)
    RESULTS["median_margin_correct"] = float(np.median(margins_c))
    RESULTS["median_margin_error"] = float(np.median(margins_e))
    # errors vs document length
    len_bins = [0, 25, 50, 100, 200, 10_000]
    by_len = []
    for lo, hi in zip(len_bins, len_bins[1:]):
        idx = [i for i, d in enumerate(eval_docs) if lo < len(d) <= hi]
        if idx:
            by_len.append({"range": f"{lo+1}-{hi}" if hi < 10_000 else f">{lo}", "n": len(idx),
                           "acc": accuracy([eval_labels[i] for i in idx], [pred[i] for i in idx])})
    RESULTS["acc_by_length"] = by_len
    print("  accuracy by length:", by_len)

    # =========================================================== OPTIONAL TASKS
    # ------------------------------------------------- six-category classifier
    print("\n=== (Opt) Six-category topic classifier ===")
    best_alpha_t, dev_t = pick_alpha(train_docs, train_topic, grid)
    tmodel = train_nb(train_docs, train_topic, alpha=best_alpha_t)
    tpred = classify_documents(eval_docs, tmodel)
    RESULTS["topic"] = summarize(f"topic NB (alpha={best_alpha_t})", eval_topic, tpred)
    RESULTS["topic"]["alpha"] = best_alpha_t
    RESULTS["topic"]["dev"] = [dev_t[a] for a in grid]
    topics = sorted(set(all_topic))
    tcm = np.zeros((6, 6), dtype=int)
    for t, p in zip(eval_topic, tpred):
        tcm[topics.index(t), topics.index(p)] += 1
    RESULTS["topic"]["confusion"] = tcm.tolist()
    RESULTS["topic"]["labels"] = topics
    RESULTS["topic"]["majority"] = max(Counter(eval_topic).values()) / len(eval_topic)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    rown = tcm / tcm.sum(1, keepdims=True)
    im = ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, tcm[i, j], ha="center", va="center",
                    color="white" if rown[i, j] > 0.6 else "black", fontsize=9)
    ax.set_xticks(range(6), topics, rotation=35, ha="right")
    ax.set_yticks(range(6), topics)
    ax.set_xlabel("predicted topic")
    ax.set_ylabel("true topic")
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized")
    fig.savefig(os.path.join(FIG, "topic_confusion.pdf"))
    plt.close(fig)

    # ------------------------------------------------------------ learning curve
    print("\n=== (Opt) Learning curve ===")
    fracs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    lc = []
    for f in fracs:
        k = max(2, int(f * len(train_docs)))
        m = train_nb(train_docs[:k], train_labels[:k], alpha=best_alpha)
        a = accuracy(eval_labels, classify_documents(eval_docs, m))
        mt = train_nb(train_docs[:k], train_topic[:k], alpha=best_alpha_t)
        at = accuracy(eval_topic, classify_documents(eval_docs, mt))
        lc.append({"frac": f, "n": k, "acc": a, "ci": acc_ci(a, len(eval_docs)), "topic_acc": at})
        print(f"  {f:>5.0%} ({k:5d} docs): sentiment acc = {a:.4f}, topic acc = {at:.4f}")
    RESULTS["learning_curve"] = lc
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ns = [r["n"] for r in lc]
    ax.plot(ns, [r["acc"] for r in lc], "o-", label="sentiment (pos/neg)")
    ax.fill_between(ns, [r["ci"][0] for r in lc], [r["ci"][1] for r in lc], alpha=0.2)
    ax.plot(ns, [r["topic_acc"] for r in lc], "s-", label="topic (6 classes)")
    ax.set_xscale("log")
    ax.set_xlabel("number of training documents (log scale)")
    ax.set_ylabel("accuracy on evaluation set")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(os.path.join(FIG, "learning_curve.pdf"))
    plt.close(fig)

    # ------------------------------------------------------ informative features
    print("\n=== (Opt) Most informative features ===")

    def top_features(model, k, c1, c2, min_count=0):
        """Rank words by log P(w|c1) - log P(w|c2) (the per-occurrence log
        likelihood ratio, i.e. the amount one occurrence shifts the log odds)."""
        scores = {}
        for w in model["vocab"]:
            if model["word_counts"][c1][w] + model["word_counts"][c2][w] < min_count:
                continue
            scores[w] = model["log_lik"][c1][w] - model["log_lik"][c2][w]
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:k], ranked[::-1][:k], scores

    def print_top_nb(k, model, c1="pos", c2="neg"):
        top1, top2, _ = top_features(model, k, c1, c2)
        print(f"  strongest {c1}:", ", ".join(f"{w} ({s:.2f})" for w, s in top1))
        print(f"  strongest {c2}:", ", ".join(f"{w} ({s:.2f})" for w, s in top2))
        return top1, top2

    feats = {}
    for a in (1.0, 10.0, 50.0):
        mf = train_nb(train_docs, train_labels, alpha=a)
        print(f" alpha = {a}")
        tp, tn = print_top_nb(15, mf)
        feats[str(a)] = {"pos": tp, "neg": tn}
    RESULTS["top_features"] = feats

    # visualization by polarity: scatter of frequency vs. log-likelihood ratio
    mf = train_nb(train_docs, train_labels, alpha=10.0)
    _, _, sc = top_features(mf, 0, "pos", "neg", min_count=30)
    words = list(sc)
    freq = np.array([mf["word_counts"]["pos"][w] + mf["word_counts"]["neg"][w] for w in words])
    llr = np.array([sc[w] for w in words])
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.scatter(freq, llr, s=4, c=np.where(llr > 0, "#1f77b4", "#d62728"), alpha=0.18, lw=0)
    # label the most polarized words in each frequency band so that labels
    # spread out horizontally; then push labels apart vertically (simple repel)
    edges = [30, 60, 120, 250, 600, 2000, 1e7]
    for sign, color in ((1, "#1f77b4"), (-1, "#d62728")):
        for lo, hi in zip(edges, edges[1:]):
            idx = [i for i in range(len(words)) if lo <= freq[i] < hi and sign * llr[i] > 0]
            idx = sorted(idx, key=lambda i: -sign * llr[i])[:3]
            ys, prev = [], None
            for i in idx:  # most extreme first; keep >= 0.24 apart toward zero
                y = llr[i]
                if prev is not None and abs(prev - y) < 0.24:
                    y = prev - sign * 0.24
                ys.append(y)
                prev = y
            xc = math.sqrt(lo * min(hi, freq.max()))  # band centre (log scale)
            for i, y in zip(idx, ys):
                size = 6.5 + 1.2 * np.log10(freq[i])
                ax.annotate(words[i], (freq[i], llr[i]), xytext=(xc, y), fontsize=size,
                            color=color, ha="center", va="center", fontweight="bold",
                            arrowprops=dict(arrowstyle="-", color=color, lw=0.4, alpha=0.5))
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("word frequency in training set (log scale)")
    ax.set_ylabel(r"$\log P(w\,|\,\mathrm{pos}) - \log P(w\,|\,\mathrm{neg})$")
    ax.set_title(r"Words by polarity ($\alpha=10$, words with $\geq 30$ occurrences)", fontsize=10)
    fig.savefig(os.path.join(FIG, "polarity_words.pdf"))
    plt.close(fig)

    # visualization by topic: words most indicative of each topic vs. the rest
    mt = train_nb(train_docs, train_topic, alpha=10.0)
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.6))
    topic_feats = {}
    tot_all = Counter()
    for c in topics:
        tot_all.update(mt["word_counts"][c])
    V = len(mt["vocab"]) + 1
    for ax, c in zip(axes.flat, topics):
        rest = tot_all.copy()
        rest.subtract(mt["word_counts"][c])
        Nc, Nr = mt["n_tokens"][c], sum(mt["n_tokens"].values()) - mt["n_tokens"][c]
        s = {w: math.log((mt["word_counts"][c][w] + 10) / (Nc + 10 * V))
                - math.log((rest[w] + 10) / (Nr + 10 * V))
             for w in mt["vocab"] if tot_all[w] >= 30}
        best = sorted(s.items(), key=lambda x: -x[1])[:12]
        topic_feats[c] = [(w, round(v, 2)) for w, v in best]
        ax.barh([w for w, _ in best][::-1], [v for _, v in best][::-1], color="#4c72b0")
        ax.set_title(c, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
    fig.supxlabel(r"$\log P(w\,|\,\mathrm{topic}) - \log P(w\,|\,\mathrm{other\ topics})$", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "topic_words.pdf"))
    plt.close(fig)
    RESULTS["topic_features"] = topic_feats
    print("  topic words:", topic_feats)

    # ------------------------------------------------------- Gaussian NB (iris)
    print("\n=== (Opt) Gaussian Naive Bayes on iris ===")
    X, Y = [], []
    with open(os.path.join(DATA, "iris.csv"), encoding="utf-8") as f:
        rd = csv.reader(f)
        header = next(rd)
        for row in rd:
            if row:
                X.append([float(v) for v in row[:4]])
                Y.append(row[4])
    X, Y = np.array(X), np.array(Y)

    def train_gnb(X, Y):
        classes = sorted(set(Y))
        prm = {}
        for c in classes:
            Xc = X[Y == c]
            prm[c] = {"log_prior": math.log(len(Xc) / len(X)),
                      "mu": Xc.mean(0), "var": Xc.var(0)}  # MLE: ddof = 0
        return {"classes": classes, "params": prm}

    def score_gnb(x, c, model):
        p = model["params"][c]
        return p["log_prior"] + np.sum(-0.5 * np.log(2 * np.pi * p["var"])
                                       - (x - p["mu"]) ** 2 / (2 * p["var"]))

    def classify_gnb(Xs, model):
        return [max(model["classes"], key=lambda c: score_gnb(x, c, model)) for x in Xs]

    from sklearn.naive_bayes import GaussianNB
    np.random.seed(572)
    n, n_test = len(Y), 40
    eval_part = np.array([False] * n)
    for i in np.random.choice(range(n), replace=False, size=n_test):
        eval_part[i] = True
    gm = train_gnb(X[~eval_part], Y[~eval_part])
    gpred = classify_gnb(X[eval_part], gm)
    gacc = accuracy(Y[eval_part], gpred)
    sk = GaussianNB().fit(X[~eval_part], Y[~eval_part])
    sk_acc = accuracy(Y[eval_part], sk.predict(X[eval_part]))
    agree = float(np.mean(np.array(gpred) == sk.predict(X[eval_part])))
    print(f"  own GNB acc = {gacc:.4f}, sklearn GaussianNB acc = {sk_acc:.4f}, agreement = {agree}")
    params = {c: {"mu": gm["params"][c]["mu"].round(3).tolist(),
                  "sd": np.sqrt(gm["params"][c]["var"]).round(3).tolist()} for c in gm["classes"]}
    # repeated random splits
    accs = []
    rng = np.random.RandomState(0)
    for _ in range(500):
        ep = np.zeros(n, bool)
        ep[rng.choice(n, n_test, replace=False)] = True
        m = train_gnb(X[~ep], Y[~ep])
        accs.append(accuracy(Y[ep], classify_gnb(X[ep], m)))
    gcm = np.zeros((3, 3), int)
    cls = gm["classes"]
    for t, p in zip(Y[eval_part], gpred):
        gcm[cls.index(t), cls.index(p)] += 1
    RESULTS["iris"] = {"acc": gacc, "sk_acc": sk_acc, "agree": agree, "params": params,
                       "features": header[:4], "repeat_mean": float(np.mean(accs)),
                       "repeat_sd": float(np.std(accs)),
                       "repeat_q": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
                       "confusion": gcm.tolist(), "classes": cls}
    print("  ", RESULTS["iris"])

    # ------------------------------------------------- scikit-learn classifiers
    print("\n=== (Opt) scikit-learn classifiers ===")
    from sklearn.svm import LinearSVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.feature_selection import SelectKBest

    ident = lambda x: x

    def train_sklearn(docs, labels, clf, k="all", vec=None):
        vec = vec or TfidfVectorizer(preprocessor=ident, tokenizer=ident, token_pattern=None,
                                     lowercase=False)
        pipeline = make_pipeline(vec, SelectKBest(k=k), clf)
        pipeline.fit(docs, labels)
        return pipeline

    def classify_documents_sklearn(docs, model):
        return model.predict(docs)

    sk_res = {}
    configs = [
        ("LinearSVC (tf-idf)", lambda: train_sklearn(train_docs, train_labels, LinearSVC())),
        ("LogisticRegression (tf-idf)", lambda: train_sklearn(train_docs, train_labels,
                                                              LogisticRegression(max_iter=2000))),
        ("MLPClassifier (tf-idf, 1000 best)", lambda: train_sklearn(
            train_docs, train_labels,
            MLPClassifier(hidden_layer_sizes=(100,), max_iter=50, early_stopping=True,
                          random_state=0), k=1000)),
        ("MultinomialNB (counts)", lambda: train_sklearn(
            train_docs, train_labels, MultinomialNB(alpha=best_alpha),
            vec=CountVectorizer(preprocessor=ident, tokenizer=ident, token_pattern=None,
                                lowercase=False))),
    ]
    for name, fn in configs:
        t = time.time()
        mdl = fn()
        p = list(classify_documents_sklearn(eval_docs, mdl))
        r = summarize(name, eval_labels, p)
        b, c, pv = mcnemar(eval_labels, pred, p)
        r.update({"mcnemar_nb_right_other_wrong": b, "mcnemar_nb_wrong_other_right": c,
                  "p": pv, "train_s": time.time() - t})
        print(f"      McNemar vs own NB: b={b}, c={c}, p={pv:.3g}")
        sk_res[name] = r
        if name.startswith("MultinomialNB"):
            sk_res[name]["agree_with_own_nb"] = float(np.mean(np.array(p) == np.array(pred)))
    RESULTS["sklearn"] = sk_res

    # --------------------------------------------------------- improve features
    print("\n=== (Opt) Improved features ===")
    feat_res = {}
    variants = {
        "unigrams (baseline)": ident,
        "unigrams + negation marking": mark_negation,
        "unigrams + bigrams": add_bigrams,
        "negation marking + bigrams": lambda d: add_bigrams(mark_negation(d)),
    }
    preds_by_var = {}
    for name, tf in variants.items():
        trd = [tf(d) for d in train_docs]
        evd = [tf(d) for d in eval_docs]
        a_best, _ = pick_alpha(trd, train_labels, grid)
        m = train_nb(trd, train_labels, alpha=a_best)
        p = classify_documents(evd, m)
        preds_by_var[name] = p
        r = summarize(f"{name} (a={a_best})", eval_labels, p)
        b, c, pv = mcnemar(eval_labels, pred, p)
        r.update({"alpha": a_best, "vocab": len(m["vocab"]), "b": b, "c": c, "p": pv})
        print(f"      McNemar vs baseline: b={b}, c={c}, p={pv:.3g}")
        feat_res[name] = r
    RESULTS["features"] = feat_res
    RESULTS["neg_example"] = " ".join(mark_negation(
        "i did n't like this camera , and the lens is not sharp at all .".split()))

    # --------------------------------------------------------- domain sensitivity
    print("\n=== (Opt) Domain sensitivity ===")
    per = {}
    for tp in topics:
        idx = [i for i, x in enumerate(all_topic) if x == tp]
        k = int(0.8 * len(idx))
        per[tp] = ([all_docs[i] for i in idx[:k]], [all_sent[i] for i in idx[:k]],
                   [all_docs[i] for i in idx[k:]], [all_sent[i] for i in idx[k:]])
    dm = np.zeros((6, 6))
    for i, src in enumerate(topics):
        m = train_nb(per[src][0], per[src][1], alpha=best_alpha)
        for j, tgt in enumerate(topics):
            dm[i, j] = accuracy(per[tgt][3], classify_documents(per[tgt][2], m))
    # pooled model trained on the in-domain-train part of all topics
    pooled_tr_d = [d for tp in topics for d in per[tp][0]]
    pooled_tr_l = [l for tp in topics for l in per[tp][1]]
    pm = train_nb(pooled_tr_d, pooled_tr_l, alpha=best_alpha)
    pooled = [accuracy(per[t][3], classify_documents(per[t][2], pm)) for t in topics]
    RESULTS["domain"] = {"topics": topics, "matrix": dm.tolist(), "pooled": pooled,
                         "n_train": {t: len(per[t][0]) for t in topics},
                         "n_test": {t: len(per[t][2]) for t in topics}}
    # camera vs books with McNemar-free CI comparison
    cb = {}
    for src, tgt in (("camera", "camera"), ("camera", "books"), ("books", "books"), ("books", "camera")):
        a = dm[topics.index(src), topics.index(tgt)]
        cb[f"{src}->{tgt}"] = {"acc": a, "ci": acc_ci(a, len(per[tgt][3]))}
    RESULTS["domain"]["camera_books"] = cb
    print(np.round(dm, 3))
    print("  pooled:", np.round(pooled, 3))
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    im = ax.imshow(dm, cmap="viridis", vmin=0.6, vmax=0.9)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{dm[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="black" if dm[i, j] > 0.78 else "white")
    ax.set_xticks(range(6), topics, rotation=35, ha="right")
    ax.set_yticks(range(6), topics)
    ax.set_xlabel("test domain")
    ax.set_ylabel("training domain")
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="sentiment accuracy")
    fig.savefig(os.path.join(FIG, "domain_matrix.pdf"))
    plt.close(fig)

    RESULTS["runtime_s"] = time.time() - t0
    with open(os.path.join(RES, "results.json"), "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=1, default=float)
    print(f"\nDone in {RESULTS['runtime_s']:.1f} s")


if __name__ == "__main__":
    main()
