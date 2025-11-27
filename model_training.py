# model_training.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tensorflow.keras import layers, models, callbacks
from data_simulation import simulate_transactions
from utils import features_from_df, fit_scaler, scale_features, save_sklearn_model, save_keras_model

def train_models(df, test_size=0.2, random_state=42):
    X = features_from_df(df)
    y = ((df['amount'] > df['amount'].mean() + 3*df['amount'].std()) | (df['items_in_cart'] > 10) | (df['ip_entropy'] < 0.02)).astype(int)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = fit_scaler(X_train)
    X_train_s = scale_features(X_train, scaler)
    X_test_s = scale_features(X_test, scaler)
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=random_state)
    iso.fit(X_train_s)
    save_sklearn_model(iso, "isolation_forest")
    ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.01)
    ocsvm.fit(X_train_s)
    save_sklearn_model(ocsvm, "oneclass_svm")
    input_dim = X_train_s.shape[1]
    latent_dim = max(4, input_dim//2)
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.Dense(32, activation='relu')(x)
    latent = layers.Dense(latent_dim, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(latent)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(input_dim, activation='linear')(x)
    auto = models.Model(inputs=inp, outputs=out)
    auto.compile(optimizer='adam', loss='mse')
    early = callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')
    auto.fit(X_train_s, X_train_s, validation_data=(X_test_s, X_test_s),
             epochs=100, batch_size=128, callbacks=[early], verbose=1)
    save_keras_model(auto, "autoencoder")
    eval_dict = {}
    iso_scores = -iso.decision_function(X_test_s)
    ocsvm_scores = -ocsvm.decision_function(X_test_s)
    recon = auto.predict(X_test_s)
    mse = np.mean(np.square(recon - X_test_s), axis=1)
    for name, scores in [('isolation_forest', iso_scores), ('oneclass_svm', ocsvm_scores), ('autoencoder', mse)]:
        smin, smax = scores.min(), scores.max()
        if smax - smin == 0:
            sn = np.zeros_like(scores)
        else:
            sn = (scores - smin) / (smax - smin)
        try:
            roc = roc_auc_score(y_test, sn)
        except Exception:
            roc = float('nan')
        thr = np.percentile(sn, 99)
        preds = (sn >= thr).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_test, preds, average='binary', zero_division=0)
        eval_dict[name] = {'roc_auc': float(roc), 'precision': float(p), 'recall': float(r), 'f1': float(f)}
    return eval_dict

if __name__ == "__main__":
    print("Generating data...")
    df = simulate_transactions(8000)
    print("Training models...")
    results = train_models(df)
    print("Evaluation results:", results)
    print("Artifacts saved in artifacts/ directory.")
