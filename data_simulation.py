# data_simulation.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def simulate_transactions(n=10000, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    base_time = datetime.utcnow()
    rows = []
    for i in range(n):
        user_id = np.random.randint(1000, 2000)
        amount = max(1.0, np.random.exponential(scale=80.0))
        hour = np.random.randint(0,24)
        device_type = np.random.choice(['mobile','desktop','tablet'], p=[0.6,0.3,0.1])
        country = np.random.choice(['IN','US','GB','DE','FR','CN','BR'], p=[0.4,0.2,0.1,0.08,0.07,0.1,0.05])
        ip_entropy = np.random.rand()
        session_length = np.random.exponential(scale=300)
        items_in_cart = np.random.poisson(2)
        is_guest = np.random.choice([0,1], p=[0.7,0.3])
        speed_score = np.clip(np.random.normal(50 - 0.01*amount, 10), 5, 100)
        ts = base_time - timedelta(seconds=np.random.randint(0, 86400*30))
        rows.append({
            'transaction_id': f"txn_{i}",
            'user_id': user_id,
            'amount': round(amount,2),
            'hour': hour,
            'device_mobile': 1 if device_type=='mobile' else 0,
            'device_desktop': 1 if device_type=='desktop' else 0,
            'device_tablet': 1 if device_type=='tablet' else 0,
            'country': country,
            'ip_entropy': ip_entropy,
            'session_length': session_length,
            'items_in_cart': items_in_cart,
            'is_guest': is_guest,
            'speed_score': speed_score,
            'timestamp': ts
        })
    df = pd.DataFrame(rows)
    n_anom = max(10, n//200)
    anom_indices = np.random.choice(df.index, n_anom, replace=False)
    for idx in anom_indices:
        df.at[idx,'amount'] *= np.random.uniform(5,20)
        df.at[idx,'ip_entropy'] = np.random.uniform(0,0.01)
        df.at[idx,'session_length'] = np.random.uniform(1,5)
        df.at[idx,'items_in_cart'] = np.random.randint(10,50)
    df.reset_index(drop=True, inplace=True)
    return df

if __name__ == "__main__":
    df = simulate_transactions(5000)
    df.to_csv("sample_transactions.csv", index=False)
    print("Saved sample_transactions.csv with shape", df.shape)
