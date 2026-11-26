# [AI 協助] 本檔由 Claude（Anthropic，透過 Claude Code）協助撰寫，Leo 負責執行與驗證
# 第二種 train/val 切法（random_state=7），給 train_baseline.py --split split_b.csv 用
import pandas as pd
from sklearn.model_selection import train_test_split
c = pd.read_csv('data/stage_2_detailed_class_info.csv').drop_duplicates('patientId')
tr, va = train_test_split(c.patientId, test_size=0.2, stratify=c['class'], random_state=7)
s = pd.DataFrame({'patientId': pd.concat([tr, va]), 'split': ['train']*len(tr) + ['val']*len(va)}).merge(c, on='patientId')
s.to_csv('data/split_b.csv', index=False)
