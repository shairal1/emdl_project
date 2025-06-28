import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

stats_file = Path(__file__).parent / 'stats.json'
if not stats_file.exists():
    raise FileNotFoundError(f"Statistics file not found: {stats_file}")

with open(stats_file, 'r', encoding='utf-8') as f:
    stats = json.load(f)

df = pd.DataFrame(stats).T.reset_index().rename(columns={'index': 'Issue'})

summary_counts = {
    'Issues Detected': df['issue_detected'].sum(),
    'Issues Fixed': df['issue_fixed'].sum(),
    'False Positives': df['false_positive'].sum()
}

plt.rcParams['figure.dpi'] = 300

plt.figure()
plt.bar(summary_counts.keys(), summary_counts.values())
plt.title('OpenHands Evaluation Summary')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('summary_bar.png')
plt.close()


N = len(df)
detected = summary_counts['Issues Detected']
fixed = summary_counts['Issues Fixed']
false_pos = summary_counts['False Positives']

rates = [detected / N * 100, fixed / N * 100, false_pos / N * 100]
labels = ['Detection Rate', 'Fix Rate', 'False Positive Rate']

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
values = rates + [rates[0]]
angles += angles[:1]

fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, values, 'o-', linewidth=2)
ax.fill(angles, values, alpha=0.25)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_ylim(0, 100)
ax.set_title('OpenHands Agent Performance')
plt.tight_layout()
plt.savefig('summary_radar.png')
plt.close()