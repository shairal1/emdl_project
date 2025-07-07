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

# Set DPI to 500 for high quality
plt.rcParams['figure.dpi'] = 500

# 1. Summary Bar Chart
plt.figure(figsize=(10, 6))
bars = plt.bar(summary_counts.keys(), summary_counts.values(), 
               color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8)
plt.title('OpenHands Evaluation Summary', fontsize=16, fontweight='bold')
plt.ylabel('Count', fontsize=12)

# Add value labels on bars
for bar, value in zip(bars, summary_counts.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            str(value), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('summary_bar.png', dpi=500, bbox_inches='tight')
plt.close()

# 2. Radar Chart
N = len(df)
detected = summary_counts['Issues Detected']
fixed = summary_counts['Issues Fixed']
false_pos = summary_counts['False Positives']

rates = [detected / N * 100, fixed / N * 100, false_pos / N * 100]
labels = ['Detection Rate', 'Fix Rate', 'False Positive Rate']

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
values = rates + [rates[0]]
angles += angles[:1]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, values, 'o-', linewidth=2, color='#2E86AB')
ax.fill(angles, values, alpha=0.25, color='#2E86AB')
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_ylim(0, 100)
ax.set_title('OpenHands Agent Performance', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('summary_radar.png', dpi=500, bbox_inches='tight')
plt.close()

# 3. Line Changes Bar Chart
# Filter out entries with errors and extract line changes
line_changes_data = []
for issue, data in stats.items():
    if isinstance(data, dict) and 'lines_changed' in data:
        line_changes_data.append((issue, data['lines_changed']))

if line_changes_data:
    # Sort by number of lines changed (descending)
    line_changes_data.sort(key=lambda x: x[1], reverse=True)
    issues, lines_changed = zip(*line_changes_data)
    
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(issues)), lines_changed, 
                   color='steelblue', alpha=0.7, edgecolor='navy')
    
    # Customize the chart
    plt.title('Number of Lines Changed by AI per Pipeline', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Pipeline', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Lines Changed', fontsize=12, fontweight='bold')
    
    # Set x-axis labels
    plt.xticks(range(len(issues)), issues, rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i, (bar, value) in enumerate(zip(bars, lines_changed)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(value), ha='center', va='bottom', fontweight='bold')
    
    # Add average line
    if lines_changed:
        avg_lines = np.mean(lines_changed)
        plt.axhline(y=avg_lines, color='red', linestyle='--', alpha=0.8, linewidth=2)
        plt.text(len(issues)-1, avg_lines + 0.5, f'Average: {avg_lines:.1f}', 
                ha='right', va='bottom', color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the chart at 500 DPI
    plt.savefig('line_changes_by_pipeline.png', dpi=500, bbox_inches='tight')
    plt.close()
    
    print(f"Line changes bar chart saved to line_changes_by_pipeline.png")

# Print summary statistics
print("\n" + "="*60)
print("OPENHANDS EVALUATION SUMMARY")
print("="*60)
print(f"Total pipelines analyzed: {N}")
print(f"Issues detected: {detected} ({detected/N*100:.1f}%)")
print(f"Issues fixed: {fixed} ({fixed/N*100:.1f}%)")
print(f"False positives: {false_pos} ({false_pos/N*100:.1f}%)")

if line_changes_data:
    total_lines = sum(lines_changed)
    avg_lines = np.mean(lines_changed)
    print(f"\nTotal lines changed: {total_lines}")
    print(f"Average lines changed per pipeline: {avg_lines:.2f}")
    
    # Show top 5 pipelines by lines changed
    print(f"\nTop 5 pipelines by lines changed:")
    for i, (issue, lines) in enumerate(line_changes_data[:5], 1):
        print(f"  {i}. {issue}: {lines} lines")

print("="*60)
print("Charts saved:")
print("- summary_bar.png (500 DPI)")
print("- summary_radar.png (500 DPI)")
print("- line_changes_by_pipeline.png (500 DPI)")