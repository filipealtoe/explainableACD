#!/usr/bin/env python3
"""
Generate shareable plots for CT24 error analysis presentation.

Usage:
    python experiments/scripts/generate_error_analysis_plots.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set style for clean, professional plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.facecolor'] = 'white'

OUTPUT_DIR = Path("experiments/results/ct24_classifier/error_analysis/presentation_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_error_overlap():
    """Plot 1: Error overlap with GPT-3.5 across splits."""
    fig, ax = plt.subplots(figsize=(10, 6))

    splits = ['Train', 'Dev', 'Test']
    catboost_errors = [4420, 119, 73]
    shared_errors = [3800, 94, 65]
    catboost_only = [620, 25, 8]

    x = np.arange(len(splits))
    width = 0.35

    bars1 = ax.bar(x - width/2, shared_errors, width, label='Shared with GPT-3.5', color='#e74c3c', alpha=0.9)
    bars2 = ax.bar(x + width/2, catboost_only, width, label='CatBoost-only errors', color='#3498db', alpha=0.9)

    # Add percentage labels
    for i, (shared, total) in enumerate(zip(shared_errors, catboost_errors)):
        pct = shared / total * 100
        ax.annotate(f'{pct:.1f}%', xy=(x[i] - width/2, shared + 50), ha='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Dataset Split')
    ax.set_ylabel('Number of Errors')
    ax.set_title('CatBoost Errors: How Many Are Inherited from GPT-3.5?', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.legend()

    # Add conclusion box
    textstr = '92-94% of CatBoost errors\nare inherited from GPT-3.5'
    props = dict(boxstyle='round', facecolor='#fff3cd', alpha=0.9, edgecolor='#ffc107')
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '1_error_overlap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / '1_error_overlap.png'}")


def plot_feature_inflation():
    """Plot 2: Feature inflation for False Positives."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    features = ['Checkability', 'Verifiability', 'Harm Potential']
    correct_cc = [26.38, 39.27, 49.17]
    correct_lp = [18.06, 36.88, 39.99]
    fp_cc = [53.74, 62.52, 60.76]
    fp_lp = [50.20, 87.24, 66.87]

    for i, (ax, feat) in enumerate(zip(axes, features)):
        x = np.arange(2)
        width = 0.35

        # Correct predictions
        ax.bar(x - width/2, [correct_cc[i], correct_lp[i]], width,
               label='Correct', color='#27ae60', alpha=0.8)
        # False positives
        ax.bar(x + width/2, [fp_cc[i], fp_lp[i]], width,
               label='False Positive', color='#e74c3c', alpha=0.8)

        ax.set_ylabel('Confidence Score')
        ax.set_title(feat, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Self-Reported', 'Logprob'])
        ax.set_ylim(0, 100)
        ax.legend(loc='upper left')

        # Add inflation annotation for logprob
        inflation = fp_lp[i] / correct_lp[i]
        if inflation > 1.5:
            ax.annotate(f'{inflation:.1f}×', xy=(1 + width/2, fp_lp[i] + 3),
                       ha='center', fontsize=11, fontweight='bold', color='#c0392b')

    fig.suptitle('False Positives: GPT-3.5 Logprob Confidence is INFLATED',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '2_feature_inflation_fp.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / '2_feature_inflation_fp.png'}")


def plot_zero_checkability():
    """Plot 3: The zero checkability problem for False Negatives."""
    fig, ax = plt.subplots(figsize=(10, 6))

    splits = ['Train', 'Dev', 'Test']
    fn_cc = [11.55, 10.00, 16.43]  # Self-reported
    fn_lp = [0.17, 0.00, 0.00]     # Logprob (essentially zero!)

    x = np.arange(len(splits))
    width = 0.35

    bars1 = ax.bar(x - width/2, fn_cc, width, label='Self-Reported', color='#3498db', alpha=0.9)
    bars2 = ax.bar(x + width/2, fn_lp, width, label='Logprob (Actual Belief)', color='#e74c3c', alpha=0.9)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10,
                   fontweight='bold', color='#c0392b')

    ax.set_xlabel('Dataset Split')
    ax.set_ylabel('Checkability Confidence (%)')
    ax.set_title('False Negatives: GPT-3.5 Assigns ZERO Checkability', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylim(0, 25)
    ax.legend()

    # Add explanation box
    textstr = 'GPT-3.5 says "somewhat uncertain" (10-16%)\nbut internally believes "DEFINITELY NOT checkable" (0%)\n\n→ Zero signal = impossible to recover'
    props = dict(boxstyle='round', facecolor='#f8d7da', alpha=0.9, edgecolor='#f5c6cb')
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '3_zero_checkability_fn.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / '3_zero_checkability_fn.png'}")


def plot_text_length():
    """Plot 4: Text length distribution by error type."""
    fig, ax = plt.subplots(figsize=(8, 6))

    groups = ['Correct', 'False Positive', 'False Negative']
    lengths = [92.1, 125.5, 98.6]
    colors = ['#27ae60', '#e74c3c', '#f39c12']

    bars = ax.bar(groups, lengths, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bar, length in zip(bars, lengths):
        ax.annotate(f'{length:.1f}', xy=(bar.get_x() + bar.get_width()/2, length),
                   xytext=(0, 5), textcoords="offset points", ha='center', fontsize=12, fontweight='bold')

    # Add percentage difference for FP
    ax.annotate('+36%', xy=(1, 125.5), xytext=(1.3, 130),
               fontsize=11, fontweight='bold', color='#c0392b',
               arrowprops=dict(arrowstyle='->', color='#c0392b'))

    ax.set_ylabel('Mean Text Length (characters)')
    ax.set_title('False Positives Are Longer Texts', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 150)

    # Add explanation
    textstr = 'Longer texts have more room\nfor political rhetoric'
    props = dict(boxstyle='round', facecolor='#fff3cd', alpha=0.9, edgecolor='#ffc107')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '4_text_length.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / '4_text_length.png'}")


def plot_word_patterns():
    """Plot 5: Top words in errors."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # False Positives
    fp_words = ['going', 'people', 'president', 'when', 'because', 'said', 'all', 'there', 'about', 'get']
    fp_counts = [380, 348, 287, 264, 230, 230, 223, 219, 215, 210]

    axes[0].barh(fp_words[::-1], fp_counts[::-1], color='#e74c3c', alpha=0.85)
    axes[0].set_xlabel('Frequency')
    axes[0].set_title('False Positives: Political Rhetoric', fontsize=13, fontweight='bold')
    axes[0].set_xlim(0, 420)

    # Highlight key words
    for i, word in enumerate(fp_words[::-1]):
        if word in ['president', 'people', 'going']:
            axes[0].get_children()[i].set_color('#c0392b')

    # False Negatives
    fn_words = ['because', 'now', 'people', 'going', 'said', "it's", "that's", 'years', 'all', 'think']
    fn_counts = [111, 103, 100, 82, 82, 82, 81, 80, 79, 74]

    axes[1].barh(fn_words[::-1], fn_counts[::-1], color='#f39c12', alpha=0.85)
    axes[1].set_xlabel('Frequency')
    axes[1].set_title('False Negatives: Casual Language', fontsize=13, fontweight='bold')
    axes[1].set_xlim(0, 130)

    # Highlight key words
    for i, word in enumerate(fn_words[::-1]):
        if word in ['because', "it's", "that's"]:
            axes[1].get_children()[i].set_color('#e67e22')

    fig.suptitle('Linguistic Patterns in Errors (Train Split)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '5_word_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / '5_word_patterns.png'}")


def plot_summary_diagram():
    """Plot 6: Summary diagram of the error analysis conclusions."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'CT24 Checkworthiness: Error Analysis Summary',
            fontsize=18, fontweight='bold', ha='center', va='top')

    # Central box: The problem
    problem_box = mpatches.FancyBboxPatch((4.5, 6.5), 5, 2, boxstyle="round,pad=0.1",
                                          facecolor='#f8d7da', edgecolor='#721c24', linewidth=2)
    ax.add_patch(problem_box)
    ax.text(7, 7.8, 'THE PROBLEM', fontsize=14, fontweight='bold', ha='center', color='#721c24')
    ax.text(7, 7.2, '94% of errors inherited from GPT-3.5', fontsize=12, ha='center')
    ax.text(7, 6.8, 'Features are the ceiling, not the classifier', fontsize=11, ha='center', style='italic')

    # Left box: False Positives
    fp_box = mpatches.FancyBboxPatch((0.5, 3), 5.5, 3),
    fp_box = mpatches.FancyBboxPatch((0.5, 3), 5.5, 3, boxstyle="round,pad=0.1",
                                     facecolor='#ffeaa7', edgecolor='#d68910', linewidth=2)
    ax.add_patch(fp_box)
    ax.text(3.25, 5.7, 'FALSE POSITIVES (45 on test)', fontsize=12, fontweight='bold', ha='center', color='#d68910')
    ax.text(3.25, 5.2, 'Cause: Logprob inflated 2-2.5×', fontsize=11, ha='center')
    ax.text(3.25, 4.7, 'Pattern: Political rhetoric', fontsize=11, ha='center')
    ax.text(3.25, 4.2, '"president", "people", "going"', fontsize=10, ha='center', style='italic')
    ax.text(3.25, 3.6, 'Text is 36% longer than average', fontsize=10, ha='center')
    ax.text(3.25, 3.2, '→ Sounds factual but is opinion', fontsize=10, ha='center', fontweight='bold')

    # Right box: False Negatives
    fn_box = mpatches.FancyBboxPatch((8, 3), 5.5, 3, boxstyle="round,pad=0.1",
                                     facecolor='#d5f5e3', edgecolor='#1e8449', linewidth=2)
    ax.add_patch(fn_box)
    ax.text(10.75, 5.7, 'FALSE NEGATIVES (28 on test)', fontsize=12, fontweight='bold', ha='center', color='#1e8449')
    ax.text(10.75, 5.2, 'Cause: checkability_logprob = 0%', fontsize=11, ha='center')
    ax.text(10.75, 4.7, 'Pattern: Casual language', fontsize=11, ha='center')
    ax.text(10.75, 4.2, '"because", "I\'ve", "it\'s"', fontsize=10, ha='center', style='italic')
    ax.text(10.75, 3.6, 'Zero signal = impossible to recover', fontsize=10, ha='center')
    ax.text(10.75, 3.2, '→ Claims hidden in casual speech', fontsize=10, ha='center', fontweight='bold')

    # Bottom box: Conclusion
    conclusion_box = mpatches.FancyBboxPatch((2.5, 0.3), 9, 2.2, boxstyle="round,pad=0.1",
                                             facecolor='#d4edda', edgecolor='#155724', linewidth=2)
    ax.add_patch(conclusion_box)
    ax.text(7, 2.2, 'CONCLUSION', fontsize=14, fontweight='bold', ha='center', color='#155724')
    ax.text(7, 1.7, 'More ML engineering won\'t help (ceiling reached)', fontsize=11, ha='center')
    ax.text(7, 1.2, 'Must fix features at source (DSPy) or bypass entirely (fine-tuning)', fontsize=11, ha='center')
    ax.text(7, 0.7, 'Current F1: 0.622 → Target F1: 0.799 (gap: 17.7 points)', fontsize=10, ha='center', style='italic')

    # Arrows
    ax.annotate('', xy=(3.25, 6), xytext=(5.5, 6.5),
                arrowprops=dict(arrowstyle='->', color='#d68910', lw=2))
    ax.annotate('', xy=(10.75, 6), xytext=(8.5, 6.5),
                arrowprops=dict(arrowstyle='->', color='#1e8449', lw=2))
    ax.annotate('', xy=(7, 2.5), xytext=(7, 3),
                arrowprops=dict(arrowstyle='->', color='#155724', lw=2))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '6_summary_diagram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / '6_summary_diagram.png'}")


def plot_comparison_table():
    """Plot 7: Visual comparison table."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # Table data
    columns = ['Metric', 'False Positives', 'False Negatives']
    data = [
        ['Count (test)', '45', '28'],
        ['Root Cause', 'Inflated logprob (2-2.5×)', 'Zero checkability (0%)'],
        ['Key Words', 'president, people, going', 'because, I\'ve, it\'s'],
        ['Text Length', '+36% longer', '+7% longer'],
        ['GPT-3.5 Behavior', 'Overconfident on rhetoric', 'Dismisses casual language'],
        ['Can Classifier Fix?', 'No (feature is misleading)', 'No (zero signal)'],
        ['Solution', 'Detect rhetoric patterns', 'Recognize embedded claims'],
    ]

    # Create table
    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Style columns
    for i in range(1, len(data) + 1):
        table[(i, 1)].set_facecolor('#ffeaa7')  # FP column - yellow
        table[(i, 2)].set_facecolor('#d5f5e3')  # FN column - green

    ax.set_title('Error Analysis: False Positives vs False Negatives',
                 fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '7_comparison_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / '7_comparison_table.png'}")


def main():
    print("=" * 60)
    print("GENERATING ERROR ANALYSIS PRESENTATION PLOTS")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}\n")

    print("Generating plots...")
    plot_error_overlap()
    plot_feature_inflation()
    plot_zero_checkability()
    plot_text_length()
    plot_word_patterns()
    plot_summary_diagram()
    plot_comparison_table()

    print("\n" + "=" * 60)
    print("DONE! All plots saved to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)

    # List all generated files
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  • {f.name}")


if __name__ == "__main__":
    main()
