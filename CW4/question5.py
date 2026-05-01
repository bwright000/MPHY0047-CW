# Question 5: Association rule mining on the heart-statlog CVD dataset.
# Binarise 13 features per spec, one-hot encode both values of each feature
# plus both class values, run Apriori, extract rules by lift and conviction.

import os
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from dataloader import heart_data
from plot_style import apply_style, BLUE, finish_figure
from timeseries import FIGURES_DIR  # shared figures dir

apply_style()


# Friendly short labels so the rule text is readable.
# Each tuple is (column_name, label_when_0, label_when_1).
FEATURE_LABELS = [
    ('age',                                    'AGE_YOUNG',      'AGE_OLD'),
    ('sex',                                    'SEX_F',          'SEX_M'),
    ('chest',                                  'CHEST_LOW',      'CHEST_HIGH'),
    ('resting_blood_pressure',                 'BP_NORMAL',      'BP_HIGH'),
    ('serum_cholestoral',                      'CHOL_NORMAL',    'CHOL_HIGH'),
    ('fasting_blood_sugar',                    'FBS_NORMAL',     'FBS_HIGH'),
    ('resting_electrocardiographic_results',   'RECG_NORMAL',    'RECG_ABNORMAL'),
    ('maximum_heart_rate_achieved',            'MAXHR_LOW',      'MAXHR_HIGH'),
    ('exercise_induced_angina',                'ANGINA_NO',      'ANGINA_YES'),
    ('oldpeak',                                'OLDPEAK_ZERO',   'OLDPEAK_POS'),
    ('slope',                                  'SLOPE_UP',       'SLOPE_NOTUP'),
    ('number_of_major_vessels',                'VESSELS_ZERO',   'VESSELS_POS'),
    ('thal',                                   'THAL_NORMAL',    'THAL_ABNORMAL'),
]


def binarise():
    """
    Apply the binarisation rules from the spec to the 13 features.
    Returns the boolean-per-feature frame (0/1) and the class series.
    """
    df = heart_data.copy()

    b = pd.DataFrame(index=df.index)
    b['age']                                   = (df['age'] > 50).astype(int)
    b['sex']                                   = df['sex'].astype(int)
    b['chest']                                 = (df['chest'] > 2.5).astype(int)
    b['resting_blood_pressure']                = (df['resting_blood_pressure'] > 125).astype(int)
    b['serum_cholestoral']                     = (df['serum_cholestoral'] > 250).astype(int)
    b['fasting_blood_sugar']                   = df['fasting_blood_sugar'].astype(int)
    b['resting_electrocardiographic_results']  = (df['resting_electrocardiographic_results'] != 0).astype(int)
    b['maximum_heart_rate_achieved']           = (df['maximum_heart_rate_achieved'] > 140).astype(int)
    b['exercise_induced_angina']               = df['exercise_induced_angina'].astype(int)
    b['oldpeak']                               = (df['oldpeak'] != 0).astype(int)
    # Slope: 0 if value=1 else 1
    b['slope']                                 = (df['slope'] != 1).astype(int)
    b['number_of_major_vessels']               = (df['number_of_major_vessels'] != 0).astype(int)
    b['thal']                                  = (df['thal'] != 3).astype(int)

    # Sanity: every column must be 0/1
    assert set(b.values.flatten().tolist()) <= {0, 1}, "Binarisation produced non-0/1 values"

    return b, df['class']


def one_hot_encode(binarised):
    """
    Expand each binary feature into TWO items (value=0 and value=1) so
    Apriori can surface rules about BOTH states of a feature.

    The class column is deliberately excluded from the basket. The spec's
    binarisation list covers only the 13 features; the class enters at the
    interpretation step via class-stratified support (see class_stratify).
    """
    items = pd.DataFrame(index=binarised.index)
    for col, lbl0, lbl1 in FEATURE_LABELS:
        items[lbl0] = (binarised[col] == 0)
        items[lbl1] = (binarised[col] == 1)
    return items


def format_itemset(frozenset_obj):
    """Render a frozenset of items as a sorted comma-separated string."""
    return ", ".join(sorted(frozenset_obj))


# Map an item label (e.g. AGE_OLD) back to (column, value) for stratification.
_LABEL_TO_COLVAL = {}
for _col, _lbl0, _lbl1 in FEATURE_LABELS:
    _LABEL_TO_COLVAL[_lbl0] = (_col, 0)
    _LABEL_TO_COLVAL[_lbl1] = (_col, 1)


def _itemset_mask(itemset, binarised):
    """Boolean Series: True for rows that match every item in the itemset."""
    mask = pd.Series(True, index=binarised.index)
    for label in itemset:
        col, val = _LABEL_TO_COLVAL[label]
        mask &= (binarised[col] == val)
    return mask


def class_stratify(rules, binarised, class_series):
    """
    For each rule, compute how strongly its itemset (antecedent ∪ consequent)
    concentrates in CVD-present vs CVD-absent patients.

    class_lift_X = P(class=X | itemset) / P(class=X)
                 = (mask & (class == X)).sum() / (mask.sum() * P(X))

    A value > 1 means the itemset is enriched in class X vs the baseline
    prevalence; < 1 means depleted. direction = whichever class_lift is larger.
    """
    p_absent = float((class_series == 'absent').mean())
    p_present = float((class_series == 'present').mean())
    is_absent = (class_series == 'absent').values
    is_present = (class_series == 'present').values

    enriched = rules.copy()
    lifts_absent = []
    lifts_present = []
    directions = []
    strengths = []

    for _, r in rules.iterrows():
        itemset = set(r['antecedents']) | set(r['consequents'])
        mask = _itemset_mask(itemset, binarised).values
        n_match = int(mask.sum())
        if n_match == 0:
            la = lp = 0.0
        else:
            la = float((mask & is_absent).sum()) / (n_match * p_absent)
            lp = float((mask & is_present).sum()) / (n_match * p_present)
        lifts_absent.append(la)
        lifts_present.append(lp)
        if la >= lp:
            directions.append('absent')
            strengths.append(la)
        else:
            directions.append('present')
            strengths.append(lp)

    enriched['class_lift_absent'] = lifts_absent
    enriched['class_lift_present'] = lifts_present
    enriched['direction'] = directions
    enriched['strength'] = strengths
    return enriched


def run():
    print("\n" + "=" * 72)
    print("  Q5 Step 1: Binarisation (13 features + class)")
    print("=" * 72)
    binarised, classes = binarise()
    print(f"  Dataset: {len(binarised)} patients, {binarised.shape[1]} features")
    print(f"  Class balance: {dict(classes.value_counts())}")
    print(f"  Binarisation summary (fraction = 1 per feature):")
    for col, _, lbl1 in FEATURE_LABELS:
        frac = binarised[col].mean()
        print(f"    {lbl1:<18} ({col:<38}) fraction=1: {frac:.3f}")

    print("\n" + "=" * 72)
    print("  Q5 Step 2: One-Hot Encoding -> Apriori Basket (features only)")
    print("=" * 72)
    items = one_hot_encode(binarised)
    print(f"  Basket: {items.shape[0]} transactions x {items.shape[1]} items")
    print(f"  (Class column is NOT in the basket - it enters via stratification.)")

    print("\n" + "=" * 72)
    print("  Q5 Step 3: Apriori (min_support = 0.25)")
    print("=" * 72)
    freq = apriori(items, min_support=0.25, use_colnames=True)
    freq = freq.sort_values('support', ascending=False).reset_index(drop=True)
    print(f"  Frequent itemsets found: {len(freq)}")
    print(f"  Max itemset size: {freq['itemsets'].apply(len).max()}")
    print(f"  Top 10 most-supported itemsets:")
    for _, row in freq.head(10).iterrows():
        print(f"    support={row['support']:.3f}  {{{format_itemset(row['itemsets'])}}}")

    print("\n" + "=" * 72)
    print("  Q5 Step 4: Association Rules (lift >= 1.15)")
    print("=" * 72)
    rules = association_rules(freq, metric='lift', min_threshold=1.15)
    cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift', 'conviction']
    rules = rules[cols].copy()
    rules = rules.sort_values(['conviction', 'lift'], ascending=False).reset_index(drop=True)
    print(f"  Rules with lift >= 1.15: {len(rules)}")

    print("\n" + "=" * 72)
    print("  Q5 Step 5: Filter Most Significant (conviction > 1.5)")
    print("=" * 72)
    significant = rules[rules['conviction'] > 1.5].reset_index(drop=True)
    print(f"  Rules with conviction > 1.5: {len(significant)}")

    print("\n" + "=" * 72)
    print("  Q5 Step 6: Class-Stratified Interpretation")
    print("=" * 72)
    stratified = class_stratify(significant, binarised, classes)
    n_absent = int((stratified['direction'] == 'absent').sum())
    n_present = int((stratified['direction'] == 'present').sum())
    print(f"  Rules indicative of CVD ABSENCE:  {n_absent}")
    print(f"  Rules indicative of CVD PRESENCE: {n_present}")

    def _print_top(label, subset, k=10):
        subset = subset.sort_values(
            ['strength', 'conviction'], ascending=False
        ).head(k)
        print(f"\n  Top {len(subset)} rules indicative of CVD {label} "
              f"(sorted by class-lift):")
        hdr = (f"  {'Antecedents':<30} -> {'Consequents':<22} "
               f"{'Sup':>5} {'Conf':>5} {'Lift':>5} {'Conv':>6} "
               f"{'L_abs':>6} {'L_pres':>6}")
        print(hdr)
        for _, r in subset.iterrows():
            ant = format_itemset(r['antecedents'])
            cons = format_itemset(r['consequents'])
            ant_s = (ant[:27] + '...') if len(ant) > 30 else ant
            cons_s = (cons[:19] + '...') if len(cons) > 22 else cons
            print(f"  {ant_s:<30} -> {cons_s:<22} "
                  f"{r['support']:>5.2f} {r['confidence']:>5.2f} "
                  f"{r['lift']:>5.2f} {r['conviction']:>6.2f} "
                  f"{r['class_lift_absent']:>6.2f} "
                  f"{r['class_lift_present']:>6.2f}")

    absent_rules = stratified[stratified['direction'] == 'absent']
    present_rules = stratified[stratified['direction'] == 'present']
    _print_top('ABSENT', absent_rules)
    _print_top('PRESENT', present_rules)

    return freq, rules, significant, stratified


freq_itemsets, all_rules, sig_rules, stratified_rules = run()


if __name__ == '__main__':
    print("\n" + "=" * 72)
    print("  Q5: Association Rule Mining — Summary")
    print("=" * 72)
    print(f"  Frequent itemsets (sup >= 0.25):   {len(freq_itemsets)}")
    print(f"  Rules (lift >= 1.15):              {len(all_rules)}")
    print(f"  Significant rules (conv > 1.5):    {len(sig_rules)}")
    n_abs = int((stratified_rules['direction'] == 'absent').sum())
    n_pre = int((stratified_rules['direction'] == 'present').sum())
    print(f"    ...indicative of ABSENCE:        {n_abs}")
    print(f"    ...indicative of PRESENCE:       {n_pre}")
