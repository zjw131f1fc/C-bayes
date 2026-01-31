"""数据完整性检查脚本"""

import pickle
import numpy as np
from collections import Counter

def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def check_basic_info(td):
    """检查基本信息"""
    print("=" * 60)
    print("1. BASIC INFO")
    print("=" * 60)
    print(f"n_obs: {td['n_obs']}")
    print(f"n_weeks: {td['n_weeks']}")
    print(f"n_celebs: {td['n_celebs']}")
    print(f"n_pros: {td['n_pros']}")
    print(f"n_seasons: {td['n_seasons']}")
    print()

def check_index_arrays(td):
    """检查索引数组"""
    print("=" * 60)
    print("2. INDEX ARRAYS")
    print("=" * 60)

    # celeb_idx
    celeb_idx = td['celeb_idx']
    print(f"celeb_idx: shape={celeb_idx.shape}, dtype={celeb_idx.dtype}")
    print(f"  range: [{celeb_idx.min()}, {celeb_idx.max()}], expected: [0, {td['n_celebs']-1}]")
    if celeb_idx.min() < 0 or celeb_idx.max() >= td['n_celebs']:
        print("  ❌ ERROR: celeb_idx out of range!")
    else:
        print("  ✓ OK")

    # pro_idx
    pro_idx = td['pro_idx']
    print(f"pro_idx: shape={pro_idx.shape}, dtype={pro_idx.dtype}")
    print(f"  range: [{pro_idx.min()}, {pro_idx.max()}], expected: [0, {td['n_pros']-1}]")
    if pro_idx.min() < 0 or pro_idx.max() >= td['n_pros']:
        print("  ❌ ERROR: pro_idx out of range!")
    else:
        print("  ✓ OK")

    # week_idx
    week_idx = td['week_idx']
    print(f"week_idx: shape={week_idx.shape}, dtype={week_idx.dtype}")
    print(f"  range: [{week_idx.min()}, {week_idx.max()}], expected: [0, {td['n_weeks']-1}]")
    if week_idx.min() < 0 or week_idx.max() >= td['n_weeks']:
        print("  ❌ ERROR: week_idx out of range!")
    else:
        print("  ✓ OK")
    print()

def check_feature_matrices(td):
    """检查特征矩阵"""
    print("=" * 60)
    print("3. FEATURE MATRICES")
    print("=" * 60)

    for name, expected_rows in [('X_celeb', td['n_celebs']),
                                 ('X_pro', td['n_pros']),
                                 ('X_obs', td['n_obs'])]:
        X = td[name]
        names = td[f'{name}_names']
        print(f"{name}: shape={X.shape}, dtype={X.dtype}")
        print(f"  expected rows: {expected_rows}")
        if X.shape[0] != expected_rows:
            print(f"  ❌ ERROR: row count mismatch!")

        # Check for NaN/Inf
        n_nan = np.isnan(X).sum()
        n_inf = np.isinf(X).sum()
        if n_nan > 0 or n_inf > 0:
            print(f"  ❌ ERROR: NaN={n_nan}, Inf={n_inf}")
        else:
            print(f"  ✓ No NaN/Inf")

        # Check feature stats
        print(f"  Features ({len(names)}):")
        for i, fname in enumerate(names):
            col = X[:, i]
            unique = np.unique(col)
            if len(unique) <= 10:
                print(f"    {fname}: unique={list(unique)}")
            else:
                print(f"    {fname}: min={col.min():.3f}, max={col.max():.3f}, mean={col.mean():.3f}, std={col.std():.3f}")
    print()

def check_week_data(td):
    """检查周级数据"""
    print("=" * 60)
    print("4. WEEK DATA")
    print("=" * 60)

    week_data = td['week_data']
    print(f"Number of weeks: {len(week_data)}")

    total_contestants = 0
    total_eliminated = 0
    weeks_with_elim = 0

    errors = []

    for w, wd in enumerate(week_data):
        mask = wd['obs_mask']
        n_contestants = wd['n_contestants']
        n_eliminated = wd['n_eliminated']
        elim_mask = wd['eliminated_mask']

        # Check consistency
        actual_contestants = mask.sum()
        actual_eliminated = elim_mask.sum()

        if actual_contestants != n_contestants:
            errors.append(f"Week {w}: obs_mask.sum()={actual_contestants} != n_contestants={n_contestants}")

        if actual_eliminated != n_eliminated:
            errors.append(f"Week {w}: eliminated_mask.sum()={actual_eliminated} != n_eliminated={n_eliminated}")

        # Check eliminated are subset of contestants
        if not np.all(elim_mask <= mask):
            errors.append(f"Week {w}: eliminated not subset of contestants!")

        total_contestants += n_contestants
        total_eliminated += n_eliminated
        if n_eliminated > 0:
            weeks_with_elim += 1

    print(f"Total contestants (sum): {total_contestants}")
    print(f"Total eliminated: {total_eliminated}")
    print(f"Weeks with elimination: {weeks_with_elim}")
    print(f"Average contestants per week: {total_contestants / len(week_data):.1f}")

    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    else:
        print("\n✓ All week data consistent")
    print()

def check_judge_scores(td):
    """检查评委分数"""
    print("=" * 60)
    print("5. JUDGE SCORES")
    print("=" * 60)

    judge_pct = td['judge_score_pct']
    judge_rank = td['judge_rank_score']

    print(f"judge_score_pct: shape={judge_pct.shape}")
    print(f"  range: [{judge_pct.min():.3f}, {judge_pct.max():.3f}]")
    print(f"  NaN: {np.isnan(judge_pct).sum()}")

    print(f"judge_rank_score: shape={judge_rank.shape}")
    print(f"  range: [{judge_rank.min():.3f}, {judge_rank.max():.3f}]")
    print(f"  NaN: {np.isnan(judge_rank).sum()}")
    print()

def check_feature_discrimination(td):
    """检查特征区分能力"""
    print("=" * 60)
    print("6. FEATURE DISCRIMINATION (Eliminated vs Survived)")
    print("=" * 60)

    X_obs = td['X_obs']
    X_obs_names = td['X_obs_names']
    week_data = td['week_data']

    # Collect eliminated and survived features
    elim_features = []
    surv_features = []

    for wd in week_data:
        if wd['n_eliminated'] == 0:
            continue
        mask = wd['obs_mask']
        elim_mask = wd['eliminated_mask']

        indices = np.where(mask)[0]
        for idx in indices:
            if elim_mask[idx]:
                elim_features.append(X_obs[idx])
            else:
                surv_features.append(X_obs[idx])

    elim_features = np.array(elim_features)
    surv_features = np.array(surv_features)

    print(f"Eliminated: {len(elim_features)}, Survived: {len(surv_features)}")
    print()
    print("Feature discrimination (elim_mean - surv_mean):")
    print("-" * 50)

    diffs = []
    for i, name in enumerate(X_obs_names):
        elim_mean = elim_features[:, i].mean()
        surv_mean = surv_features[:, i].mean()
        diff = elim_mean - surv_mean
        diffs.append((abs(diff), name, diff, elim_mean, surv_mean))

    diffs.sort(reverse=True)
    for _, name, diff, elim_mean, surv_mean in diffs:
        marker = "***" if abs(diff) > 0.5 else "**" if abs(diff) > 0.2 else "*" if abs(diff) > 0.1 else ""
        print(f"  {name:25s}: diff={diff:+.3f} {marker}")
    print()

def check_data_leakage(td):
    """检查数据泄露"""
    print("=" * 60)
    print("7. DATA LEAKAGE CHECK")
    print("=" * 60)

    X_obs = td['X_obs']
    X_obs_names = td['X_obs_names']
    week_data = td['week_data']

    # Check if any feature perfectly predicts elimination
    suspicious = []

    for i, name in enumerate(X_obs_names):
        # For each week, check if feature perfectly separates eliminated from survived
        perfect_weeks = 0
        total_weeks = 0

        for wd in week_data:
            if wd['n_eliminated'] == 0:
                continue
            total_weeks += 1

            mask = wd['obs_mask']
            elim_mask = wd['eliminated_mask']
            indices = np.where(mask)[0]

            elim_vals = X_obs[indices[elim_mask[indices]], i]
            surv_vals = X_obs[indices[~elim_mask[indices]], i]

            if len(elim_vals) > 0 and len(surv_vals) > 0:
                # Check if ranges don't overlap
                if elim_vals.max() < surv_vals.min() or elim_vals.min() > surv_vals.max():
                    perfect_weeks += 1

        if perfect_weeks > total_weeks * 0.5:  # More than 50% perfect separation
            suspicious.append((name, perfect_weeks, total_weeks))

    if suspicious:
        print("⚠️  SUSPICIOUS FEATURES (may have data leakage):")
        for name, perfect, total in suspicious:
            print(f"  {name}: perfect separation in {perfect}/{total} weeks ({100*perfect/total:.1f}%)")
    else:
        print("✓ No obvious data leakage detected")
    print()

def check_celeb_pro_features(td):
    """检查名人和舞者特征"""
    print("=" * 60)
    print("8. CELEBRITY & PRO FEATURES")
    print("=" * 60)

    X_celeb = td['X_celeb']
    X_celeb_names = td['X_celeb_names']

    print(f"X_celeb: {X_celeb.shape}")
    print("Celebrity features:")
    for i, name in enumerate(X_celeb_names):
        col = X_celeb[:, i]
        unique = np.unique(col)
        if len(unique) <= 5:
            print(f"  {name}: unique={list(unique)}")
        else:
            print(f"  {name}: min={col.min():.2f}, max={col.max():.2f}, mean={col.mean():.2f}")

    print()

    X_pro = td['X_pro']
    X_pro_names = td['X_pro_names']

    print(f"X_pro: {X_pro.shape}")
    print("Pro features:")
    for i, name in enumerate(X_pro_names):
        col = X_pro[:, i]
        unique = np.unique(col)
        if len(unique) <= 5:
            print(f"  {name}: unique={list(unique)}")
        else:
            print(f"  {name}: min={col.min():.2f}, max={col.max():.2f}, mean={col.mean():.2f}")
    print()

def main():
    print("Loading data...")
    datas = load_data('datas.pkl')
    td = datas['train_data']

    print("\n" + "=" * 60)
    print("DATA INTEGRITY CHECK")
    print("=" * 60 + "\n")

    check_basic_info(td)
    check_index_arrays(td)
    check_feature_matrices(td)
    check_week_data(td)
    check_judge_scores(td)
    check_feature_discrimination(td)
    check_data_leakage(td)
    check_celeb_pro_features(td)

    print("=" * 60)
    print("CHECK COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
