"""测试模型是否能正常运行"""
import sys
print("Step 1: Importing...")
sys.stdout.flush()

from src.utils import load_config
from src.preprocess import load_data, validate_data, filter_data
from src.model import build_model

print("Step 2: Loading config and data...")
sys.stdout.flush()

config = load_config('config.yaml')
datas = {}
datas = load_data('datas(1).pkl', datas)

print("Step 3: Filtering data...")
sys.stdout.flush()

train_datas, test_datas = filter_data(datas, 0)
print(f"  train: {train_datas['train_data']['n_obs']} obs")

print("Step 4: Building model...")
sys.stdout.flush()

model = build_model(config, train_datas)
print(f"  Model built: {len(model.free_RVs)} free RVs")

print("Step 5: Testing model compilation...")
sys.stdout.flush()

import pymc as pm
with model:
    # 先测试先验采样
    print("  Testing prior predictive...")
    sys.stdout.flush()
    prior = pm.sample_prior_predictive(samples=10)
    print("  Prior predictive returned")
    sys.stdout.flush()
    print(f"  Prior keys: {list(prior.prior.data_vars.keys())[:5]}...")
    sys.stdout.flush()

print("Step 6: Testing MCMC (1 sample only)...")
sys.stdout.flush()

with model:
    trace = pm.sample(draws=1, tune=1, chains=1, cores=1, progressbar=True)
    print("  MCMC OK!")

print("All tests passed!")
