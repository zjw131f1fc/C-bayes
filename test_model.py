"""测试模型是否能正常运行 (NumPyro 版本)"""
import sys
print("Step 1: Importing...")
sys.stdout.flush()

from src.utils import load_config
from src.preprocess import load_data, validate_data, filter_data
from src.model import build_model, train

print("Step 2: Loading config and data...")
sys.stdout.flush()

config = load_config('config.yaml')
datas = {}
datas = load_data('datas.pkl', datas)

print("Step 3: Filtering data...")
sys.stdout.flush()

train_datas, test_datas = filter_data(datas, 0)
print(f"  train: {train_datas['train_data']['n_obs']} obs")

print("Step 4: Building model...")
sys.stdout.flush()

model_params = build_model(config, train_datas)
print(f"  Model params ready: {list(model_params.keys())}")

print("Step 5: Testing MCMC (10 samples)...")
sys.stdout.flush()

train_datas = train(config, model_params, train_datas)
print("  MCMC completed!")

print("Step 6: Checking results...")
sys.stdout.flush()
samples = train_datas['trace']
print(f"  Sample keys: {list(samples.keys())}")
print(f"  alpha shape: {samples['alpha'].shape}")

print("All tests passed!")
