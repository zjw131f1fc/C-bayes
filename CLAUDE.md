# 编码规范

## 环境

conda activate MCM

## 核心接口

```python
config = load_config(path)           # 加载YAML配置
datas = load_data(path)              # 加载数据字典
save_data(datas, path)               # 保存数据字典
```

## 统一函数签名

```python
datas = process_xxx(config, datas)   # 数据处理，链式更新datas
model = train(config, model, datas)  # 模型训练
datas = generate_xxx(config, model, datas)  # 模型推理
visualize_xxx(config, datas, tag)    # 可视化，输出到 outputs/{tag}/
```

## 示例

```python
config = load_config('config.yaml')
datas = {}

datas = load_raw_data(config, datas)
datas = preprocess(config, datas)
datas = build_features(config, datas)

model = load_model(config)
model = train(config, model, datas)
datas = predict(config, model, datas)

visualize_results(config, datas, 'results')
save_data(datas, 'output.pkl')
```
