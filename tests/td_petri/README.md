# Td_petri 测试套件

本目录包含 Td_petri 重构后的测试套件。

## 测试文件

### 单元测试
- `test_config.py` - PetriConfig 配置模块测试
- `test_interval_utils.py` - 区间工具函数测试
- `test_path_and_action.py` - PathRegistry 和 ActionSpaceBuilder 测试

### 集成测试
- `test_integration.py` - TimedPetri 集成测试，验证新模块协同工作

### 性能测试
- `test_performance.py` - 性能基准测试，确保无性能退化

## 运行测试

### 运行所有测试
```bash
pytest tests/td_petri/ -v
```

### 运行特定测试文件
```bash
pytest tests/td_petri/test_config.py -v
```

### 运行性能测试（显示输出）
```bash
pytest tests/td_petri/test_performance.py -v -s
```

### 运行集成测试
```bash
pytest tests/td_petri/test_integration.py -v
```

## 测试覆盖率

运行测试并生成覆盖率报告：
```bash
pytest tests/td_petri/ --cov=solutions.Td_petri --cov-report=html
```

## 测试组织

测试按照模块组织：
- **core/** 模块 → `test_config.py`
- **resources/** 模块 → `test_interval_utils.py`
- **rl/** 模块 → `test_path_and_action.py`
- **tdpn.py** 主文件 → `test_integration.py`
- 性能验证 → `test_performance.py`

## 预期结果

所有测试应该通过，性能测试应显示：
- 初始化时间 < 5秒
- Reset时间 < 1秒
- Step时间 < 0.5秒
- 观测构建 < 1ms
