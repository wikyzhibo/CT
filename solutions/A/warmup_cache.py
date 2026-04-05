import json
from solutions.A.model_builder import precompute_all_routes

with open(r"D:\Code\CT\config\cluster_tool\route_config.json", encoding="utf-8") as f:
    route_config = json.load(f)

results = precompute_all_routes(
    route_config=route_config,
    n_wafer=10,          # 与训练时一致
    ttime=5,
    cleaning_enabled=False,
    p_residual_time=15,
    d_residual_time=10,
)

for route_name, was_miss in results.items():
    status = "计算并缓存" if was_miss else "已命中缓存（跳过）"
    print(f"  {route_name}: {status}")