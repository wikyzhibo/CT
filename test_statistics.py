"""
测试统计接口功能
"""
import sys
sys.path.insert(0, r"c:\Users\khand\OneDrive\code\dqn\CT")

from solutions.PPO.enviroment import Env_PN
from solutions.Continuous_model.pn import Petri

def test_statistics_disabled():
    """测试训练模式（统计关闭）"""
    print("=" * 60)
    print("测试1: 训练模式 (enable_statistics=False)")
    print("=" * 60)
    
    net = Petri(enable_statistics=False)
    net.reset()
    
    # 执行几步
    for _ in range(5):
        enabled = net.get_enable_t()
        if enabled:
            net.step(t=enabled[0], wait=False)
    
    # 调用统计方法
    stats = net.calc_wafer_statistics()
    
    print(f"统计结果: {stats}")
    print(f"预期: 空字典 {{}}")
    print(f"✓ 通过" if stats == {} else "✗ 失败")
    print()

def test_statistics_enabled():
    """测试可视化模式（统计开启）"""
    print("=" * 60)
    print("测试2: 可视化模式 (enable_statistics=True)")
    print("=" * 60)
    
    net = Petri(enable_statistics=True)
    net.reset()
    
    # 执行几步
    for _ in range(10):
        enabled = net.get_enable_t()
        if enabled:
            net.step(t=enabled[0], wait=False)
        else:
            net.step(wait=True)
    
    # 调用统计方法
    stats = net.calc_wafer_statistics()
    
    print("统计结果:")
    print(f"  system_avg: {stats.get('system_avg', 'N/A')}")
    print(f"  system_max: {stats.get('system_max', 'N/A')}")
    print(f"  system_diff: {stats.get('system_diff', 'N/A')}")
    print(f"  completed_count: {stats.get('completed_count', 'N/A')}")
    print(f"  in_progress_count: {stats.get('in_progress_count', 'N/A')}")
    print(f"  chambers: {len(stats.get('chambers', {}))} 个腔室")
    print(f"  transports: {len(stats.get('transports', {}))} 个机械手")
    print(f"  transports_detail: {len(stats.get('transports_detail', {}))} 个运输位")
    
    # 验证数据结构
    required_keys = ['system_avg', 'system_max', 'system_diff', 
                     'completed_count', 'in_progress_count',
                     'chambers', 'transports', 'transports_detail']
    
    missing_keys = [k for k in required_keys if k not in stats]
    
    if missing_keys:
        print(f"\n✗ 失败: 缺少字段 {missing_keys}")
    else:
        print(f"\n✓ 通过: 所有必需字段都存在")
    
    # 显示详细的腔室统计
    print("\n腔室统计详情:")
    for place_name, metrics in stats.get('chambers', {}).items():
        print(f"  {place_name}: avg={metrics.get('avg', 0):.2f}, max={metrics.get('max', 0):.2f}")
    
    # 显示机械手统计
    print("\n机械手统计详情:")
    for robot_name, metrics in stats.get('transports', {}).items():
        print(f"  {robot_name}: avg={metrics.get('avg', 0):.2f}, max={metrics.get('max', 0):.2f}")
    
    print()

def test_adapter_integration():
    """测试 petri_adapter 集成"""
    print("=" * 60)
    print("测试3: petri_adapter 集成")
    print("=" * 60)
    
    try:
        from visualization.petri_adapter import PetriAdapter
        
        env = Env_PN()
        adapter = PetriAdapter(env)
        
        # 检查是否启用了统计
        print(f"enable_statistics: {getattr(adapter.net, 'enable_statistics', 'N/A')}")
        
        # 重置并执行几步
        adapter.reset()
        for _ in range(5):
            actions = adapter.get_enabled_actions()
            enabled = [a.action_id for a in actions if a.enabled]
            if enabled:
                adapter.step(enabled[0])
        
        # 获取状态信息
        state = adapter.get_current_state()
        
        print(f"\nStateInfo.stats 包含的键:")
        for key in state.stats.keys():
            print(f"  - {key}")
        
        # 验证统计数据
        has_stats = any(k in state.stats for k in ['system_avg', 'chambers', 'transports'])
        
        if has_stats:
            print(f"\n✓ 通过: StateInfo.stats 包含统计数据")
        else:
            print(f"\n✗ 失败: StateInfo.stats 不包含统计数据")
        
    except ImportError as e:
        print(f"✗ 跳过: 无法导入 visualization 模块 ({e})")
    
    print()

if __name__ == "__main__":
    test_statistics_disabled()
    test_statistics_enabled()
    test_adapter_integration()
    
    print("=" * 60)
    print("测试完成")
    print("=" * 60)
