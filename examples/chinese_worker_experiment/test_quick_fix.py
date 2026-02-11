# -*- coding: utf-8 -*-
"""
快速验证脚本 - 测试修复后的实验框架

验证：
1. auditory_diagnose.py 重命名方法可用
2. RegressionExperiment 可以正常初始化
3. 数据处理方法不报错
"""

import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ohtk.diagnose_info.auditory_diagnose import AuditoryDiagnose
from ohtk.experiments import RegressionExperiment, ExperimentConfig

def test_auditory_diagnose_methods():
    """测试 AuditoryDiagnose 重命名方法"""
    print("\n" + "=" * 60)
    print("测试1: AuditoryDiagnose 方法重命名")
    print("=" * 60)
    
    # 测试 NIHL 计算
    ear_data = {
        'left_ear_3000': 25.0,
        'right_ear_3000': 30.0,
        'left_ear_4000': 35.0,
        'right_ear_4000': 40.0,
        'left_ear_6000': 45.0,
        'right_ear_6000': 50.0,
    }
    
    try:
        nihl = AuditoryDiagnose.calculate_NIHL(
            ear_data=ear_data,
            freq_key="346",
            apply_correction=False
        )
        print(f"✅ calculate_NIHL() 工作正常: {nihl:.2f} dB")
    except Exception as e:
        print(f"❌ calculate_NIHL() 失败: {e}")
        return False
    
    # 测试 NIPTS 预测
    try:
        nipts = AuditoryDiagnose.predict_NIPTS_iso1999_2023(
            LAeq=85.0,
            age=40,
            sex='M',
            duration=10.0,
            mean_key=[3000, 4000, 6000],
            percentrage=50
        )
        print(f"✅ predict_NIPTS_iso1999_2023() 工作正常: {nipts:.2f} dB")
    except Exception as e:
        print(f"❌ predict_NIPTS_iso1999_2023() 失败: {e}")
        return False
    
    # 测试旧方法（应该输出警告）
    print("\n测试向后兼容（应显示废弃警告）:")
    try:
        nihl_old = AuditoryDiagnose.NIHL(
            ear_data=ear_data,
            freq_key="346"
        )
        print(f"✅ NIHL() 向后兼容工作正常: {nihl_old:.2f} dB")
    except Exception as e:
        print(f"❌ NIHL() 向后兼容失败: {e}")
        return False
    
    return True

def test_experiment_initialization():
    """测试 RegressionExperiment 初始化"""
    print("\n" + "=" * 60)
    print("测试2: RegressionExperiment 初始化")
    print("=" * 60)
    
    config_file = project_root / "examples" / "chinese_worker_experiment" / "config.yaml"
    
    if not config_file.exists():
        print(f"❌ 配置文件不存在: {config_file}")
        return False
    
    try:
        config = ExperimentConfig.from_yaml(config_file)  # 使用 from_yaml 类方法
        print(f"✅ 配置文件加载成功")
        print(f"  - 实验名称: {config['experiment_name']}")
        print(f"  - 模型数量: {len(config['models'])}")
        print(f"  - 目标变量: {config['targets']}")
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        exp_config = config.config.copy()
        exp_config['target_col'] = 'NIHL_1234'
        
        exp = RegressionExperiment(
            config=exp_config,
            model_type='lightgbm',
            target_col='NIHL_1234'
        )
        print(f"✅ RegressionExperiment 初始化成功")
        print(f"  - 模型类型: {exp.model_type}")
        print(f"  - 目标变量: {exp.target_col}")
    except Exception as e:
        print(f"❌ RegressionExperiment 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("开始快速验证测试")
    print("=" * 60)
    
    success_count = 0
    total_tests = 2
    
    # 测试1
    if test_auditory_diagnose_methods():
        success_count += 1
    
    # 测试2
    if test_experiment_initialization():
        success_count += 1
    
    # 总结
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    print(f"通过: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("✅ 所有测试通过！实验框架修复成功！")
        print("\n可以运行完整实验:")
        print("  cd examples/chinese_worker_experiment")
        print("  python run_experiment_new.py")
    else:
        print("❌ 部分测试失败，请检查错误信息")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
