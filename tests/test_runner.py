"""测试运行器脚本"""
import sys
import traceback

def test_imports():
    """测试基本导入"""
    print("=" * 60)
    print("测试 1: 基本导入测试")
    print("=" * 60)
    try:
        from ohtk.staff_info import StaffInfo
        print("✓ StaffInfo 导入成功")
        return True
    except Exception as e:
        print(f"✗ StaffInfo 导入失败: {e}")
        traceback.print_exc()
        return False

def test_help():
    """测试 help 功能"""
    print("\n" + "=" * 60)
    print("测试 2: StaffInfo.help() 功能")
    print("=" * 60)
    try:
        from ohtk.staff_info import StaffInfo
        StaffInfo.help()
        print("\n✓ help() 功能正常")
        return True
    except Exception as e:
        print(f"\n✗ help() 功能失败: {e}")
        traceback.print_exc()
        return False

def test_get_expected_fields():
    """测试 get_expected_fields 功能"""
    print("\n" + "=" * 60)
    print("测试 3: StaffInfo.get_expected_fields() 功能")
    print("=" * 60)
    try:
        from ohtk.staff_info import StaffInfo
        fields = StaffInfo.get_expected_fields()
        print(f"返回的字段列表:\n{fields}")
        print(f"\n字段数量: {len(fields)}")
        print(f"✓ get_expected_fields() 功能正常")
        return True
    except Exception as e:
        print(f"\n✗ get_expected_fields() 功能失败: {e}")
        traceback.print_exc()
        return False

def test_check_order_auto_calculation():
    """测试 check_order 自动计算"""
    print("\n" + "=" * 60)
    print("测试 4: check_order 和 days_since_first 自动计算")
    print("=" * 60)
    try:
        from ohtk.staff_info import StaffInfo
        
        staff = StaffInfo(
            staff_id="TEST001",
            creation_date=["2024-01-15", "2024-06-20", "2025-01-10"],
            sex=[1, 1, 1],
            age=[35, 35, 36],
            NIHL346=[18.5, 19.0, 20.5]
        )
        
        print(f"输入数据:")
        print(f"  staff_id: {staff.staff_id}")
        print(f"  creation_date: {staff.creation_date}")
        print(f"  sex: {staff.sex}")
        print(f"  age: {staff.age}")
        print(f"  NIHL346: {staff.NIHL346}")
        print(f"\n自动计算结果:")
        print(f"  check_order: {staff.check_order}")
        print(f"  days_since_first: {staff.days_since_first}")
        
        # 验证结果
        assert staff.check_order == [1, 2, 3], f"check_order 不正确: {staff.check_order}"
        assert staff.days_since_first is not None, "days_since_first 为 None"
        assert len(staff.days_since_first) == 3, f"days_since_first 长度不正确: {len(staff.days_since_first)}"
        
        print(f"\n✓ check_order 和 days_since_first 自动计算功能正常")
        return True
    except Exception as e:
        print(f"\n✗ check_order 自动计算失败: {e}")
        traceback.print_exc()
        return False

def test_pytest():
    """运行 pytest 测试"""
    print("\n" + "=" * 60)
    print("测试 5: 运行 pytest 测试套件")
    print("=" * 60)
    try:
        import pytest
        print("pytest 已安装，开始运行测试...")
        result = pytest.main(["-v", "tests/"])
        if result == 0:
            print("\n✓ 所有 pytest 测试通过")
            return True
        else:
            print(f"\n✗ pytest 测试失败，返回代码: {result}")
            return False
    except ImportError:
        print("⚠ pytest 未安装，跳过此测试")
        print("  提示: 使用 'pip install pytest' 安装")
        return None
    except Exception as e:
        print(f"\n✗ pytest 运行出错: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试 StaffInfo 新功能...\n")
    
    results = []
    results.append(("基本导入", test_imports()))
    results.append(("help() 功能", test_help()))
    results.append(("get_expected_fields() 功能", test_get_expected_fields()))
    results.append(("check_order 自动计算", test_check_order_auto_calculation()))
    results.append(("pytest 测试套件", test_pytest()))
    
    # 输出总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)
    
    for name, result in results:
        status = "✓ 通过" if result is True else ("✗ 失败" if result is False else "⚠ 跳过")
        print(f"{name}: {status}")
    
    print(f"\n总计: {passed} 通过, {failed} 失败, {skipped} 跳过")
    
    # 返回退出码
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
