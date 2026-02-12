"""
步骤5: 综合修复验证
1. 在加载数据之初去除 L/R-3000, L/R-4000, L/R-6000 为空的行
2. 对比 better_ear 等选项的默认设置
3. 验证 PTAResult 修复后的效果
"""

import sys
sys.path.insert(0, r'c:\Users\Liuhe\Study_and_Work\0.Main\0.Current\浙健\2.Code\Mine\occupational_health_toolkits')

import pandas as pd
import numpy as np
import os
from datetime import datetime
from loguru import logger
from ohtk.staff_info import StaffInfo
from ohtk.diagnose_info.auditory_diagnose import AuditoryDiagnose
from ohtk.detection_info.auditory_detection.PTA_result import PTAResult

# 配置
logger.remove()
logger.add(sys.stderr, level="INFO")
BASE_DIR = r'c:\Users\Liuhe\Study_and_Work\0.Main\0.Current\浙健\2.Code\Mine\occupational_health_toolkits'
DATA_FILE = os.path.join(BASE_DIR, 'examples/data/All_Chinese_worker_exposure_data_0401.xlsx')

def check_better_ear_defaults():
    """对比 better_ear 等选项的默认设置"""
    logger.info("\n" + "="*80)
    logger.info("检查 better_ear 等选项的默认设置")
    logger.info("="*80)
    
    # 检查 PTAResult 的默认设置
    logger.info("\n1. PTAResult 类默认设置:")
    logger.info(f"   better_ear: {PTAResult.model_fields.get('better_ear', 'N/A')}")
    logger.info(f"   默认值: {PTAResult.model_fields.get('better_ear').default if PTAResult.model_fields.get('better_ear') else 'N/A'}")
    
    # 测试创建 PTAResult 对象时的默认行为
    test_data = {
        "L-500": 10, "L-1000": 15, "L-2000": 20, "L-3000": 25, "L-4000": 30, "L-6000": 35,
        "R-500": 12, "R-1000": 18, "R-2000": 22, "R-3000": 28, "R-4000": 32, "R-6000": 38,
    }
    
    try:
        pta = PTAResult(data=test_data)
        logger.info(f"   测试对象 better_ear: {pta.better_ear}")
        logger.info(f"   测试对象 better_ear_data: {pta.better_ear_data}")
        logger.info(f"   测试对象 poorer_ear: {pta.poorer_ear}")
        logger.info(f"   测试对象 poorer_ear_data: {pta.poorer_ear_data}")
    except Exception as e:
        logger.error(f"   创建 PTAResult 失败: {e}")

def filter_data_with_required_frequencies(df):
    """
    过滤数据：只保留 L/R-3000, L/R-4000, L/R-6000 都有值的行
    """
    logger.info("\n" + "="*80)
    logger.info("数据过滤: 去除关键频率为空的行")
    logger.info("="*80)
    
    required_cols = ['L-3000', 'L-4000', 'L-6000', 'R-3000', 'R-4000', 'R-6000']
    
    logger.info(f"\n原始数据行数: {len(df)}")
    logger.info(f"需要检查的列: {required_cols}")
    
    # 检查每列的空值情况
    for col in required_cols:
        if col in df.columns:
            null_count = df[col].isna().sum()
            logger.info(f"  {col}: {null_count} 个空值")
        else:
            logger.warning(f"  {col}: 列不存在!")
    
    # 过滤：所有关键频率都必须有值
    mask = pd.Series([True] * len(df))
    for col in required_cols:
        if col in df.columns:
            mask = mask & df[col].notna()
    
    filtered_df = df[mask].copy()
    logger.info(f"\n过滤后数据行数: {len(filtered_df)} ({len(filtered_df)/len(df)*100:.1f}%)")
    
    return filtered_df

def build_staff_info_safe(row):
    """安全地构建 StaffInfo"""
    try:
        row_dict = row.to_dict()
        
        # 字段映射
        staff_id = row_dict.get('staff_id')
        if not staff_id:
            staff_id = f"worker_{row.name}"
        
        # 处理 creation_date
        if 'creation_date' not in row_dict or pd.isna(row_dict.get('creation_date')):
            row_dict['creation_date'] = datetime(2021, 1, 1)
        
        # 构建 DataFrame
        row_df = pd.DataFrame([row_dict])
        
        # 使用 load_from_dataframe 构建
        staff = StaffInfo.load_from_dataframe(row_df, staff_id)
        return staff
    except Exception as e:
        logger.debug(f"构建 StaffInfo 失败: {e}")
        return None

def calculate_all_metrics(staff, row):
    """计算所有指标"""
    results = {}
    
    # 1. NIHL 计算
    try:
        ear_data = {}
        freq_map = {
            'left': {500: 'L-500', 1000: 'L-1000', 2000: 'L-2000', 
                     3000: 'L-3000', 4000: 'L-4000', 6000: 'L-6000'},
            'right': {500: 'R-500', 1000: 'R-1000', 2000: 'R-2000',
                      3000: 'R-3000', 4000: 'R-4000', 6000: 'R-6000'}
        }
        
        for ear, mappings in freq_map.items():
            for freq, source_col in mappings.items():
                target_col = f'{ear}_ear_{freq}'
                val = row.get(source_col)
                if not pd.isna(val):
                    ear_data[target_col] = val
        
        if ear_data:
            results['NIHL_346'] = AuditoryDiagnose.calculate_NIHL(
                ear_data=ear_data, freq_key="346", 
                age=row.get('age'), sex=row.get('sex'), apply_correction=False
            )
            results['NIHL_1234'] = AuditoryDiagnose.calculate_NIHL(
                ear_data=ear_data, freq_key="1234",
                age=row.get('age'), sex=row.get('sex'), apply_correction=False
            )
    except Exception as e:
        logger.debug(f"NIHL 计算失败: {e}")
        results['NIHL_346'] = None
        results['NIHL_1234'] = None
    
    # 2. NIPTS ISO 2013
    try:
        results['NIPTS_ISO2013'] = staff.NIPTS_predict_iso1999_2013(mean_key=[3000, 4000, 6000])
    except Exception as e:
        logger.debug(f"NIPTS ISO 2013 计算失败: {e}")
        results['NIPTS_ISO2013'] = None
    
    # 3. NIPTS ISO 2023
    try:
        results['NIPTS_ISO2023'] = staff.NIPTS_predict_iso1999_2023(mean_key=[3000, 4000, 6000])
    except Exception as e:
        logger.debug(f"NIPTS ISO 2023 计算失败: {e}")
        results['NIPTS_ISO2023'] = None
    
    return results

def process_data(df):
    """处理数据"""
    logger.info("\n" + "="*80)
    logger.info("开始处理数据")
    logger.info("="*80)
    
    results = []
    staff_success = 0
    nihl_success = 0
    nipts_2013_success = 0
    nipts_2023_success = 0
    
    for idx in range(len(df)):
        if idx % 100 == 0:
            logger.info(f"处理第 {idx+1}/{len(df)} 行...")
        
        row = df.iloc[idx]
        
        # 构建 StaffInfo
        staff = build_staff_info_safe(row)
        if staff is None:
            continue
        
        staff_success += 1
        
        # 计算指标
        metrics = calculate_all_metrics(staff, row)
        
        if metrics['NIHL_346'] is not None and not np.isnan(metrics['NIHL_346']):
            nihl_success += 1
        if metrics['NIPTS_ISO2013'] is not None and not np.isnan(metrics['NIPTS_ISO2013']):
            nipts_2013_success += 1
        if metrics['NIPTS_ISO2023'] is not None and not np.isnan(metrics['NIPTS_ISO2023']):
            nipts_2023_success += 1
        
        result = {
            'staff_id': staff.staff_id,
            'sex': staff.staff_sex,
            'age': staff.staff_age,
            'duration': staff.staff_duration,
            'LAeq': row.get('LAeq'),
            'NIHL_346': metrics['NIHL_346'],
            'NIHL_1234': metrics['NIHL_1234'],
            'NIPTS_ISO2013': metrics['NIPTS_ISO2013'],
            'NIPTS_ISO2023': metrics['NIPTS_ISO2023'],
            'original_NIPTS': row.get('NIPTS'),
            'original_NIPTS_pred_2013': row.get('NIPTS_pred_2013'),
            'original_NIPTS_pred_2023': row.get('NIPTS_pred_2023'),
        }
        results.append(result)
    
    logger.info(f"\n处理完成:")
    logger.info(f"  StaffInfo 构建成功: {staff_success}/{len(df)} ({staff_success/len(df)*100:.1f}%)")
    logger.info(f"  NIHL 计算成功: {nihl_success}/{len(df)} ({nihl_success/len(df)*100:.1f}%)")
    logger.info(f"  NIPTS ISO 2013 计算成功: {nipts_2013_success}/{len(df)} ({nipts_2013_success/len(df)*100:.1f}%)")
    logger.info(f"  NIPTS ISO 2023 计算成功: {nipts_2023_success}/{len(df)} ({nipts_2023_success/len(df)*100:.1f}%)")
    
    return pd.DataFrame(results)

def compare_results(results_df):
    """对比结果"""
    logger.info("\n" + "="*80)
    logger.info("结果对比")
    logger.info("="*80)
    
    # ISO 2013 对比
    valid_2013 = results_df[
        results_df['original_NIPTS_pred_2013'].notna() & 
        results_df['NIPTS_ISO2013'].notna()
    ]
    if len(valid_2013) > 0:
        diff = valid_2013['NIPTS_ISO2013'] - valid_2013['original_NIPTS_pred_2013']
        logger.info(f"\nNIPTS_pred_2013 对比 (有效行数: {len(valid_2013)}):")
        logger.info(f"  平均差异: {diff.mean():.4f}")
        logger.info(f"  标准差: {diff.std():.4f}")
        logger.info(f"  最大绝对差异: {diff.abs().max():.4f}")
    
    # ISO 2023 对比 - 只对比两者都有值的情况
    valid_2023 = results_df[
        results_df['original_NIPTS_pred_2023'].notna() & 
        results_df['NIPTS_ISO2023'].notna()
    ]
    if len(valid_2023) > 0:
        diff = valid_2023['NIPTS_ISO2023'] - valid_2023['original_NIPTS_pred_2023']
        logger.info(f"\nNIPTS_pred_2023 对比 (有效行数: {len(valid_2023)}):")
        logger.info(f"  平均差异: {diff.mean():.4f}")
        logger.info(f"  标准差: {diff.std():.4f}")
        logger.info(f"  最大绝对差异: {diff.abs().max():.4f}")
    
    # NIHL 统计
    valid_nihl = results_df[results_df['NIHL_346'].notna()]
    if len(valid_nihl) > 0:
        logger.info(f"\nNIHL 统计 (有效行数: {len(valid_nihl)}):")
        logger.info(f"  均值: {valid_nihl['NIHL_346'].mean():.2f}")
        logger.info(f"  范围: [{valid_nihl['NIHL_346'].min():.2f}, {valid_nihl['NIHL_346'].max():.2f}]")

def save_results(results_df):
    """保存结果"""
    output_dir = os.path.join(BASE_DIR, 'examples/chinese_worker_experiment/results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存完整结果
    output_file = os.path.join(output_dir, 'step1_comprehensive_results.csv')
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    logger.info(f"\n完整结果已保存: {output_file}")
    
    # 保存对比报告
    report_file = os.path.join(output_dir, 'step1_comparison_report.csv')
    
    report_data = []
    for _, row in results_df.iterrows():
        report_row = {
            'staff_id': row['staff_id'],
            'sex': row['sex'],
            'age': row['age'],
            'duration': row['duration'],
            'LAeq': row['LAeq'],
            'original_NIPTS': row['original_NIPTS'],
            'original_NIPTS_pred_2013': row['original_NIPTS_pred_2013'],
            'original_NIPTS_pred_2023': row['original_NIPTS_pred_2023'],
            'calculated_NIHL_346': row['NIHL_346'],
            'calculated_NIHL_1234': row['NIHL_1234'],
            'calculated_NIPTS_ISO2013': row['NIPTS_ISO2013'],
            'calculated_NIPTS_ISO2023': row['NIPTS_ISO2023'],
            'diff_NIPTS_2013': row['NIPTS_ISO2013'] - row['original_NIPTS_pred_2013'] 
                if pd.notna(row['NIPTS_ISO2013']) and pd.notna(row['original_NIPTS_pred_2013']) else None,
            'diff_NIPTS_2023': row['NIPTS_ISO2023'] - row['original_NIPTS_pred_2023']
                if pd.notna(row['NIPTS_ISO2023']) and pd.notna(row['original_NIPTS_pred_2023']) else None,
        }
        report_data.append(report_row)
    
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(report_file, index=False, encoding='utf-8-sig')
    logger.info(f"对比报告已保存: {report_file}")

def main():
    logger.info("="*80)
    logger.info("步骤1 加载数据为 StaffInfo")
    logger.info("="*80)
    
    # 1. 检查 better_ear 默认设置
    check_better_ear_defaults()
    
    # 2. 加载数据
    logger.info(f"\n加载数据: {DATA_FILE}")
    df = pd.read_excel(DATA_FILE)
    logger.info(f"原始数据形状: {df.shape}")
    
    # 3. 过滤数据（去除关键频率为空的行）
    filtered_df = filter_data_with_required_frequencies(df)
    
    # 4. 处理数据
    results_df = process_data(filtered_df)
    
    # 5. 对比结果
    compare_results(results_df)
    
    # 6. 保存结果
    save_results(results_df)
    
    logger.info("\n" + "="*80)
    logger.info("步骤1 完成!")
    logger.info("="*80)

if __name__ == "__main__":
    main()
