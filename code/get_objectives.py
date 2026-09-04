# -*- coding: utf-8 -*-
"""
计算个体的目标值函数
支持分子优化目标（QED, SA, JNK3, GSK3B, LogP, DRD2, Similarity）
"""

import sys
import numpy as np
from typing import Dict, Optional, Callable, Union

# ===== sklearn兼容层 =====
# 旧版本sklearn模型（如GSK3B）使用了已废弃的模块路径
# 需要创建别名以兼容
try:
    import sklearn.ensemble
    import sklearn.tree
    sys.modules['sklearn.ensemble.forest'] = sklearn.ensemble
    sys.modules['sklearn.tree.tree'] = sklearn.tree
except ImportError:
    pass  # sklearn未安装时忽略

# ===== 全局Oracle缓存 =====
# 避免重复初始化Oracle，提升300倍性能！
_ORACLE_CACHE = {}

def _get_oracle(name: str):
    """
    获取或创建Oracle实例（使用缓存）
    
    Args:
        name: Oracle名称，如 'qed', 'sa', 'GSK3B'
        
    Returns:
        Oracle实例
    """
    if name not in _ORACLE_CACHE:
        from tdc import Oracle
        _ORACLE_CACHE[name] = Oracle(name)
        print(f"[Oracle缓存] 已创建 {name} Oracle")
    return _ORACLE_CACHE[name]


def get_objectives(individual, 
                   n_objectives: int, 
                   config: Dict) -> np.ndarray:
    """
    计算单个个体的目标值
    
    Args:
        individual: SMILES 字符串
        n_objectives: 目标函数数量
        config: 目标函数配置字典（必需），包含：
            - molecular_objectives: 目标函数列表 ['qed', 'sa', 'jnk3', 'similarity']
            - reference_smiles: (可选) 参考分子的SMILES，用于计算相似度
        
    Returns:
        objectives: 目标值向量 (n_objectives,)
    
    Example:
        >>> config = {
        ...     'molecular_objectives': ['qed', 'sa', 'similarity'],
        ...     'reference_smiles': 'CC(C)O'
        ... }
        >>> obj = get_objectives('CCO', n_objectives=3, config=config)
    """
    # 从 config 读取需要计算的目标列表
    molecular_objectives = config.get('molecular_objectives', ['qed', 'sa'])
    
    # 只取前 n_objectives 个目标
    objectives_to_calculate = molecular_objectives[:n_objectives]
    
    # 获取参考分子（用于相似度计算）
    reference_smiles = config.get('reference_smiles', None)
    
    # 调用 _calculate_molecular_objective 一次性计算所有目标
    objectives = _calculate_molecular_objective(
        individual, 
        objectives_to_calculate,
        reference_smiles=reference_smiles
    )
    
    return objectives


def _calculate_molecular_objective(smiles: str, 
                                    objective_list: list,
                                    reference_smiles: Optional[str] = None) -> np.ndarray:
    """
    计算分子的所有目标值
    
    Args:
        smiles: SMILES 字符串
        objective_list: 目标函数名称列表，例如 ['qed', 'sa', 'jnk3', 'similarity']
        reference_smiles: 参考分子的SMILES（用于计算相似度，可选）
        
    Returns:
        objectives: 目标值向量 (len(objective_list),)
    """
    
    def calculate_Dissimilarity(smi: str) -> float:
        """
        计算与Pioglitazone的非相似性（Dissimilarity）
        
        使用ECFP4指纹计算Tanimoto相似度，然后通过高斯修饰器转换
        目标是相似度接近0（即希望生成不同的分子）
        
        Args:
            smi: 待计算的SMILES字符串
            
        Returns:
            float: 非相似性得分 [0, 1]，经过高斯修饰器处理
        """
        try:
            from rdkit import Chem, DataStructs
            from rdkit.Chem import AllChem
            import math
            
            # Pioglitazone的SMILES
            pioglitazone_smiles = 'O=C1NC(=O)SC1Cc3ccc(OCCc2ncc(cc2)CC)cc3'
            
            # 将SMILES转换为分子对象
            mol = Chem.MolFromSmiles(smi)
            ref_mol = Chem.MolFromSmiles(pioglitazone_smiles)
            
            if mol is None or ref_mol is None:
                return 0.0
            
            # 计算ECFP4指纹（Morgan指纹，半径2）
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fp_ref = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)
            
            # 计算Tanimoto相似度
            similarity = DataStructs.TanimotoSimilarity(fp, fp_ref)
            
            # 应用高斯修饰器：GaussianModifier(mu=0, sigma=0.1)
            # score = exp(-0.5 * ((x - mu) / sigma)^2)
            mu = 0.0
            sigma = 0.1
            score = math.exp(-0.5 * ((similarity - mu) / sigma) ** 2)
            
            return float(score)
            
        except Exception as e:
            print(f"Error calculating Dissimilarity: {e}")
            return 0.0
    def calculate_jnk3(smi: str) -> float:
        """计算 JNK3 激酶抑制活性"""
        try:
            import sys, os
            oracle = _get_oracle('JNK3')
            # 抑制Oracle调用时的输出
            original_stdout, original_stderr = sys.stdout, sys.stderr
            devnull = open(os.devnull, 'w')
            sys.stdout, sys.stderr = devnull, devnull
            try:
                result = float(oracle(smi))
            finally:
                sys.stdout, sys.stderr = original_stdout, original_stderr
                devnull.close()
            return result
        except Exception as e:
            print(f"Error calculating JNK3: {e}")
            return 0.0

    def calculate_MW(smi: str) -> float:
        """
        计算分子量得分（Molecular Weight Score）
        
        目标是分子量接近Pioglitazone的分子量
        使用高斯修饰器，目标分子量为Pioglitazone的分子量，标准差为10
        
        Args:
            smi: 待计算的SMILES字符串
            
        Returns:
            float: 分子量得分 [0, 1]，经过高斯修饰器处理
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            import math
            
            # Pioglitazone的SMILES和目标分子量
            pioglitazone_smiles = 'O=C1NC(=O)SC1Cc3ccc(OCCc2ncc(cc2)CC)cc3'
            ref_mol = Chem.MolFromSmiles(pioglitazone_smiles)
            
            if ref_mol is None:
                target_molw = 356.44  # Pioglitazone的理论分子量
            else:
                target_molw = Descriptors.MolWt(ref_mol)
            
            # 计算当前分子的分子量
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return 0.0
            
            molw = Descriptors.MolWt(mol)
            
            # 应用高斯修饰器：GaussianModifier(mu=target_molw, sigma=10)
            mu = target_molw
            sigma = 10.0
            score = math.exp(-0.5 * ((molw - mu) / sigma) ** 2)
            
            return float(score)
            
        except Exception as e:
            print(f"Error calculating MW: {e}")
            return 0.0
    
    def calculate_RB(smi: str) -> float:
        """
        计算可旋转键数得分（Rotatable Bonds Score）
        
        目标是可旋转键数量接近2
        使用高斯修饰器，均值为2，标准差为0.5
        
        Args:
            smi: 待计算的SMILES字符串
            
        Returns:
            float: 可旋转键得分 [0, 1]，经过高斯修饰器处理
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Lipinski
            import math
            
            # 计算当前分子的可旋转键数
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return 0.0
            
            num_rotatable = Lipinski.NumRotatableBonds(mol)
            
            # 应用高斯修饰器：GaussianModifier(mu=2, sigma=0.5)
            mu = 2.0
            sigma = 0.5
            score = math.exp(-0.5 * ((num_rotatable - mu) / sigma) ** 2)
            
            return float(score)
            
        except Exception as e:
            print(f"Error calculating RB: {e}")
            return 0.0
    
    # 定义各个目标函数的计算方法
    def calculate_qed(smi: str) -> float:
        """计算 QED (Quantitative Estimate of Druglikeness)"""
        try:
            import sys, os
            oracle = _get_oracle('qed')
            # 抑制Oracle调用时的输出
            original_stdout, original_stderr = sys.stdout, sys.stderr
            devnull = open(os.devnull, 'w')
            sys.stdout, sys.stderr = devnull, devnull
            try:
                result = float(oracle(smi))
            finally:
                sys.stdout, sys.stderr = original_stdout, original_stderr
                devnull.close()
            return result
        except Exception as e:
            print(f"Error calculating QED (TDC oracle): {e}")
            return 0.0
    
    def calculate_qed_moses(smi: str) -> float:
        """使用 MOSES/RDKit 版本计算 QED
        
        与 MOMO 任务1一致：
        - 先用 RDKit 将 SMILES 转为 Mol
        - 再调用 RDKit 自带的 QED 实现计算 QED
        
        说明：MOMO 中的 moses.metrics.QED 本质是对 RDKit QED 的封装，
        这里直接使用 RDKit.Chem.QED.qed(mol)，避免对 moses 包的依赖。
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import QED as RDKitQED
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return 0.0
            # RDKit QED 与 MOSES QED 数值等价
            return float(RDKitQED.qed(mol))
        except Exception as e:
            print(f"Error calculating QED (RDKit/MOSES-compatible): {e}")
            return 0.0
    
    def calculate_sa(smi: str) -> float:
        """计算 SA (Synthetic Accessibility) 并归一化
        
        使用RDKit的SA_Score模块计算合成可达性分数
        原始分数范围约1-10（越小越容易合成）
        归一化后范围0-1（越大越容易合成）
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import RDConfig
            import os
            import sys as _sys
            
            # 导入SA_Score模块
            _sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
            import sascorer
            
            # 将SMILES转换为分子对象
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return 0.0
            
            # 计算原始SA分数（范围约1-10，越小越容易合成）
            sa_score = sascorer.calculateScore(mol)
            
            # 归一化 SA 分数: (10 - SA) / 9，使得值越大越好
            normalized_sa = (10.0 - sa_score) / 9.0
            return float(normalized_sa)
        except Exception as e:
            print(f"Error calculating SA: {e}")
            return 0.0
    
    def calculate_jnk3(smi: str) -> float:
        """计算 JNK3 激酶抑制活性"""
        try:
            import sys, os
            oracle = _get_oracle('JNK3')
            # 抑制Oracle调用时的输出
            original_stdout, original_stderr = sys.stdout, sys.stderr
            devnull = open(os.devnull, 'w')
            sys.stdout, sys.stderr = devnull, devnull
            try:
                result = float(oracle(smi))
            finally:
                sys.stdout, sys.stderr = original_stdout, original_stderr
                devnull.close()
            return result
        except Exception as e:
            print(f"Error calculating JNK3: {e}")
            return 0.0
    
    def calculate_gsk3b(smi: str) -> float:
        """计算 GSK3B 激酶抑制活性"""
        try:
            import sys, os
            oracle = _get_oracle('GSK3B')
            # 抑制Oracle调用时的输出
            original_stdout, original_stderr = sys.stdout, sys.stderr
            devnull = open(os.devnull, 'w')
            sys.stdout, sys.stderr = devnull, devnull
            try:
                result = float(oracle(smi))
            finally:
                sys.stdout, sys.stderr = original_stdout, original_stderr
                devnull.close()
            return result
        except Exception as e:
            import traceback
            print(f"Error calculating GSK3B for SMILES '{smi}': {e}")
            traceback.print_exc()
            return 0.0
    
    def calculate_logp(smi: str) -> float:
        """计算 LogP (脂水分配系数)"""
        try:
            import sys, os
            oracle = _get_oracle('logp')
            # 抑制Oracle调用时的输出
            original_stdout, original_stderr = sys.stdout, sys.stderr
            devnull = open(os.devnull, 'w')
            sys.stdout, sys.stderr = devnull, devnull
            try:
                result = float(oracle(smi))
            finally:
                sys.stdout, sys.stderr = original_stdout, original_stderr
                devnull.close()
            return result
        except Exception as e:
            print(f"Error calculating LogP: {e}")
            return 0.0
    
    
    def calculate_drd2(smi: str) -> float:
        """计算 DRD2 受体活性"""
        try:
            import sys, os
            oracle = _get_oracle('DRD2')
            # 抑制Oracle调用时的输出
            original_stdout, original_stderr = sys.stdout, sys.stderr
            devnull = open(os.devnull, 'w')
            sys.stdout, sys.stderr = devnull, devnull
            try:
                result = float(oracle(smi))
            finally:
                sys.stdout, sys.stderr = original_stdout, original_stderr
                devnull.close()
            return result
        except Exception as e:
            print(f"Error calculating DRD2: {e}")
            return 0.0
    
    def calculate_similarity(smi: str) -> float:
        """
        计算与参考分子的Tanimoto相似度
        
        使用Morgan指纹（radius=2, nBits=2048）计算Tanimoto相似度
        
        Args:
            smi: 待计算的SMILES字符串
            
        Returns:
            float: Tanimoto相似度 [0, 1]，若计算失败返回0.0
        """
        
        if reference_smiles is None:
            print("Warning: reference_smiles not provided for similarity calculation")
            return 0.0
        
        try:
            from rdkit import Chem, DataStructs
            from rdkit.Chem import AllChem
            
            # 将SMILES转换为分子对象
            mol = Chem.MolFromSmiles(smi)
            ref_mol = Chem.MolFromSmiles(reference_smiles)
            
            if mol is None or ref_mol is None:
                # 不打印单个错误，由上层统计
                return 0.0
            
            # 计算Morgan指纹（半径2，2048位）
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fp_ref = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)
            
            # 计算Tanimoto相似度
            similarity = DataStructs.TanimotoSimilarity(fp, fp_ref)
            return float(similarity)
            
        except Exception as e:
            print(f"Error calculating Similarity: {e}")
            return 0.0
    
    # 目标函数映射字典
    objective_functions = {
        'qed': calculate_qed,
        'qed_moses': calculate_qed_moses,
        'sa': calculate_sa,
        'jnk': calculate_jnk3,
        'gsk3b': calculate_gsk3b,
        'gsk3': calculate_gsk3b,
        'gsk': calculate_gsk3b,
        'logp': calculate_logp,
        'drd2': calculate_drd2,
        'drd': calculate_drd2,
        'similarity': calculate_similarity,
        'sim': calculate_similarity,  # 别名
        'dissimilarity': calculate_Dissimilarity,
        'mw': calculate_MW, 
        'rb': calculate_RB,
        'jnk3': calculate_jnk3,
        'jnk': calculate_jnk3,
    }
    
    # 计算所有目标值
    n_objectives = len(objective_list)
    objectives = np.zeros(n_objectives)
    
    for i, obj_name in enumerate(objective_list):
        obj_name_lower = obj_name.lower()
        
        if obj_name_lower in objective_functions:
            objectives[i] = objective_functions[obj_name_lower](smiles)
        else:
            print(f"Warning: Unknown objective '{obj_name}', setting to 0")
            objectives[i] = 0.0
    
    return objectives


def get_objectives_batch(population: list, 
                        n_objectives: int, 
                        config: Dict,
                        n_jobs: int = -1,
                        use_parallel: bool = True) -> np.ndarray:
    """
    批量计算种群的目标值（支持并行计算）
    
    Args:
        population: SMILES 字符串列表
        n_objectives: 目标函数数量
        config: 目标函数配置字典（必需），包含：
            - molecular_objectives: 目标函数列表
            - reference_smiles: (可选) 参考分子的SMILES，用于计算相似度
        n_jobs: 并行任务数（-1表示使用所有CPU核心，1表示串行）
        use_parallel: 是否使用并行计算（默认True）
        
    Returns:
        objectives: 目标值矩阵 (N, n_objectives)
    
    Example:
        >>> config = {
        ...     'molecular_objectives': ['qed', 'sa', 'similarity'],
        ...     'reference_smiles': 'CC(C)O'
        ... }
        >>> pop = ['CCO', 'CCCO', 'CCCCO']
        >>> objs = get_objectives_batch(pop, n_objectives=3, config=config, n_jobs=4)
        >>> print(objs.shape)  # 输出: (3, 3)
    """
    N = len(population)
    objectives = np.zeros((N, n_objectives))
    
    # 如果不使用并行，或n_jobs=1，或种群很小，使用串行计算
    if not use_parallel or n_jobs == 1 or N < 10:
        for i in range(N):
            objectives[i] = get_objectives(population[i], n_objectives, config)
    else:
        # 并行计算（推荐用于大种群）
        try:
            from joblib import Parallel, delayed
            import multiprocessing
            
            # 确定实际使用的进程数
            if n_jobs == -1:
                n_jobs = min(multiprocessing.cpu_count(), N)
            
            # 并行计算每个个体的目标值
            results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
                delayed(get_objectives)(population[i], n_objectives, config)
                for i in range(N)
            )
            
            # 整合结果
            for i, obj in enumerate(results):
                objectives[i] = obj
                
        except ImportError:
            # joblib未安装，降级为串行
            print("[警告] joblib未安装，使用串行计算。安装方法: pip install joblib")
            for i in range(N):
                objectives[i] = get_objectives(population[i], n_objectives, config)
        except Exception as e:
            # 并行计算出错，降级为串行
            print(f"[警告] 并行计算出错，降级为串行: {e}")
            for i in range(N):
                objectives[i] = get_objectives(population[i], n_objectives, config)
    
    return objectives


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试 get_objectives 函数")
    print("=" * 60)
    
    # 测试分子（SMILES字符串）
    # 从 data_qed/qed_test.csv 读取 SMILES 列表，取第一条作为测试用分子
    import os
    import pandas as pd
    data_file = os.path.join(os.path.dirname(__file__), 'data_qed', 'qed_test.csv')
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"找不到测试数据文件: {data_file}")

    # 考虑到文件可能没有表头且只有一列，这里统一按 header=None 读取
    df_test = pd.read_csv(data_file, header=None)
    if df_test.shape[1] < 1 or df_test.shape[0] < 1:
        raise ValueError("qed_test.csv 中没有可用的 SMILES 数据")

    # 第一列视为 SMILES，取第1-20行作为测试 SMILES
    test_smiles = str(df_test.iloc[20, 0])


    # ========== 测试3: QED_MOSES + SA ==========
    print("\n【测试3】QED_MOSES + SA (二目标，验证calculate_qed_moses)" )
    print("-" * 60)
    config3 = {
        'molecular_objectives': ['qed_moses', 'sa','qed']
    }

    print(f"SMILES: {test_smiles}")
    print(f"目标: {config3['molecular_objectives']}")

    try:
        obj3 = get_objectives(test_smiles, n_objectives=3, config=config3)
        print(f"结果: QED_MOSES={obj3[0]:.4f}, SA={obj3[1]:.4f},qed_tdc={obj3[2]:.4f}")
    except Exception as e:
        print(f"错误: {e}")

    
  