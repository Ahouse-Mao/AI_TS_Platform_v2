"""
参数校验技能综合测试

测试覆盖 ParamValidateSkill.validate() 的全部校验规则，
包括合法参数、各类错误和警告场景。

用法：
  uv run python skills/test_param_validate.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skills.param_validate_skill import ParamValidateSkill

# ============================================================
# 辅助
# ============================================================

_pass = 0
_fail = 0

def section(title: str):
    print(f"\n  {'=' * 60}")
    print(f"  {title}")
    print(f"  {'=' * 60}")

def test(label: str, params: dict, expect_valid: bool | None,
         expect_errors: list[str] | None = None,
         expect_warnings: list[str] | None = None):
    """执行单条测试并比对结果"""
    global _pass, _fail
    pvs = ParamValidateSkill()
    result = pvs.validate(params)

    # 检查 valid 状态
    valid_ok = True
    if expect_valid is not None and result["valid"] != expect_valid:
        valid_ok = False

    # 检查 errors（断言期望的错误都在结果中出现）
    errors_ok = True
    if expect_errors is not None:
        for exp in expect_errors:
            if not any(exp in e for e in result["errors"]):
                errors_ok = False
                break

    # 检查 warnings
    warnings_ok = True
    if expect_warnings is not None:
        for exp in expect_warnings:
            if not any(exp in w for w in result["warnings"]):
                warnings_ok = False
                break

    if valid_ok and errors_ok and warnings_ok:
        _pass += 1
        status = "✅"
    else:
        _fail += 1
        status = "❌"
        details = []
        if not valid_ok:
            details.append(f"expect valid={expect_valid}, got valid={result['valid']}")
        if not errors_ok:
            details.append(f"expect errors~{expect_errors}, got={result['errors']}")
        if not warnings_ok:
            details.append(f"expect warnings~{expect_warnings}, got={result['warnings']}")
        detail_str = " | ".join(details)

    print(f"\n  {status} {label}")
    print(f"      valid={result['valid']}, errors={result['errors']}, warnings={result['warnings']}")
    if not (valid_ok and errors_ok and warnings_ok):
        print(f"      ⚠  {detail_str}")


# 一份「完美合法」的基准参数
GOOD = {
    "model_name": "DLinear",
    "dataset": "ETTh1",
    "seq_len": 96,
    "pred_len": 96,
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 50,
    "d_model": 512,
    "n_heads": 8,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 2048,
    "dropout": 0.05,
    "features": "M",
    "patience": 5,
    "use_gpu": True,
    "patch_len": 16,
    "stride": 8,
}


# ============================================================
# 1. model_name 校验
# ============================================================
section("1. model_name")

test("1.1 全部合法模型", {}, None)  # 快速验证所有支持的模型
for m in ParamValidateSkill.SUPPORTED_MODELS:
    test(f"  支持: {m}", {**GOOD, "model_name": m}, True)

test("1.2 缺失 model_name",
     {k: v for k, v in GOOD.items() if k != "model_name"},
     False, ["缺少必填参数"])

test("1.3 model_name 类型错误",
     {**GOOD, "model_name": 123},
     False, ["必须为字符串"])

test("1.4 不支持的模型",
     {**GOOD, "model_name": "UnknownModel"},
     False, ["不支持的模型"])

# ============================================================
# 2. dataset 校验
# ============================================================
section("2. dataset")

for d in ParamValidateSkill.SUPPORTED_DATASETS:
    test(f"  支持: {d}", {**GOOD, "dataset": d}, True)

test("2.1 缺失 dataset",
     {k: v for k, v in GOOD.items() if k != "dataset"},
     False, ["缺少必填参数"])

test("2.2 dataset 类型错误",
     {**GOOD, "dataset": 123},
     False, ["必须为字符串"])

test("2.3 不支持的数据集",
     {**GOOD, "dataset": "NonExistent"},
     False, ["不支持的数据集"])

# ============================================================
# 3. seq_len 校验
# ============================================================
section("3. seq_len")

test("3.1 缺失 seq_len",
     {k: v for k, v in GOOD.items() if k != "seq_len"},
     False, ["缺少或无法解析"])

test("3.2 seq_len < 1",
     {**GOOD, "seq_len": 0},
     False, ["必须 ≥ 1"])

test("3.3 seq_len=2048 较大告警",
     {**GOOD, "seq_len": 2048},
     True, None, ["较大"])

test("3.4 seq_len=4096 非常大告警",
     {**GOOD, "seq_len": 4096},
     True, None, ["非常大"])

test("3.5 seq_len=1500 较大告警",
     {**GOOD, "seq_len": 1500},
     True, None, ["较大"])

# ============================================================
# 4. pred_len 校验 + 交叉校验
# ============================================================
section("4. pred_len + 交叉校验")

test("4.1 缺失 pred_len",
     {k: v for k, v in GOOD.items() if k != "pred_len"},
     False, ["缺少或无法解析"])

test("4.2 pred_len < 1",
     {**GOOD, "pred_len": 0},
     False, ["必须 ≥ 1"])

test("4.3 pred_len > seq_len（告警）",
     {**GOOD, "seq_len": 96, "pred_len": 120},
     True, None, ["pred_len=120 > seq_len=96"])

test("4.4 pred_len >> seq_len*2（告警）",
     {**GOOD, "seq_len": 96, "pred_len": 300},
     True, None, ["远超"])

# ============================================================
# 5. batch_size 校验
# ============================================================
section("5. batch_size")

test("5.1 batch_size=100 非 2 的幂次",
     {**GOOD, "batch_size": 100},
     True, None, ["不是 2 的幂次"])

test("5.2 batch_size=64 合法",
     {**GOOD, "batch_size": 64},
     True)

test("5.3 batch_size=0",
     {**GOOD, "batch_size": 0},
     False, ["必须为正整数"])

test("5.4 缺失 batch_size（非必填，不应报错）",
     {k: v for k, v in GOOD.items() if k != "batch_size"},
     True)

# ============================================================
# 6. learning_rate 校验
# ============================================================
section("6. learning_rate")

test("6.1 lr=0（非法）",
     {**GOOD, "learning_rate": 0},
     False, ["必须在 (0, 1)"])

test("6.2 lr=1.0（非法）",
     {**GOOD, "learning_rate": 1.0},
     False, ["必须在 (0, 1)"])

test("6.3 lr=0.05 偏大告警",
     {**GOOD, "learning_rate": 0.05},
     True, None, ["偏大"])

test("6.4 lr=1e-6 偏小告警",
     {**GOOD, "learning_rate": 1e-6},
     True, None, ["偏小"])

test("6.5 缺失 learning_rate（非必填）",
     {k: v for k, v in GOOD.items() if k != "learning_rate"},
     True)

# ============================================================
# 7. epochs 校验
# ============================================================
section("7. epochs")

test("7.1 epochs=0",
     {**GOOD, "epochs": 0},
     False, ["必须为正整数"])

test("7.2 epochs=1000 过大告警",
     {**GOOD, "epochs": 1000},
     True, None, ["较大"])

# ============================================================
# 8. d_model 校验
# ============================================================
section("8. d_model")

test("8.1 d_model=0",
     {**GOOD, "d_model": 0},
     False, ["必须为正整数"])

test("8.2 d_model=384 非 2 的幂次",
     {**GOOD, "d_model": 384},
     True, None, ["不是 2 的幂次"])

test("8.3 d_model=512 合法",
     {**GOOD, "d_model": 512},
     True)

# ============================================================
# 9. n_heads 校验 + d_model 联检
# ============================================================
section("9. n_heads + d_model 联检")

test("9.1 n_heads=0",
     {**GOOD, "n_heads": 0},
     False, ["必须为正整数"])

test("9.2 n_heads=7 不整除 d_model=512",
     {**GOOD, "d_model": 512, "n_heads": 7},
     False, ["不能整除"])

test("9.3 n_heads=8 整除 d_model=512 合法",
     {**GOOD, "d_model": 512, "n_heads": 8},
     True)

# ============================================================
# 10. e_layers / d_layers 校验
# ============================================================
section("10. e_layers / d_layers")

test("10.1 e_layers=0",
     {**GOOD, "e_layers": 0},
     False, ["必须 ≥ 1"])

test("10.2 d_layers=0",
     {**GOOD, "d_layers": 0},
     False, ["必须 ≥ 1"])

test("10.3 e_layers=12 过深告警",
     {**GOOD, "e_layers": 12},
     True, None, ["较深"])

# ============================================================
# 11. d_ff
# ============================================================
section("11. d_ff")

test("11.1 d_ff=0",
     {**GOOD, "d_ff": 0},
     False, ["必须为正整数"])

# ============================================================
# 12. dropout
# ============================================================
section("12. dropout")

test("12.1 dropout=-0.1",
     {**GOOD, "dropout": -0.1},
     False, ["必须在 [0, 1)"])

test("12.2 dropout=1.0",
     {**GOOD, "dropout": 1.0},
     False, ["必须在 [0, 1)"])

test("12.3 dropout=0.8 欠拟合告警",
     {**GOOD, "dropout": 0.8},
     True, None, ["可能导致欠拟合"])

test("12.4 dropout=0.3 合法",
     {**GOOD, "dropout": 0.3},
     True)

# ============================================================
# 13. features
# ============================================================
section("13. features")

test("13.1 features='ABC'",
     {**GOOD, "features": "ABC"},
     False, ["必须为"])

test("13.2 features='M' 合法",
     {**GOOD, "features": "M"},
     True)

test("13.3 features='S' 合法",
     {**GOOD, "features": "S"},
     True)

test("13.4 features='MS' 合法",
     {**GOOD, "features": "MS"},
     True)

# ============================================================
# 14. patience
# ============================================================
section("14. patience")

test("14.1 patience=0",
     {**GOOD, "patience": 0},
     False, ["必须 ≥ 1"])

# ============================================================
# 15. use_gpu
# ============================================================
section("15. use_gpu")

test("15.1 use_gpu='yes' 字符串",
     {**GOOD, "use_gpu": "yes"},
     False, ["必须为布尔值"])

test("15.2 use_gpu=1 整数",
     {**GOOD, "use_gpu": 1},
     False, ["必须为布尔值"])

test("15.3 use_gpu=True 合法",
     {**GOOD, "use_gpu": True},
     True)

test("15.4 use_gpu=False 合法",
     {**GOOD, "use_gpu": False},
     True)

# ============================================================
# 16. PatchTST 特有参数
# ============================================================
section("16. PatchTST 特有参数")

test("16.1 PatchTST 缺 patch_len",
     {k: v for k, v in {**GOOD, "model_name": "PatchTST"}.items() if k != "patch_len"},
     False, ["缺少必填参数"])

test("16.2 PatchTST 缺 stride",
     {k: v for k, v in {**GOOD, "model_name": "PatchTST"}.items() if k != "stride"},
     False, ["缺少必填参数"])

test("16.3 PatchTST patch_len=0",
     {**GOOD, "model_name": "PatchTST", "patch_len": 0},
     False, ["必须 ≥ 1"])

test("16.4 PatchTST 全部合法",
     {**GOOD, "model_name": "PatchTST"},
     True)

# ============================================================
# 17. lradj
# ============================================================
section("17. lradj")

test("17.1 lradj='unknown' 告警",
     {**GOOD, "lradj": "unknown"},
     True, None, ["不是标准"])

test("17.2 lradj='type1' 合法",
     {**GOOD, "lradj": "type1"},
     True)

# ============================================================
# 18. 字符串数值兼容性
# ============================================================
section("18. 字符串数值兼容")

test("18.1 全部字符串数值（模拟 API JSON 输入）",
     {
         **GOOD,
         "seq_len": "96",
         "pred_len": "96",
         "batch_size": "32",
         "learning_rate": "0.001",
         "epochs": "50",
         "d_model": "512",
         "n_heads": "8",
         "e_layers": "2",
         "d_layers": "1",
         "d_ff": "2048",
         "dropout": "0.05",
         "patience": "5",
         "patch_len": "16",
         "stride": "8",
     },
     True)

test("18.2 非法字符串无法解析",
     {**GOOD, "seq_len": "abc"},
     False, ["无法解析"])

# ============================================================
# 19. 多错误同时出现
# ============================================================
section("19. 多错误叠加")

test("19.1 三个参数同时出错",
     {
         **GOOD,
         "model_name": "BadModel",
         "dataset": "BadData",
         "seq_len": -1,
     },
     False, ["不支持的模型", "不支持的数据集", "必须 ≥ 1"])

# ============================================================
# 汇总
# ============================================================
print(f"\n")
print(f"  {'=' * 60}")
print(f"  测试完成: ✅ {_pass} 通过, ❌ {_fail} 失败")
print(f"  {'=' * 60}")
print()

if _fail > 0:
    sys.exit(1)
