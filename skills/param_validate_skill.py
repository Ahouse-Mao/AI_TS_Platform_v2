"""
Param Validate Skill — 参数验证

职责：
- 校验模型参数的合理性和合法性
- 检查参数类型、取值范围、互斥条件

输入：params (dict)
输出：{"valid": bool, "errors": list[str], "warnings": list[str]}
"""

import math
import logging
from typing import Any

logger = logging.getLogger(__name__)

# 类型转换辅助：尝试将传入的值转为 float/int，失败返回 None
def _to_float(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except (ValueError, TypeError):
            return None
    return None

def _to_int(v: Any) -> int | None:
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v) if v == int(v) else None
    if isinstance(v, str):
        try:
            return int(v)
        except (ValueError, TypeError):
            return None
    return None


class ParamValidateSkill:
    """
    参数验证技能

    Prompt（供 LLM 调用时使用）:
    ---
    你负责验证模型训练参数的合理性：
    1. seq_len >= 1 且通常 <= 1024
    2. pred_len >= 1 且通常 <= seq_len * 2
    3. batch_size >= 1 且为 2 的幂次
    4. learning_rate 在 (0, 1) 之间，通常 1e-5 ~ 1e-2
    5. epochs >= 1
    6. model_name 必须在支持的模型列表中
    7. dataset 必须在可用数据集列表中
    8. d_model 为正整数，建议 2 的幂次
    9. n_heads 为正整数，且 d_model % n_heads == 0
    10. e_layers / d_layers >= 1
    11. dropout 在 [0, 1) 之间
    12. features 必须为 "M" / "S" / "MS" 之一
    13. patience >= 1
    14. use_gpu 为布尔值
    15. patch_len / stride >= 1（仅 PatchTST 模型）

    支持的模型：[DLinear, PatchTST, Autoformer, Informer, Transformer, Linear, NLinear, ModernTCN]
    支持的数据集：[ETTh1, ETTh2, ETTm1, ETTm2, weather, electricity, traffic, exchange_rate, national_illness]
    ---
    """

    PROMPT = """验证模型训练参数的合理性：
1. seq_len >= 1 且通常 <= 1024
2. pred_len >= 1
3. batch_size >= 1 且为 2 的幂次
4. learning_rate 在 (0, 1) 之间（通常 1e-5 ~ 1e-2）
5. epochs >= 1
6. model_name 在支持列表中
7. dataset 在可用数据集列表中

支持的模型：DLinear, PatchTST, Autoformer, Informer, Transformer, Linear, NLinear, ModernTCN
支持的数据集：ETTh1, ETTh2, ETTm1, ETTm2, weather, electricity, traffic, exchange_rate, national_illness"""

    SUPPORTED_MODELS = {
        "DLinear", "PatchTST", "Autoformer", "Informer",
        "Transformer", "Linear", "NLinear", "ModernTCN",
    }
    SUPPORTED_DATASETS = {
        "ETTh1", "ETTh2", "ETTm1", "ETTm2",
        "weather", "WTH",
        "electricity", "ECL",
        "traffic",
        "exchange_rate", "Exchange",
        "national_illness", "ili", "ILI",
    }
    # 模型名 → 所需额外参数
    _MODEL_REQUIRED_EXTRA = {
        "PatchTST": ["patch_len", "stride"],
    }

    def validate(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        验证参数合法性

        Args:
            params: 模型参数字典，包含 model_name, dataset, seq_len, pred_len 等

        Returns:
            {
                "valid": bool,
                "errors": list[str],    # 致命错误
                "warnings": list[str],  # 警告（不阻止执行）
            }
        """
        logger.info(f"[ParamValidateSkill] 验证参数: {list(params.keys())}")
        errors: list[str] = []
        warnings: list[str] = []

        # ============================================================
        # 1. model_name
        # ============================================================
        model = params.get("model_name")
        if not model:
            errors.append("缺少必填参数: model_name")
        elif not isinstance(model, str):
            errors.append(f"model_name 必须为字符串，实际类型: {type(model).__name__}")
        elif model not in self.SUPPORTED_MODELS:
            supported = ", ".join(sorted(self.SUPPORTED_MODELS))
            errors.append(f"不支持的模型: {model}，可选: {supported}")

        # ============================================================
        # 2. dataset
        # ============================================================
        dataset = params.get("dataset")
        if not dataset:
            errors.append("缺少必填参数: dataset")
        elif not isinstance(dataset, str):
            errors.append(f"dataset 必须为字符串，实际类型: {type(dataset).__name__}")
        elif dataset not in self.SUPPORTED_DATASETS:
            supported = ", ".join(sorted(self.SUPPORTED_DATASETS))
            errors.append(f"不支持的数据集: {dataset}，可选: {supported}")

        # ============================================================
        # 3. seq_len
        # ============================================================
        seq_len = _to_int(params.get("seq_len"))
        if seq_len is None:
            errors.append("缺少或无法解析必填参数: seq_len (应为正整数)")
        else:
            if seq_len < 1:
                errors.append(f"seq_len 必须 ≥ 1，实际: {seq_len}")
            elif seq_len > 2048:
                warnings.append(f"seq_len={seq_len} 非常大，可能导致显存不足（建议 ≤ 1024）")
            elif seq_len > 1024:
                warnings.append(f"seq_len={seq_len} 较大，可能导致显存不足")

        # ============================================================
        # 4. pred_len
        # ============================================================
        pred_len = _to_int(params.get("pred_len"))
        if pred_len is None:
            errors.append("缺少或无法解析必填参数: pred_len (应为正整数)")
        else:
            if pred_len < 1:
                errors.append(f"pred_len 必须 ≥ 1，实际: {pred_len}")
            # 与 seq_len 交叉校验
            if seq_len is not None and seq_len >= 1 and pred_len > seq_len * 2:
                warnings.append(
                    f"pred_len={pred_len} 远超 seq_len={seq_len} 的 2 倍，"
                    f"预测质量可能大幅下降"
                )
            elif seq_len is not None and seq_len >= 1 and pred_len > seq_len:
                warnings.append(
                    f"pred_len={pred_len} > seq_len={seq_len}，部分模型可能不支持外推预测"
                )

        # ============================================================
        # 5. batch_size
        # ============================================================
        batch_size = _to_int(params.get("batch_size"))
        if batch_size is not None:
            if batch_size < 1:
                errors.append(f"batch_size 必须为正整数，实际: {batch_size}")
            elif batch_size & (batch_size - 1) != 0:
                warnings.append(f"batch_size={batch_size} 不是 2 的幂次，建议使用 16/32/64/128/256")

        # ============================================================
        # 6. learning_rate
        # ============================================================
        lr = _to_float(params.get("learning_rate"))
        if lr is not None:
            if lr <= 0 or lr >= 1:
                errors.append(f"learning_rate 必须在 (0, 1) 之间，实际: {lr}")
            else:
                if lr > 0.01:
                    warnings.append(f"learning_rate={lr} 偏大（常规范围 1e-5 ~ 1e-2），训练可能不稳定")
                elif lr < 1e-5:
                    warnings.append(f"learning_rate={lr} 偏小（常规范围 1e-5 ~ 1e-2），收敛可能很慢")

        # ============================================================
        # 7. epochs
        # ============================================================
        epochs = _to_int(params.get("epochs"))
        if epochs is not None:
            if epochs < 1:
                errors.append(f"epochs 必须为正整数，实际: {epochs}")
            elif epochs > 500:
                warnings.append(f"epochs={epochs} 较大，请确保 patience 设置了早停")

        # ============================================================
        # 8. d_model
        # ============================================================
        d_model = _to_int(params.get("d_model"))
        if d_model is not None:
            if d_model < 1:
                errors.append(f"d_model 必须为正整数，实际: {d_model}")
            elif d_model & (d_model - 1) != 0:
                warnings.append(f"d_model={d_model} 不是 2 的幂次，建议使用 64/128/256/512/1024")

        # ============================================================
        # 9. n_heads（需与 d_model 联检）
        # ============================================================
        n_heads = _to_int(params.get("n_heads"))
        if n_heads is not None:
            if n_heads < 1:
                errors.append(f"n_heads 必须为正整数，实际: {n_heads}")
            elif d_model is not None and d_model >= 1:
                if d_model % n_heads != 0:
                    errors.append(
                        f"n_heads={n_heads} 不能整除 d_model={d_model}，"
                        f"注意力机制无法分配"
                    )

        # ============================================================
        # 10. e_layers / d_layers
        # ============================================================
        for key, label in [("e_layers", "编码器层数"), ("d_layers", "解码器层数")]:
            val = _to_int(params.get(key))
            if val is not None:
                if val < 1:
                    errors.append(f"{label} ({key}) 必须 ≥ 1，实际: {val}")
                elif val > 10:
                    warnings.append(f"{label} ({key})={val} 较深，训练时间可能较长")

        # ============================================================
        # 11. d_ff
        # ============================================================
        d_ff = _to_int(params.get("d_ff"))
        if d_ff is not None and d_ff < 1:
            errors.append(f"d_ff 必须为正整数，实际: {d_ff}")

        # ============================================================
        # 12. dropout
        # ============================================================
        dropout = _to_float(params.get("dropout"))
        if dropout is not None:
            if dropout < 0 or dropout >= 1:
                errors.append(f"dropout 必须在 [0, 1) 之间，实际: {dropout}")
            elif dropout > 0.5:
                warnings.append(f"dropout={dropout} 较大，可能导致欠拟合")

        # ============================================================
        # 13. features
        # ============================================================
        features = params.get("features")
        if features is not None:
            if features not in ("M", "S", "MS"):
                errors.append(f"features 必须为 'M' / 'S' / 'MS' 之一，实际: {features}")

        # ============================================================
        # 14. patience
        # ============================================================
        patience = _to_int(params.get("patience"))
        if patience is not None and patience < 1:
            errors.append(f"patience 必须 ≥ 1，实际: {patience}")

        # ============================================================
        # 15. use_gpu
        # ============================================================
        use_gpu = params.get("use_gpu")
        if use_gpu is not None and not isinstance(use_gpu, bool):
            errors.append(f"use_gpu 必须为布尔值，实际类型: {type(use_gpu).__name__}")

        # ============================================================
        # 16. patch_len / stride（仅 PatchTST 相关）
        # ============================================================
        if model == "PatchTST":
            for key, label in [("patch_len", "Patch 长度"), ("stride", "Patch 步长")]:
                val = _to_int(params.get(key))
                if val is None:
                    errors.append(f"模型 {model} 缺少必填参数: {key}")
                elif val < 1:
                    errors.append(f"{label} ({key}) 必须 ≥ 1，实际: {val}")

        # ============================================================
        # 17. lradj
        # ============================================================
        lradj = params.get("lradj")
        if lradj is not None and lradj not in ("type1", "type2", "type3", "cosine"):
            warnings.append(f"lradj='{lradj}' 不是标准学习率策略（type1/type2/type3/cosine）")

        # ============================================================
        # 汇总
        # ============================================================
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }
