"""
Param Validate Skill — 参数验证

职责：
- 校验模型参数的合理性和合法性
- 检查参数类型、取值范围、互斥条件

输入：params (dict)
输出：{"valid": bool, "errors": list[str], "warnings": list[str]}
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


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

    支持的模型：[DLinear, PatchTST, Autoformer, Informer, Transformer, Linear, NLinear]
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

支持的模型：DLinear, PatchTST, Autoformer, Informer, Transformer, Linear, NLinear
支持的数据集：ETTh1, ETTh2, ETTm1, ETTm2, weather, electricity, traffic, exchange_rate, national_illness"""

    SUPPORTED_MODELS = {
        "DLinear", "PatchTST", "Autoformer", "Informer",
        "Transformer", "Linear", "NLinear",
    }
    SUPPORTED_DATASETS = {
        "ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather",
        "electricity", "traffic", "exchange_rate", "national_illness",
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
        errors, warnings = [], []

        # model_name 校验
        model = params.get("model_name", "")
        if model and model not in self.SUPPORTED_MODELS:
            errors.append(f"不支持的模型: {model}，可选: {self.SUPPORTED_MODELS}")

        # dataset 校验
        dataset = params.get("dataset", "")
        if dataset and dataset not in self.SUPPORTED_DATASETS:
            errors.append(f"不支持的数据集: {dataset}，可选: {self.SUPPORTED_DATASETS}")

        # seq_len 校验
        seq_len = params.get("seq_len")
        if seq_len is not None:
            if not isinstance(seq_len, int) or seq_len < 1:
                errors.append(f"seq_len 必须为正整数，实际: {seq_len}")
            elif seq_len > 1024:
                warnings.append(f"seq_len={seq_len} 较大，可能导致显存不足")

        # pred_len 校验
        pred_len = params.get("pred_len")
        if pred_len is not None:
            if not isinstance(pred_len, int) or pred_len < 1:
                errors.append(f"pred_len 必须为正整数，实际: {pred_len}")

        # batch_size 校验
        batch_size = params.get("batch_size")
        if batch_size is not None:
            if not isinstance(batch_size, int) or batch_size < 1:
                errors.append(f"batch_size 必须为正整数，实际: {batch_size}")

        # learning_rate 校验
        lr = params.get("learning_rate")
        if lr is not None:
            if not isinstance(lr, (int, float)) or lr <= 0 or lr >= 1:
                errors.append(f"learning_rate 必须在 (0, 1) 之间，实际: {lr}")

        # epochs 校验
        epochs = params.get("epochs")
        if epochs is not None:
            if not isinstance(epochs, int) or epochs < 1:
                errors.append(f"epochs 必须为正整数，实际: {epochs}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }
