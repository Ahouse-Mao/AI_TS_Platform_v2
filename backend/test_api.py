"""
后端 API 测试脚本

测试所有已注册的 API 端点：
  - GET  /                 根路径信息
  - GET  /health           健康检查
  - POST /api/train        创建训练任务
  - GET  /api/status/{id}  查询任务状态
  - GET  /api/tasks        列出所有任务
  - POST /api/infer        创建推理任务

用法：
  # 先启动服务（另一个终端）
  uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000

  # 运行测试
  uv run python backend/test_api.py
  uv run python backend/test_api.py --epochs 2       # 快速训练（2轮）
  uv run python backend/test_api.py --host 127.0.0.1 --port 8000
"""

import argparse
import sys
import time
import json
from urllib.parse import urljoin

import requests

# ============================================================
# 常量
# ============================================================
POLL_INTERVAL = 3        # 轮询间隔（秒）
MAX_POLL_TIME = 600      # 最长等待（秒）


# ============================================================
# 测试客户端
# ============================================================

class APITester:
    """封装所有 API 测试用例"""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        # 测试结果统计
        self.passed = 0
        self.failed = 0

    # --------------------------------------------------------
    # 辅助方法
    # --------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        """发起请求并检查 HTTP 状态码"""
        url = urljoin(self.base_url, path)
        resp = self.session.request(method, url, timeout=30, **kwargs)
        assert resp.status_code < 500, (
            f"[{method}] {path} 返回 {resp.status_code}: {resp.text[:200]}"
        )
        return resp

    def _check(self, name: str, condition: bool, detail: str = "") -> None:
        """记录一条测试断言结果"""
        if condition:
            self.passed += 1
            print(f"  ✅ {name}")
        else:
            self.failed += 1
            print(f"  ❌ {name} — {detail}")

    def _print_result(self):
        """打印当前测试汇总"""
        total = self.passed + self.failed
        print(f"\n  📊 进度: {total} 项 | ✅ {self.passed} 通过 | ❌ {self.failed} 失败")

    # --------------------------------------------------------
    # 测试用例
    # --------------------------------------------------------

    def test_root(self):
        """GET / — 根路径信息"""
        print("\n🔍 1. 测试根路径 GET /")
        resp = self._request("GET", "/")
        data = resp.json()
        self._check("返回 service 字段", "service" in data)
        self._check("包含 endpoints 说明", "endpoints" in data)
        self._check("状态码 200", resp.status_code == 200)
        self._print_result()
        return data

    def test_health(self):
        """GET /health — 健康检查"""
        print("\n🔍 2. 测试健康检查 GET /health")
        resp = self._request("GET", "/health")
        data = resp.json()
        self._check("返回 status=ok", data.get("status") == "ok")
        self._check("状态码 200", resp.status_code == 200)
        self._print_result()

    def test_train(self, epochs: int = 2, fast: bool = True):
        """POST /api/train — 创建训练任务并轮询到完成"""
        print(f"\n🔍 3. 测试训练 POST /api/train (epochs={epochs})")
        payload = {
            "model_name": "DLinear",
            "dataset": "ETTh1",
            "seq_len": 96,
            "pred_len": 96,
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": epochs,
            "patience": 3,
        }
        if fast:
            # 快速模式：关掉一些不必要的开销
            payload.update({
                "use_amp": False,
                "do_predict": False,
            })

        resp = self._request("POST", "/api/train", json=payload)
        data = resp.json()

        self._check("返回 task_id", bool(data.get("task_id")))
        self._check("状态为 running/pending", data.get("status") in ("running", "pending"))
        self._check("状态码 200", resp.status_code == 200)

        task_id = data["task_id"]
        print(f"     📌 task_id = {task_id}")
        self._print_result()

        # 轮询到完成
        print(f"\n🔍 3b. 轮询训练任务 GET /api/status/{task_id}")
        task_result = self._poll_task(task_id)
        return task_result

    def test_inference(self, checkpoint_path: str | None = None):
        """POST /api/infer — 创建推理任务并轮询到完成"""
        print(f"\n🔍 4. 测试推理 POST /api/infer")
        payload = {
            "model_name": "DLinear",
            "dataset": "ETTh1",
            "seq_len": 96,
            "pred_len": 96,
        }
        if checkpoint_path:
            payload["checkpoint_path"] = checkpoint_path

        resp = self._request("POST", "/api/infer", json=payload)
        data = resp.json()

        self._check("返回 task_id", bool(data.get("task_id")))
        self._check("状态为 running/pending", data.get("status") in ("running", "pending"))
        self._check("状态码 200", resp.status_code == 200)

        task_id = data["task_id"]
        print(f"     📌 task_id = {task_id}")
        self._print_result()

        # 轮询到完成
        print(f"\n🔍 4b. 轮询推理任务 GET /api/status/{task_id}")
        task_result = self._poll_task(task_id)
        return task_result

    def test_list_tasks(self):
        """GET /api/tasks — 列出所有任务"""
        print("\n🔍 5. 测试任务列表 GET /api/tasks")
        resp = self._request("GET", "/api/tasks")
        tasks = resp.json()
        self._check("返回列表", isinstance(tasks, list))
        self._check(f"包含 {len(tasks)} 个任务", len(tasks) > 0)
        if tasks:
            self._check("每个任务有 task_id", all("task_id" in t for t in tasks))
            self._check("每个任务有 status", all("status" in t for t in tasks))
        self._check("状态码 200", resp.status_code == 200)
        self._print_result()
        return tasks

    # --------------------------------------------------------
    # 内部方法
    # --------------------------------------------------------

    def _poll_task(self, task_id: str) -> dict:
        """轮询任务直到完成/失败"""
        start = time.time()
        last_status = ""

        while True:
            elapsed = time.time() - start
            if elapsed > MAX_POLL_TIME:
                print(f"     ⏰ 轮询超时（{MAX_POLL_TIME}s）")
                self.failed += 1
                return {"status": "timeout"}

            resp = self._request("GET", f"/api/status/{task_id}")
            data = resp.json()
            cur = data.get("status", "")

            if cur != last_status and cur:
                pct = data.get("progress", 0)
                print(f"     ⏳ {cur} … ({pct}%)")
                last_status = cur

            if cur == "completed":
                self._check("训练/推理完成", True)
                self._check("有 checkpoint_path", bool(data.get("checkpoint_path")))
                self._check("有 metrics", bool(data.get("metrics")))
                if data.get("metrics"):
                    m = data["metrics"]
                    self._check("metrics 包含 mse", "mse" in m)
                    self._check("metrics 包含 mae", "mae" in m)
                print(f"     📊 metrics: {json.dumps(data.get('metrics', {}), indent=6)}")
                self._print_result()
                return data

            if cur == "failed":
                err = data.get("error", "未知错误")
                print(f"     ❌ 任务失败: {err}")
                self._check("推理/训练不报错", False, detail=err)
                self._print_result()
                return data

            time.sleep(POLL_INTERVAL)

    # --------------------------------------------------------
    # 汇总
    # --------------------------------------------------------

    def summary(self):
        """打印最终测试结果"""
        total = self.passed + self.failed
        print("\n" + "=" * 50)
        print(f"  测试完成: {total} 项")
        print(f"  ✅ 通过: {self.passed}")
        print(f"  ❌ 失败: {self.failed}")
        print("=" * 50)
        return self.failed == 0


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AI-TS-Platform 后端 API 测试")
    parser.add_argument("--host", default="0.0.0.0", help="后端地址 (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="后端端口 (default: 8000)")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数 (default: 2)")
    parser.add_argument("--no-train", action="store_true", help="跳过训练测试")
    parser.add_argument("--no-infer", action="store_true", help="跳过推理测试")
    parser.add_argument("--checkpoint", default=None, help="推理用的检查点路径（可选）")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print(f"🌐 后端地址: {base_url}")
    print(f"🏋️  训练轮数: {args.epochs}")

    # 先检查服务是否可达
    try:
        requests.get(f"{base_url}/health", timeout=5)
    except requests.ConnectionError:
        print(f"\n❌ 无法连接到 {base_url}，请确保服务已启动：")
        print(f"   uv run uvicorn backend.app:app --host {args.host} --port {args.port}")
        sys.exit(1)

    tester = APITester(base_url)

    # ---- 执行测试 ----
    tester.test_root()
    tester.test_health()

    train_result = None
    if not args.no_train:
        train_result = tester.test_train(epochs=args.epochs)

    if not args.no_infer:
        cp = args.checkpoint
        if not cp and train_result:
            cp = train_result.get("checkpoint_path")
        tester.test_inference(checkpoint_path=cp)

    tester.test_list_tasks()

    # ---- 汇总 ----
    ok = tester.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
