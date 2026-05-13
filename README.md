# 统一数据格式
{
  "status": "success",
  "agent": "plan",
  "task_id": "xxx",
  "agent_data": { 存放agent传递数据
    "intent": "",
    ......
    "work": "",
    "eval": ""
    "agent_params": {
        max_iteration 控制最大循环
        visualize 控制是否进行可视乎啊
    }，
    "agent_state": { 用于控制agent的状态参数
        iteration
    }，
    "history": [] 用于存储过去迭代的信息，包括参数、指标、以及eval的建议，便于后续summary总结，存入的是精简版快照
  },
  "errors": [], 错误信息存放
  "next_action": "work" 指导下一个agent
}


{
"plan": {
    "model_inital_params": {
        关键参数
        seq_len
        pred_len

        初始化模型的参数
    }，
    
    
    
},
"work": {
    "model_inital_params": {
        plan传递来的参数
    },
    "result": {
        这里存放模型返回的训练日志，可以直接读取csv/log日志
    }
}
}


# 智能体集群
plan：分析用户意图，结合RAG检索，给出初始化参数
work：接收plan或eval回传的json，根据里面的参数，回传后端API进行填写，并启动训练
eval：接收work的数据并分析，给出优化意见，重新矫正参数，同时控制最大迭代深度
summary：总结训练经验，存入RAG


# 智能体编排
典型的三个任务
* 1.普通训练任务，不自动迭代
plan给出参数 ——> work进行训练
结束后视情况可视化

* 2.训练任务，自动迭代
plan给出参数 ——> work进行训练 ——> eval评估，继续优化 ——> 达到轮数结束 ——> summary总结训练经验，存入RAG
结束后视情况可视化

* 3.推理任务
plan给出最近的相关模型加载地址 ——> work进行推理
结束后视情况可视化

# skills
RAG检索
模型检查点查找
训练/推理API调用
日志解析
可视化绘图
向量数据库读写
参数验证：检查参数是否合理
metric计算： mse、mae计算

# 大模型json output
设置 response_format 参数为 {'type': 'json_object'}。
用户传入的 system 或 user prompt 中必须含有 json 字样，并给出希望模型输出的 JSON 格式的样例，以指导模型来输出合法 JSON。
需要合理设置 max_tokens 参数，防止 JSON 字符串被中途截断。