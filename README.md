# 简易 ReAct 任务规划 Agent

一个**纯 Python 标准库、零第三方依赖、完全离线**的 ReAct Agent 教学项目。不依赖 LangChain，不调用任何大模型 API，用一个确定性的规则引擎替代 LLM 完成"思考与任务拆解"，帮助你从源码层面真正理解 ReAct（Reasoning + Acting）范式的核心循环。

## 一、项目介绍

ReAct 的核心思想是让智能体在 **Thought（思考）→ Action（调用工具）→ Observation（观测结果）** 的循环中完成任务：先想清楚下一步做什么，再调用工具执行，然后根据观测结果继续推理，直到产出最终答案。

本项目实现了这一完整闭环：

- **Thought（思考/规划）**：由 `Planner` 规则引擎完成。通过关键词识别意图、正则抽取参数，把一句自然语言拆解为有序的行动计划；支持把多个意图组合成多步任务。
- **Action（行动）**：Agent 通过 `ToolRegistry` 按名称查找并调用工具，参数通过函数签名做严格校验。
- **Observation（观测）**：工具统一返回字符串结果，写入 Agent 的历史记录（短期记忆），并可注入下一步的工具参数，实现步骤间数据传递。

### 内置六个工具

| 工具名 | 功能 | 说明 |
| --- | --- | --- |
| `sum_numbers` | 数字求和 | 支持数字列表或包含数字的字符串，自动提取 |
| `math_calc` | 四则运算计算器 | `ast` 白名单求值（不用 eval），支持 + - * / % ** 与括号 |
| `count_words` | 文本单词统计 | 英文单词数、中文字符数、总字符数、非空行数 |
| `get_datetime` | 时间日期查询 | 当前日期、时间、星期与 Unix 时间戳 |
| `save_to_txt` | 保存文本到本地 | 输出到 `output/` 目录，自动建目录，含路径穿越防护 |
| `reply` | 预设问答回复 | 问候 / 自我介绍 / 能力介绍 / 致谢等闲聊场景 |

### 问答能力（无需大模型）

除工具任务外，Agent 还支持规则式问答：

- **闲聊问答**：`你好` / `你是谁` / `你能做什么` / `谢谢` —— Planner 识别意图后由 `reply` 工具给出预设答案，仍走完整的 Thought → Action → Observation 流程；
- **实时问答**：`现在几点了` / `今天星期几` —— 由 `get_datetime` 实时获取；
- **计算问答**：`请计算 (3 + 5) * 2 等于多少`，也支持 `3 乘以 4` 这类中文算式写法。

### 多步任务示例

输入：`请统计文本 "Agent tools memory planning" 的单词数，并把结果保存到 result.txt`

Agent 会自动拆解为两轮 ReAct：

1. `count_words(text="Agent tools memory planning")` → Observation 得到统计结果
2. `save_to_txt(filename="result.txt", content=<上一步 Observation>)` → 结果落盘

## 二、项目结构

```
easyagent/
├── agent.py      # 主文件：Agent 类、Planner 规划器、ReAct 循环、CLI 入口
├── tools.py      # Tool / ToolRegistry 工具注册机制、分层异常、3 个内置工具
├── web.py        # Web 服务：标准库 http.server 暴露页面与 JSON 接口
├── index.html    # 前端页面：原生 HTML/CSS/JS 单文件，无任何 CDN 依赖
├── .gitignore    # 忽略 __pycache__、output/、虚拟环境、IDE 配置
└── README.md     # 项目说明文档
```

运行后工具生成的文件统一放在自动创建的 `output/` 目录中。

## 三、运行方式

环境要求：**Python 3.9+**，无需 `pip install` 任何包，无需联网，无需 API Key。

### 方式一：交互式模式

```bash
python agent.py
```

启动后可输入：

- `1` / `2` / `3`：运行内置演示任务
- 自然语言任务，例如 `请对 1、2、3 求和`
- `tools`：查看已注册工具清单
- `help`：查看帮助
- `quit`：退出

### 方式二：一次性任务模式

```bash
python agent.py "请对 12、7.5、3、20 求和"
python agent.py "把 \"你好，ReAct\" 保存到 hello.txt"
```

### 方式三：Web 界面（浏览器可视化 ReAct 过程）

```bash
python web.py          # 默认 8000 端口
python web.py 8080     # 指定端口
```

启动后浏览器访问 `http://127.0.0.1:8000`：输入自然语言任务（或点击示例按钮），页面会渲染每一轮的 Thought / Action / Observation 时间线与最终结论；顶部自动展示已注册工具清单。

接口约定（均为 JSON）：

| 接口 | 方法 | 说明 |
| --- | --- | --- |
| `/` | GET | 返回前端页面 |
| `/api/tools` | GET | 已注册工具清单（名称、描述、参数说明） |
| `/api/run` | POST | 请求体 `{"task": "..."}`，返回 `{"answer", "events"}` 结构化 ReAct 轨迹 |

## 四、项目技术栈

- **语言**：Python 3（仅使用标准库）
- **只用标准库**：`re`（正则意图识别与参数抽取）、`dataclasses`（计划步骤与运行记录建模）、`inspect`（工具函数签名校验）、`pathlib`（路径安全与文件写入）、`ast`（四则运算白名单安全求值）、`datetime`（时间日期查询）、`typing`（类型注解）、`sys`（命令行参数）、`http.server`（Web 服务与 JSON API）
- **前端**：原生 HTML / CSS / JavaScript 单文件（`index.html`），零框架、零 CDN、零构建，动态文本全部经 `textContent` 渲染以防止 XSS
- **设计模式**：
  - 注册表模式（Registry）+ 装饰器：工具即插即用
  - 策略模式：`Planner` 与 Agent 解耦，规则规划器可整体替换为 LLM 规划器
  - 分层异常：`ToolError → ToolParameterError / ToolExecutionError`

## 五、面试可以讲解的要点

### 1. ReAct 范式的本质是什么？

ReAct = Reasoning + Acting，关键不是"提示词格式"，而是**闭环控制**：模型/规划器输出动作 → 程序负责真实执行并注入 Observation → 下一步推理必须基于真实观测而不是臆想结果。本项目中 `Planner.plan()` 负责推理产物（`PlanStep`），`Agent.run()` 负责执行循环与观测注入，两者职责分离。

### 2. Agent 和普通脚本的区别？

普通脚本是"写死的固定流程"；Agent 多了三件事：

1. **任务规划**：根据输入动态决定调用哪些工具、什么顺序（`Planner`）；
2. **工具选择**：通过名称在注册表里动态查找工具，而不是硬编码 if-else 调函数；
3. **观测驱动**：上一步 Observation 可以成为下一步的输入（哨兵 `OBSERVATION` 注入机制），这是多步任务串联的基础。

### 3. 工具注册机制是怎么设计的？（开闭原则）

新增工具只需写一个函数并加上 `@registry.register(...)` 装饰器，**Agent 主循环一行代码都不用改**——对扩展开放、对修改关闭。`Tool.invoke()` 在执行前用 `inspect.signature().bind()` 做参数校验，把"参数错误"和"执行错误"分成两类异常，错误信息会明确呈现在 Observation 中。

### 4. 没有大模型，"思考"如何实现？这个设计如何演进？

`Planner` 是一个确定性规则引擎：关键词识别意图、正则抽取参数、按"先计算后保存"排序生成步骤。它与 Agent 之间只通过 `plan(task) -> List[PlanStep]` 这一个接口协作。接入 LLM 时，只需新写一个 `LLMPlanner`（让模型按约定 JSON/格式输出步骤列表）注入 `Agent`，ReAct 循环、工具层、异常处理都可以原样复用——这就是面向接口设计的价值。

### 5. 工程细节亮点

- **兜底逻辑**：任务无法匹配任何规则时返回明确提示（"无法解析该任务"），而不是卡死或崩溃；
- **步数上限**：`max_steps` 防止异常计划导致无限循环；
- **异常分层**：参数类错误（重试无意义，直接终止）与执行类错误分别捕获，Observation 中错误来源清晰，方便调试；
- **文件安全**：保存工具只接受纯文件名（`Path(...).name` 剥离目录成分）+ 文件名白名单 + 固定输出目录，防止路径穿越；
- **安全计算**：四则运算用 `ast` 白名单求值而不是 `eval`，杜绝任意代码执行，并对幂运算指数设上限防卡死；
- **跨平台**：统一 UTF-8 读写，`pathlib` 处理路径，Windows / macOS / Linux 行为一致；
- **可观测性**：完整打印每一轮 Thought / Action / Observation / Final Answer，这正是调试 Agent 的标准姿势；
- **双入口架构**：CLI 与 Web 共用同一个 Agent 内核——`Agent.run()` 在打印控制台日志的同时产出结构化事件流（`last_trace`），`web.py` 用标准库 `http.server` 将其暴露为 JSON 接口，前端原生 JS 渲染 ReAct 时间线，无需任何前端框架。

### 6. 可以延伸的演进方向（被问"如何改进"时回答）

- 用 LLMPlanner 替换规则 Planner，工具清单自动注入系统提示词；
- 增加执行期反思（Reflection）：Observation 失败后自动修正参数重试；
- 扩展工具集（HTTP 请求、计算器、数据库查询），由于注册机制存在，扩展成本极低；
- 把 `StepRecord` 短期记忆持久化，支持跨会话的长任务。

## 六、免责声明

本项目仅用于学习 ReAct 思想与 Agent 工程结构，代码中不包含任何密钥或外部服务调用。
