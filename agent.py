"""agent.py —— 简易 ReAct 任务规划 Agent（纯 Python 标准库，完全离线）。

ReAct = Reasoning + Acting，核心循环为：

    Thought（思考：当前要做什么）
        → Action（行动：调用工具）
        → Observation（观测：拿到工具返回结果）
        → Thought（基于观测继续推理，直到任务完成）

由于本项目不接入任何大模型 API，"思考"与"任务拆解"由一个确定性的
规则引擎（``Planner``）完成：通过关键词 / 正则识别意图并生成有序的
行动计划。该规划器与 LLM 规划器实现同一接口，未来可以无缝替换。

运行方式：
    python agent.py                      # 交互式模式
    python agent.py "请对 1、2、3 求和"   # 一次性任务模式
    python web.py                        # Web 界面模式（浏览器访问 127.0.0.1:8000）
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tools import (
    ToolError,
    ToolExecutionError,
    ToolParameterError,
    ToolRegistry,
    registry as default_registry,
)


# ---------------------------------------------------------------------------
# 一、数据结构
# ---------------------------------------------------------------------------


# 计划步骤参数中的"哨兵值"：执行时会被替换为上一步工具的 Observation，
# 用于实现步骤之间的数据传递（例如：先统计单词，再把统计结果保存到文件）。
OBSERVATION = object()


class PlanningError(Exception):
    """任务规划阶段失败。

    与 ToolParameterError 区分：前者发生在"还没调用工具之前"，
    表示任务无法被解析或缺少必要信息。
    """


@dataclass
class PlanStep:
    """计划中的单个步骤（一次 Thought + Action 的静态描述）。

    属性：
        thought: 本步思考，说明为什么要调用这个工具。
        tool_name: 要调用的工具名称。
        args: 调用参数；值若为哨兵 ``OBSERVATION``，
              执行时替换为上一步的观测结果。
    """

    thought: str
    tool_name: str
    args: Dict[str, Any]


@dataclass
class StepRecord:
    """一次完整 ReAct 迭代的运行记录（构成 Agent 的短期上下文记忆）。"""

    index: int
    thought: str
    tool_name: str
    args: Dict[str, Any]
    observation: str
    success: bool


# ---------------------------------------------------------------------------
# 二、Planner：基于规则的离线任务拆解器
# ---------------------------------------------------------------------------


class Planner:
    """规则式任务规划器。

    工作流程：
        1. 用关键词识别任务中包含的意图（求和 / 四则运算 / 单词统计 /
           时间查询 / 保存 / 闲聊问答，可组合）；
        2. 用正则从自然语言中抽取参数（数字、算式、引号文本、文件名）；
        3. 按"先计算、后保存"的顺序生成 ``PlanStep`` 列表；
        4. 问候 / 自我介绍等闲聊问题由 ``reply`` 工具直接给出预设答案。

    一个任务可以拆解出多个步骤，例如
    "统计文本 'hello world' 的单词数，并把结果保存到 r.txt" 会生成：
        Step1: count_words(text='hello world')
        Step2: save_to_txt(filename='r.txt', content=<上一步观测>)
    """

    # 意图关键词（小写匹配，中文原样）
    SUM_KEYWORDS = ("求和", "相加", "总和", "加起来", "sum", "add")
    COUNT_KEYWORDS = (
        "单词统计",
        "统计单词",
        "单词数",
        "多少个单词",
        "多少词",
        "词数",
        "字数",
        "word count",
        "wordcount",
        "count words",
    )
    SAVE_KEYWORDS = ("保存", "写入", "存到", "存为", "save")
    # 时间 / 日期关键词（"几点""几号""星期"等强特征词，避免误伤普通句子）
    TIME_KEYWORDS = (
        "几点", "日期", "几号", "多少号", "星期", "礼拜", "时间戳",
        "current time", "what time",
    )
    # 四则运算触发词：只有出现这些词才尝试提取算式，防止把日期等数字串误判
    CALC_KEYWORDS = ("计算", "算一下", "算算", "求值", "运算", "等于", "calculate")

    # 闲聊问答：按 身份 > 能力 > 致谢 > 问候 的优先级匹配
    IDENTITY_RE = re.compile(
        r"你是谁|你叫什么|你叫啥|你是什么|自我介绍|介绍.{0,6}自己|who\s+are\s+you",
        re.IGNORECASE,
    )
    CAPABILITY_RE = re.compile(
        r"你能做什么|你能干什么|你会什么|可以做什么|能做什么|能干嘛|会些什么"
        r"|有什么功能|有哪些功能|功能介绍|帮助|\bhelp\b",
        re.IGNORECASE,
    )
    THANKS_RE = re.compile(r"谢谢|感谢|\bthanks\b|thank\s+you", re.IGNORECASE)
    GREETING_RE = re.compile(
        r"你好|您好|嗨|早上好|中午好|下午好|晚上好|在吗|\bhello\b|\bhi\b",
        re.IGNORECASE,
    )

    # 预设问答文案（由 reply 工具原样返回）
    IDENTITY_ANSWER = (
        "我是简易 ReAct 任务规划 Agent：一个纯 Python 标准库实现、完全离线运行的"
        "智能体示例。我通过 思考(Thought) → 行动(Action) → 观测(Observation) 的"
        "循环完成任务，所有\"思考\"由内置规则引擎完成，不依赖任何大模型。"
    )
    CAPABILITY_ANSWER = (
        "我目前支持这些任务：\n"
        "1. 数字求和：请对 1、2、3 求和\n"
        "2. 四则运算：请计算 (3 + 5) * 2 等于多少\n"
        "3. 文本单词统计：统计文本 \"hello world\" 的单词数\n"
        "4. 时间查询：现在几点了 / 今天星期几\n"
        "5. 保存文本：把 \"你好\" 保存到 hello.txt\n"
        "6. 闲聊问答：你是谁 / 你能做什么 / 谢谢"
    )
    GREETING_ANSWER = (
        "你好！我是 ReAct Agent，可以帮你求和、四则运算、统计单词、查时间、"
        "保存文本。对我说\"你能做什么\"可以查看全部能力。"
    )
    THANKS_ANSWER = "不客气！还有其他任务随时吩咐。"

    # 数字（含小数、正负号）
    NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
    # 引号包裹的文本：同时兼容英文引号 "" 和中文引号 “”
    QUOTED_RE = re.compile(r'[“"「](.+?)[”"」]', re.DOTALL)
    # xxx.txt 文件名
    FILENAME_RE = re.compile(r"[\w\u4e00-\u9fff\-]+\.txt", re.UNICODE)
    # 任意带后缀的文件名，用于识别"用户显式指定了但不是 txt"的情况
    ANY_FILENAME_RE = re.compile(r"[\w\u4e00-\u9fff\-]+\.[A-Za-z0-9]{1,8}", re.UNICODE)

    def __init__(self) -> None:
        # 没匹配到任何意图时的兜底提示（优化点 3）
        self.fallback_message = (
            "无法解析该任务：我目前支持【数字求和 / 四则运算 / 文本单词统计 / "
            "时间日期查询 / 保存文本 / 问候与自我介绍】类任务，"
            "可输入 help 查看示例。"
        )

    # -- 对外入口 ----------------------------------------------------------

    def plan(self, task: str) -> Optional[List[PlanStep]]:
        """把自然语言任务拆解为有序的计划步骤。

        Args:
            task: 用户输入的自然语言任务。

        Returns:
            计划步骤列表；若任务不包含任何可识别意图，返回 ``None``，
            由 Agent 输出兜底提示（而不是让程序卡死）。

        Raises:
            PlanningError: 识别到了意图但抽取不到必要参数时抛出。
        """
        # lstrip("\ufeff") 用于去除某些管道输入带入的 UTF-8 BOM
        text = task.strip().lstrip("﻿")
        lowered = text.lower()

        # 天气等暂不支持的话题，直接走兜底，避免误判成其他意图
        if "天气" in lowered or "weather" in lowered:
            return None

        want_sum = any(keyword in lowered for keyword in self.SUM_KEYWORDS)
        want_count = any(keyword in lowered for keyword in self.COUNT_KEYWORDS)
        want_save = any(keyword in lowered for keyword in self.SAVE_KEYWORDS)
        want_time = any(keyword in lowered for keyword in self.TIME_KEYWORDS)
        chat_answer = self._match_chat(text)
        expression = self._extract_expression(text, lowered)

        # 兜底逻辑：所有意图一个都没识别出来
        if (
            not (want_sum or want_count or want_save or want_time)
            and chat_answer is None
            and expression is None
        ):
            return None

        steps: List[PlanStep] = []
        quoted_texts = self.QUOTED_RE.findall(text)

        # 步骤 1：时间日期查询（时间是动态数据，必须实时获取）
        if want_time:
            steps.append(
                PlanStep(
                    thought=(
                        "识别到时间/日期问答意图，当前时间需要实时获取，"
                        "下一步调用 get_datetime 工具。"
                    ),
                    tool_name="get_datetime",
                    args={},
                )
            )

        # 步骤 2：数字求和（若同时识别出四则算式，则交给计算器处理）
        if want_sum and expression is None:
            numbers = [float(n) for n in self.NUMBER_RE.findall(text)]
            if not numbers:
                raise PlanningError(
                    "识别到求和意图，但没有在任务中发现任何数字。"
                )
            steps.append(
                PlanStep(
                    thought=(
                        "这是一个数字求和任务，我已从任务中提取到 "
                        f"{len(numbers)} 个数字，下一步调用 sum_numbers 工具。"
                    ),
                    tool_name="sum_numbers",
                    args={"numbers": numbers},
                )
            )

        # 步骤 3：文本单词统计
        if want_count:
            target_text = self._extract_count_text(text, quoted_texts)
            steps.append(
                PlanStep(
                    thought=(
                        "这是一个文本单词统计任务，我已定位到待统计文本，"
                        "下一步调用 count_words 工具。"
                    ),
                    tool_name="count_words",
                    args={"text": target_text},
                )
            )

        # 步骤 4：四则运算（ast 白名单安全求值，不用 eval）
        if expression is not None:
            steps.append(
                PlanStep(
                    thought=(
                        "识别到四则运算意图，已从任务中提取出算式，"
                        "下一步调用 math_calc 工具安全求值。"
                    ),
                    tool_name="math_calc",
                    args={"expression": expression},
                )
            )

        # 步骤 5：闲聊问答（问候 / 自我介绍 / 能力 / 致谢，reply 直接回复）
        if chat_answer is not None:
            steps.append(
                PlanStep(
                    thought=(
                        "识别到闲聊问答意图，无需调用计算类工具，"
                        "下一步调用 reply 工具直接给出预设回复。"
                    ),
                    tool_name="reply",
                    args={"text": chat_answer},
                )
            )

        # 步骤 6：保存到文件（一定排在最后，便于把前面的观测结果写入文件）
        if want_save:
            filename = self._extract_filename(text)
            content = self._extract_save_content(
                text, quoted_texts, has_prior_step=bool(steps)
            )
            steps.append(
                PlanStep(
                    thought=(
                        "需要把结果持久化到本地 txt 文件，"
                        + (
                            "将上一步工具返回的观测结果作为文件内容，"
                            if content is OBSERVATION
                            else "已从任务中提取到要保存的文本内容，"
                        )
                        + f"下一步调用 save_to_txt 工具（文件名：{filename}）。"
                    ),
                    tool_name="save_to_txt",
                    args={"filename": filename, "content": content},
                )
            )

        return steps

    # -- 参数抽取辅助方法 --------------------------------------------------

    def _match_chat(self, text: str) -> Optional[str]:
        """按优先级匹配闲聊问答意图，命中则返回预设回复文本。

        优先级：身份 > 能力 > 致谢 > 问候
        （例如"你好，你能做什么"应优先给出能力介绍而不是单纯打招呼）。

        Args:
            text: 用户输入的原始任务文本。

        Returns:
            命中意图时返回预设回复文本，否则返回 None。
        """
        if self.IDENTITY_RE.search(text):
            return self.IDENTITY_ANSWER
        if self.CAPABILITY_RE.search(text):
            return self.CAPABILITY_ANSWER
        if self.THANKS_RE.search(text):
            return self.THANKS_ANSWER
        if self.GREETING_RE.search(text):
            return self.GREETING_ANSWER
        return None

    def _extract_expression(self, text: str, lowered: str) -> Optional[str]:
        """从任务中提取四则运算算式；没有计算类关键词时返回 None。

        提取前先把 × ÷ ^ 乘以 除以 加 减 等写法归一化成 Python 运算符，
        再截取由数字与运算符组成的片段，并排除形如日期的数字串。

        Args:
            text: 用户输入的原始任务文本。
            lowered: 已经 lower() 过的文本，用于关键词匹配。

        Returns:
            可交给 math_calc 的算式字符串；无法提取时返回 None。
        """
        if not any(keyword in lowered for keyword in self.CALC_KEYWORDS):
            return None

        s = text
        # 全角与中文运算写法归一化
        s = s.replace("×", "*").replace("÷", "/").replace("^", "**")
        s = s.replace("＋", "+").replace("－", "-")
        s = s.replace("（", "(").replace("）", ")")
        s = re.sub(r"(?<=\d)\s*(?:乘以|乘)\s*(?=\d)", "*", s)
        s = re.sub(r"(?<=\d)\s*(?:除以|除)\s*(?=\d)", "/", s)
        s = re.sub(r"(?<=\d)\s*加\s*(?=\d)", "+", s)
        s = re.sub(r"(?<=\d)\s*减\s*(?=\d)", "-", s)
        s = s.replace("=", " ")

        # 取最长的"数字+运算符"片段作为算式候选
        candidates = re.findall(r"[0-9.()\+\-*/%\s]+", s)
        if not candidates:
            return None
        expression = max(candidates, key=len).strip(" \t")
        expression = re.sub(r"[\s+\-*/%.]+$", "", expression).strip(" \t")

        # 必须同时含有数字与运算符才算算式；形如日期的串不算
        if not expression or not re.search(r"\d", expression):
            return None
        if not re.search(r"[+\-*/%]", expression):
            return None
        if re.fullmatch(r"\d{4}-\d{1,2}(-\d{1,2})?", expression):
            return None
        return expression

    def _extract_count_text(
        self, text: str, quoted_texts: List[str]
    ) -> str:
        """提取待统计的文本：优先取第一个引号片段，其次取冒号后的内容。"""
        if quoted_texts:
            return quoted_texts[0]
        # 兼容 "请统计下面文本的单词数：hello world" 这类写法
        if "：" in text or ":" in text:
            tail = re.split(r"[：:]", text, maxsplit=1)[-1].strip()
            if tail:
                return tail
        raise PlanningError(
            "识别到单词统计意图，但没有找到待统计的文本，"
            "建议用引号把文本包裹起来，例如：统计文本 \"hello world\" 的单词数。"
        )

    # 中文里动词与文件名之间没有空格（如"保存到笔记.txt"），
    # 提取后需要把匹配结果前面的动词 / 介词成分剥掉
    _VERB_PREFIX_RE = re.compile(
        r"^.*(?:保存到|保存为|保存成|保存|写入到|写入|存到|存为|存成|存|到|为|成)"
    )
    # 正文末尾可能附带"，并保存到 xxx.txt"子句，提取纯保存任务正文时裁掉
    _SAVE_CLAUSE_RE = re.compile(
        r"[，,]?\s*(?:并\s*)?(?:将其|把它|把结果)?"
        r"\s*(?:的文本|文本)?"
        r"\s*(?:保存到|保存为|保存成|保存|写入到|写入|存到|存为|存成|存)"
        r"\s*[\w\u4e00-\u9fff\-]*\.[A-Za-z0-9]{1,8}\s*[。.！!]?\s*$"
    )

    def _clean_filename(self, name: str) -> str:
        """剥离文件名候选前粘连的中文动词，如 "保存到笔记.txt" -> "笔记.txt"。"""
        return self._VERB_PREFIX_RE.sub("", name, count=1)

    def _extract_filename(self, text: str) -> str:
        """提取目标文件名；任务未指定时使用默认文件名。

        Raises:
            PlanningError: 用户显式指定了非 .txt 后缀的文件名。
        """
        match = self.FILENAME_RE.search(text)
        if match:
            return self._clean_filename(match.group(0))
        other = self.ANY_FILENAME_RE.search(text)
        if other:
            raise PlanningError(
                "只支持保存为 .txt 文件，"
                f"检测到不支持的文件名：{self._clean_filename(other.group(0))}。"
            )
        return "result.txt"

    def _extract_save_content(
        self,
        text: str,
        quoted_texts: List[str],
        has_prior_step: bool,
    ) -> Any:
        """提取要保存的内容。

        规则：
            - 如果任务里出现了多个引号片段（如 "统计 'A' 并保存 'B'"），
              最后一个片段视为显式指定的文件内容；
            - 如果前面已有计算步骤且没有额外的引号内容，
              则使用哨兵 OBSERVATION —— 保存上一步观测结果；
            - 单纯保存任务时，尝试取 "内容为/：" 之后的文本。
        """
        # 多个引号片段：最后一个是给保存动作用的
        if len(quoted_texts) >= 2:
            return quoted_texts[-1]
        if len(quoted_texts) == 1 and not has_prior_step:
            return quoted_texts[0]
        if has_prior_step:
            return OBSERVATION

        # 纯保存任务：仅匹配明确的 "内容为 xxx" / "内容是 xxx" / "内容：xxx"，
        # 避免把 "把内容保存到 note.csv" 这类句子误识别为文件正文
        match = re.search(r"内容(?:为|是|：|:)+\s*(.+)", text, re.DOTALL)
        if match and match.group(1).strip():
            # 裁掉正文尾部粘连的 "，并保存到 xxx.txt" 之类的子句
            content = self._SAVE_CLAUSE_RE.sub("", match.group(1)).strip()
            if content:
                return content
        raise PlanningError(
            "识别到保存意图，但没有找到要保存的文本内容，"
            "建议用引号包裹内容，例如：把 \"你好\" 保存到 hello.txt。"
        )


# ---------------------------------------------------------------------------
# 三、Agent：ReAct 核心循环
# ---------------------------------------------------------------------------


class Agent:
    """简易 ReAct Agent。

    协作关系：
        - ``Planner`` 负责"想"：把任务拆解成有序的 ``PlanStep``；
        - ``ToolRegistry`` 负责"做"：按名称找到并执行工具；
        - Agent 自身负责"循环"：逐步骤打印 Thought / Action / Observation，
          维护历史记录，并把上一步 Observation 注入下一步参数。
    """

    def __init__(
        self,
        registry: ToolRegistry,
        planner: Optional[Planner] = None,
        max_steps: int = 8,
    ) -> None:
        """初始化 Agent。

        Args:
            registry: 已注册好工具的工具注册表。
            planner: 任务规划器，为 None 时使用默认的规则式 ``Planner``。
            max_steps: 单次任务最大执行步数，防止意外的无限循环。
        """
        self.registry = registry
        self.planner = planner or Planner()
        self.max_steps = max_steps
        # 短期记忆：每执行一步就追加一条 StepRecord
        self.history: List[StepRecord] = []
        # 执行轨迹（结构化事件流）：与控制台打印同步记录，供 Web 界面渲染
        self.last_trace: List[Dict[str, Any]] = []

    def run(self, task: str) -> str:
        """执行一个完整任务，返回最终结论文本。

        Args:
            task: 用户输入的自然语言任务。

        Returns:
            最终答案（通常是最后一步工具的 Observation）；
            规划失败或执行报错时返回以错误标识开头的提示文本。
        """
        # 统一规范化输入：去首尾空白与管道输入可能带入的 UTF-8 BOM
        task = task.strip().lstrip("﻿")

        # 重置执行轨迹：每轮任务的事件流（供 Web 界面渲染时间线）
        self.last_trace = []

        print("=" * 70)
        print(f"任务：{task}")
        print("-" * 70)

        # ---------- Thought 阶段（全局）：任务拆解 ----------
        try:
            plan = self.planner.plan(task)
        except PlanningError as exc:
            # 识别到意图但参数不全，给出明确的修正建议
            print(f"Thought（规划失败）：{exc}")
            self.last_trace.append({"type": "plan_error", "message": str(exc)})
            return f"[规划失败] {exc}"

        # 兜底：完全无法理解任务，直接提示，避免卡死（优化点 3）
        if plan is None:
            print(f"Thought（无法理解）：{self.planner.fallback_message}")
            self.last_trace.append(
                {"type": "plan_error", "message": self.planner.fallback_message}
            )
            return f"[无法解析] {self.planner.fallback_message}"

        print(f"Thought（任务拆解）：本任务共拆解为 {len(plan)} 个步骤。")
        self.last_trace.append({"type": "plan", "steps": len(plan)})
        if len(plan) > self.max_steps:
            message = f"计划步数 {len(plan)} 超过上限 {self.max_steps}。"
            self.last_trace.append({"type": "plan_error", "message": message})
            return f"[步骤超限] {message}"

        # ---------- ReAct 循环：逐步 Thought → Action → Observation ----------
        last_observation = ""
        for index, step in enumerate(plan, start=1):
            print(f"\n--- 第 {index} 轮 ReAct ---")
            print(f"Thought: {step.thought}")

            # 参数绑定：把哨兵 OBSERVATION 替换为上一步真实观测
            bound_args = self._bind_arguments(step.args, last_observation)
            action_text = f"{step.tool_name}({self._render_args(bound_args)})"
            print(f"Action: {action_text}")

            try:
                tool = self.registry.get(step.tool_name)
                observation = tool.invoke(**bound_args)
            except ToolParameterError as exc:
                # 参数类错误：重试同一动作无意义，终止本轮任务（优化点 4）
                message = self._fail(index, step, bound_args, exc, "参数错误")
                self._emit_step(index, step.thought, action_text, message, False)
                return message
            except ToolExecutionError as exc:
                # 执行类错误：同样终止，但错误来源不同，单独标识（优化点 4）
                message = self._fail(index, step, bound_args, exc, "执行错误")
                self._emit_step(index, step.thought, action_text, message, False)
                return message
            except ToolError as exc:
                # 其他工具异常兜底
                message = self._fail(index, step, bound_args, exc, "工具错误")
                self._emit_step(index, step.thought, action_text, message, False)
                return message

            print(f"Observation: {observation}")
            last_observation = observation
            self.history.append(
                StepRecord(
                    index=index,
                    thought=step.thought,
                    tool_name=step.tool_name,
                    args=bound_args,
                    observation=observation,
                    success=True,
                )
            )
            self._emit_step(index, step.thought, action_text, observation, True)

        # ---------- 任务完成：输出最终答案 ----------
        print("\n" + "-" * 70)
        print(f"Final Answer: {last_observation}")
        print("=" * 70)
        self.last_trace.append({"type": "final", "answer": last_observation})
        return last_observation

    # -- 内部辅助方法 ------------------------------------------------------

    def _bind_arguments(
        self, args: Dict[str, Any], last_observation: str
    ) -> Dict[str, Any]:
        """把参数中的哨兵 ``OBSERVATION`` 替换为上一步观测结果。"""
        bound: Dict[str, Any] = {}
        for key, value in args.items():
            bound[key] = last_observation if value is OBSERVATION else value
        return bound

    def _emit_step(
        self,
        index: int,
        thought: str,
        action: str,
        observation: str,
        success: bool,
    ) -> None:
        """向 ``last_trace`` 追加一轮 ReAct 事件，供 Web 界面渲染时间线。"""
        self.last_trace.append(
            {
                "type": "step",
                "index": index,
                "thought": thought,
                "action": action,
                "observation": observation,
                "success": success,
            }
        )

    def _fail(
        self,
        index: int,
        step: PlanStep,
        bound_args: Dict[str, Any],
        exc: BaseException,
        label: str,
    ) -> str:
        """统一处理工具调用失败：记录历史并在 Observation 中输出清晰错误。"""
        message = f"{label}：{exc}"
        print(f"Observation: [错误] {message}")
        self.history.append(
            StepRecord(
                index=index,
                thought=step.thought,
                tool_name=step.tool_name,
                args=bound_args,
                observation=message,
                success=False,
            )
        )
        print("=" * 70)
        return f"[{label}] {message}"

    @staticmethod
    def _render_args(args: Dict[str, Any]) -> str:
        """把调用参数渲染成便于阅读的单行文本（长文本截断显示）。"""
        parts = []
        for key, value in args.items():
            text = str(value).replace("\n", " ")
            if len(text) > 40:
                text = text[:40] + "..."
            parts.append(f'{key}="{text}"')
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# 四、命令行入口
# ---------------------------------------------------------------------------

# 内置演示任务，覆盖单工具、多工具串联与问答场景
DEMO_TASKS: Dict[str, str] = {
    "1": "请对 12、7.5、3、20 这几个数字求和。",
    "2": '请统计下面文本的单词数："ReAct means Reasoning and Acting. 推理与行动结合。"',
    "3": '请统计文本 "Agent tools memory planning" 的单词数，并把结果保存到 result.txt。',
    "4": "请计算 (3 + 5) * 2 - 10 / 4 等于多少",
    "5": "现在几点了？今天星期几？",
    "6": "你能做什么？",
}

HELP_TEXT = """\
可用命令：
  1 - 6       运行内置演示任务（求和 / 统计 / 保存 / 四则运算 / 时间 / 能力问答）
  help        查看帮助与示例
  tools       查看已注册工具清单
  quit / exit 退出程序
也可以直接输入自然语言任务，例如：
  请对 1、2、3 求和
  请计算 (3 + 5) * 2 等于多少
  统计文本 "hello world from agent" 的单词数
  现在几点了
  你是谁 / 你能做什么
  把 "你好，ReAct" 保存到 hello.txt
  请对 10、20 求和，并把结果保存到 sum.txt\
"""


def main() -> None:
    """命令行入口：支持「一次性任务」和「交互式」两种模式。"""
    agent = Agent(registry=default_registry)

    # 一次性任务模式：python agent.py "任务内容"
    if len(sys.argv) > 1:
        agent.run(" ".join(sys.argv[1:]))
        return

    # 交互式模式
    print("=" * 70)
    print("简易 ReAct 任务规划 Agent（纯标准库 / 离线运行）")
    print("=" * 70)
    print(HELP_TEXT)

    while True:
        try:
            task = input("\n请输入任务> ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+C / Ctrl+D 优雅退出
            print("\n再见！")
            break

        if not task:
            continue
        if task.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if task.lower() == "help":
            print(HELP_TEXT)
            continue
        if task.lower() == "tools":
            print(default_registry.describe_all())
            continue
        if task in DEMO_TASKS:
            agent.run(DEMO_TASKS[task])
            continue

        agent.run(task)


if __name__ == "__main__":
    main()
