"""tools.py —— 内置工具集合。

本模块包含三部分内容：

1. 分层异常体系：``ToolError`` / ``ToolParameterError`` / ``ToolExecutionError``，
   用于区分"参数解析失败"和"工具执行失败"，方便 Agent 在 Observation 中
   给出清晰的错误信息。
2. ``Tool`` 数据类与 ``ToolRegistry`` 工具注册表：通过装饰器即可注册新工具，
   遵循"对扩展开放、对修改关闭"的开闭原则。
3. 内置工具：
   - ``sum_numbers``   数字求和
   - ``math_calc``     四则运算安全计算器（ast 解析，不用 eval）
   - ``count_words``   文本单词统计
   - ``get_datetime``  当前日期时间查询
   - ``save_to_txt``   保存文本到本地 txt 文件
   - ``reply``         预设文本回复（问候 / 自我介绍等问答场景）

全程只使用 Python 标准库（re / pathlib / inspect / dataclasses / typing）。
"""

from __future__ import annotations

import ast
import inspect
import operator
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# 一、分层异常体系
# ---------------------------------------------------------------------------


class ToolError(Exception):
    """所有工具相关异常的基类。"""


class ToolParameterError(ToolError):
    """参数解析 / 参数校验失败时抛出。

    典型场景：调用求和工具却没有提供任何数字、保存文件时缺少文件名等。
    这类错误属于"调用方式不对"，重试同一个动作通常没有意义。
    """


class ToolExecutionError(ToolError):
    """工具内部执行失败时抛出。

    典型场景：写文件时磁盘不可用、运行期发生未预期的异常等。
    """


# ---------------------------------------------------------------------------
# 二、Tool 数据类与工具注册表
# ---------------------------------------------------------------------------


@dataclass
class Tool:
    """对一个可被 Agent 调用的工具的元数据与执行逻辑的封装。

    属性：
        name: 工具唯一名称，Planner 在 Action 阶段通过该名称查找工具。
        description: 工具的自然语言描述，说明"这个工具能做什么"。
        params_desc: 参数名 -> 参数说明，用于帮助理解每个参数的含义。
        handler: 实际执行逻辑的可调用对象，返回值必须是字符串（作为 Observation）。
    """

    name: str
    description: str
    params_desc: Dict[str, str]
    handler: Callable[..., str]

    def invoke(self, **kwargs: Any) -> str:
        """按关键字参数调用工具。

        执行前先用函数签名做参数绑定校验，参数不匹配时抛出
        ``ToolParameterError``；执行过程中的其他异常统一包装成
        ``ToolExecutionError``，并保留原始异常链（``from e``）便于调试。

        Args:
            **kwargs: 传给底层 handler 的关键字参数。

        Returns:
            handler 执行后返回的字符串结果（即 ReAct 中的 Observation）。

        Raises:
            ToolParameterError: 参数数量 / 名称与工具签名不一致时抛出。
            ToolExecutionError: 工具内部执行抛出任何其他异常时抛出。
        """
        signature = inspect.signature(self.handler)
        try:
            # bind() 会按形参名做严格校验，多传、漏传都会抛 TypeError
            signature.bind(**kwargs)
        except TypeError as exc:
            raise ToolParameterError(
                f"工具 [{self.name}] 参数不匹配：{exc}；"
                f"期望参数：{list(self.params_desc.keys())}"
            ) from exc

        try:
            return self.handler(**kwargs)
        except ToolError:
            # 业务层主动抛出的参数 / 执行异常直接向上传播，不再包装
            raise
        except Exception as exc:  # noqa: BLE001 - 工具层需要兜底所有未知异常
            raise ToolExecutionError(
                f"工具 [{self.name}] 执行失败：{type(exc).__name__}: {exc}"
            ) from exc


class ToolRegistry:
    """工具注册表：集中管理所有可被 Agent 调用的工具。

    典型用法::

        registry = ToolRegistry()

        @registry.register(
            name="sum_numbers",
            description="对一组数字求和",
            params_desc={"numbers": "数字列表或包含数字的字符串"},
        )
        def sum_numbers(numbers):
            ...
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        params_desc: Optional[Dict[str, str]] = None,
    ) -> Callable[[Callable[..., str]], Callable[..., str]]:
        """装饰器：把一个普通函数注册为工具。

        Args:
            name: 工具唯一名称，重复注册会抛出 ``ValueError``。
            description: 工具能力描述。
            params_desc: 参数说明字典；为 None 时根据函数签名自动生成空说明。

        Returns:
            原封不动的被装饰函数（装饰器不改变函数行为，只做登记）。
        """

        def decorator(func: Callable[..., str]) -> Callable[..., str]:
            if name in self._tools:
                raise ValueError(f"工具名称重复注册：{name}")
            # 未显式提供参数说明时，用函数签名中的形参名兜底
            desc = params_desc or {
                param: "（未提供参数说明）"
                for param in inspect.signature(func).parameters
            }
            self._tools[name] = Tool(
                name=name,
                description=description,
                params_desc=desc,
                handler=func,
            )
            return func

        return decorator

    def get(self, name: str) -> Tool:
        """按名称获取工具，不存在时抛出 ``ToolParameterError``。"""
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self._tools) or "（注册表为空）"
            raise ToolParameterError(
                f"未找到工具 [{name}]，当前可用工具：{available}"
            )
        return tool

    def names(self) -> List[str]:
        """返回全部已注册工具名。"""
        return list(self._tools)

    def describe_all(self) -> str:
        """渲染全部工具的能力清单，供 Planner / 调试时查看。"""
        lines = []
        for tool in self._tools.values():
            params = ", ".join(
                f"{key}({val})" for key, val in tool.params_desc.items()
            )
            lines.append(f"- {tool.name}: {tool.description} 参数: {params}")
        return "\n".join(lines)

    def items(self) -> List[Tool]:
        """返回全部工具对象列表，供 Web 界面渲染结构化工具清单。"""
        return list(self._tools.values())


# ---------------------------------------------------------------------------
# 三、三个内置工具
# ---------------------------------------------------------------------------

# 所有工具产生的文件统一放在项目根目录下的 output/ 文件夹中
_OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# 内置默认注册表，agent.py 直接导入使用
registry = ToolRegistry()


_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")
# 英文单词（含 don't / state-of-the-art 这类带连字符、撇号的写法）
_ENGLISH_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")
# 中日韩统一表意文字区间中的常用汉字
_CJK_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff]")
# 文件名白名单：字母、数字、下划线、连字符、中文，必须以 .txt 结尾
_FILENAME_PATTERN = re.compile(r"^[\w\u4e00-\u9fff\-]+\.txt$", re.UNICODE)


@registry.register(
    name="sum_numbers",
    description="对一组数字求和，支持传入数字列表或包含数字的字符串（自动提取）。",
    params_desc={"numbers": "list[float] 或包含数字的字符串，如 '1, 2, 3'"},
)
def sum_numbers(numbers: Union[List[float], str, int, float]) -> str:
    """数字求和工具。

    Args:
        numbers: 可以是数字列表，也可以是形如 ``"12、7.5、3"`` 的字符串，
            字符串中的数字会通过正则自动提取。

    Returns:
        形如 ``"求和结果：42.5（共 3 个数字：12 + 7.5 + 3）"`` 的观测文本。

    Raises:
        ToolParameterError: 提供的数据中提取不到任何数字时抛出。
    """
    extracted: List[float] = []
    if isinstance(numbers, (int, float)):
        extracted = [float(numbers)]
    elif isinstance(numbers, list):
        for item in numbers:
            if isinstance(item, (int, float)):
                extracted.append(float(item))
            elif isinstance(item, str):
                extracted.extend(float(n) for n in _NUMBER_PATTERN.findall(item))
    elif isinstance(numbers, str):
        extracted = [float(n) for n in _NUMBER_PATTERN.findall(numbers)]
    else:
        raise ToolParameterError(
            f"numbers 参数类型不支持：{type(numbers).__name__}"
        )

    if not extracted:
        raise ToolParameterError("未能从输入中解析出任何数字，请检查后重试。")

    total = sum(extracted)
    expression = " + ".join(_format_number(n) for n in extracted)
    return f"求和结果：{_format_number(total)}（共 {len(extracted)} 个数字：{expression}）"


@registry.register(
    name="count_words",
    description="统计文本中的英文单词数、中文字符数、总字符数与行数。",
    params_desc={"text": "需要统计的文本字符串"},
)
def count_words(text: str) -> str:
    """文本单词统计工具。

    统计口径：
        - 英文单词：按正则切分，支持 ``don't``、``state-of-the-art`` 等写法；
        - 中文字符：逐字计数（中文写作里"字数"通常以单字计）；
        - 总字符数：去除首尾空白后的字符串长度；
        - 行数：按换行符切分后的非空行数。

    Args:
        text: 待统计的文本。

    Returns:
        包含各项统计结果的多行观测文本。

    Raises:
        ToolParameterError: 输入不是字符串或为空白字符串时抛出。
    """
    if not isinstance(text, str):
        raise ToolParameterError(
            f"text 参数必须是字符串，实际类型：{type(text).__name__}"
        )
    if not text.strip():
        raise ToolParameterError("待统计的文本为空，请提供有效内容。")

    english_words = _ENGLISH_WORD_PATTERN.findall(text)
    cjk_chars = _CJK_CHAR_PATTERN.findall(text)
    line_count = len([line for line in text.splitlines() if line.strip()])

    return (
        f"文本统计完成：英文/数字单词 {len(english_words)} 个；"
        f"中文字符 {len(cjk_chars)} 个；"
        f"总字符数 {len(text.strip())}；非空行数 {line_count}。"
    )


@registry.register(
    name="save_to_txt",
    description="把文本内容保存到项目 output/ 目录下的 txt 文件，目录不存在时自动创建。",
    params_desc={
        "filename": "目标文件名，必须以 .txt 结尾，如 'result.txt'",
        "content": "要写入文件的文本内容",
    },
)
def save_to_txt(filename: str, content: str) -> str:
    """保存文本到本地 txt 文件工具。

    安全策略：
        - 只接受文件名（``Path.name`` 会剥离任何目录成分），
          防止 ``../secret.txt`` 这类路径穿越；
        - 文件名必须匹配白名单正则，且后缀必须是 ``.txt``；
        - 输出目录固定为项目根目录下的 ``output/``，不存在则自动创建。

    Args:
        filename: 目标文件名。
        content: 要写入的文本。

    Returns:
        保存成功的观测文本，包含文件绝对路径与写入字符数。

    Raises:
        ToolParameterError: 文件名或内容不合法时抛出。
        ToolExecutionError: 写文件失败时抛出（由 Tool.invoke 统一包装）。
    """
    if not isinstance(filename, str) or not filename.strip():
        raise ToolParameterError("filename 不能为空，且必须是字符串。")
    if not isinstance(content, str):
        raise ToolParameterError(
            f"content 必须是字符串，实际类型：{type(content).__name__}"
        )

    # 剥离任何目录成分，只保留纯文件名，杜绝路径穿越
    safe_name = Path(filename.strip()).name
    # 用户没写后缀时友好地补全；写了非 .txt 后缀则明确报错
    if "." not in safe_name:
        safe_name += ".txt"
    if not _FILENAME_PATTERN.match(safe_name):
        raise ToolParameterError(
            f"文件名 [{filename}] 不合法：只允许字母、数字、中文、下划线、连字符，"
            "且必须以 .txt 结尾。"
        )

    # 优化点 1：首次运行时 output/ 目录不存在则自动创建
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = _OUTPUT_DIR / safe_name
    # encoding="utf-8" 保证中文在 Windows / macOS / Linux 下读写一致
    target.write_text(content, encoding="utf-8")

    return (
        f"文件保存成功：{target}（写入 {len(content)} 个字符）。"
    )


# ---------------------------------------------------------------------------
# 四、问答扩展工具：四则运算 / 时间查询 / 预设回复
# ---------------------------------------------------------------------------

# ast 节点类型 -> 安全运算函数 的白名单映射（不用 eval，杜绝任意代码执行）
_BIN_OPS: Dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: Dict[type, Any] = {ast.USub: operator.neg, ast.UAdd: operator.pos}
# 幂运算指数上限，防止 9**9**9 这类超长计算卡死进程
_MAX_POW_EXPONENT = 1000


def _safe_eval(node: ast.AST) -> Union[int, float]:
    """递归求值 ast 节点，只允许白名单内的运算（安全计算器核心）。

    Args:
        node: ``ast.parse(expr, mode="eval")`` 得到的语法树节点。

    Returns:
        计算结果（int 或 float）。

    Raises:
        ToolParameterError: 出现白名单之外的成分或除零时抛出。
    """
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)  # 排除 True/False 被当作 1/0
    ):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW_EXPONENT:
            raise ToolParameterError(
                f"指数过大（超过 {_MAX_POW_EXPONENT}），拒绝计算以防卡死。"
            )
        try:
            return _BIN_OPS[type(node.op)](left, right)
        except ZeroDivisionError as exc:
            raise ToolParameterError("除数为 0，无法计算。") from exc
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ToolParameterError(f"算式中包含不支持的成分：{type(node).__name__}")


@registry.register(
    name="math_calc",
    description="安全计算四则运算算式，支持 + - * / % ** 与括号（ast 解析，不用 eval）。",
    params_desc={"expression": "算式字符串，如 '(3 + 5) * 2'"},
)
def math_calc(expression: str) -> str:
    """四则运算计算器工具（问答场景：xxx 等于多少 / 帮我算一下）。

    Args:
        expression: 形如 ``"(3 + 5) * 2"`` 的算式字符串。

    Returns:
        形如 ``"计算结果：16（算式：(3 + 5) * 2)"`` 的观测文本。

    Raises:
        ToolParameterError: 算式为空、语法错误、含不支持成分或除零时抛出。
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ToolParameterError("算式不能为空。")
    normalized = expression.strip()
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ToolParameterError(f"算式语法错误：{exc.msg}") from exc
    result = _safe_eval(tree)
    return f"计算结果：{_format_number(float(result))}（算式：{normalized}）"


@registry.register(
    name="get_datetime",
    description="查询当前的日期、时间、星期与 Unix 时间戳。",
    params_desc={},
)
def get_datetime() -> str:
    """时间日期查询工具（问答场景：现在几点 / 今天几号 / 星期几）。"""
    now = datetime.now()
    weekdays = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")
    return (
        f"当前时间：{now:%Y-%m-%d %H:%M:%S}（{weekdays[now.weekday()]}）；"
        f"Unix 时间戳：{int(now.timestamp())}"
    )


@registry.register(
    name="reply",
    description="直接返回一段预设文本，用于问候、自我介绍等无需计算的问答场景。",
    params_desc={"text": "要回复的文本内容"},
)
def reply(text: str) -> str:
    """预设回复工具：把 Planner 生成的答案原样作为 Observation 返回。

    Args:
        text: 要回复的文本。

    Returns:
        原样返回的文本。

    Raises:
        ToolParameterError: 文本为空白时抛出。
    """
    if not isinstance(text, str) or not text.strip():
        raise ToolParameterError("回复内容不能为空。")
    return text


def _format_number(value: float) -> str:
    """格式化数字：整数去掉 ``.0``，浮点保留有效形态，便于阅读。"""
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"
