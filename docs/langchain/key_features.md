# LangChain框架的关键特性

## 1. 模型如何返回结构化数据

### 1.1 使用 .with_structured_output()方法
> 在处理结构化输出时，有一种方法是通过传入一个模式（schema），该模式指定了所需输出属性的名称、类型和描述。这种方法通常用于确保输出的数据格式和内容与预期一致，特别是在信息提取和数据处理应用中。

我们可以使用 Pydantica 或TypedDict 或 JSON 类型的Schema。

```
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv,find_dotenv
from langchain_core.pydantic_v1 import  BaseModel,Field
from typing import Optional
from typing_extensions import Annotated,TypedDict
# 加载环境变量
load_dotenv(find_dotenv())
# 实例化模型
llm = ChatOpenAI(model="gpt-4o-mini")

# 使用 Pydantic类型
class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline of the joke")
    rating: Optional[int] = Field(default=None,description="How funny the joke is,from 1 to 10")


structed_llm = llm.with_structured_output(Joke)
print(structed_llm.invoke("Tell me a joke about cats"))

# json 输出
json_schema = {
    "title": "joke",
    "description": "Joke to tell user.",
    "type": "object",
    "properties": {
        "setup": {
            "type": "string",
            "description": "The setup of the joke",
        },
        "punchline": {
            "type": "string",
            "description": "The punchline to the joke",
        },
        "rating": {
            "type": "integer",
            "description": "How funny the joke is, from 1 to 10",
            "default": None,
        },
    },
    "required": ["setup", "punchline"],
}
json_llm = llm.with_structured_output(json_schema)
print(json_llm.invoke("Tell me a joke about dogs"))
    
```

### 1.2 多Schema 的自动选择

```
from typing import Union
from langchain_core.pydantic_v1 import BaseModel,Field
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())
llm = ChatOpenAI(model="gpt-4o-mini")

class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )


class ConversationalResponse(BaseModel):
    """通用模型"""
    """Respond in a conversational manner. Be kind and helpful."""

    response: str = Field(description="A conversational response to the user's query")


class Response(BaseModel):
    output: Union[Joke, ConversationalResponse]

struct_llm = llm.with_structured_output(Response)
print(struct_llm.invoke("Tell me a joke about pandas"))
print("="*50)
print(struct_llm.invoke("How are you today?"))
```

### 1.3 流输出
```
from typing_extensions import Annotated,TypedDict
from langchain_openai import ChatOpenAI
from typing import Annotated,Optional
from dotenv import find_dotenv,find_dotenv

load_dotenv(find_dotenv())
llm = ChatOpenAI(model="gpt-4o-mini")

class Joke(TypedDict):
    setup: Annotated[str, ..., "The setup of the joke"]
    punchline: Annotated[str, ..., "The punchline of the joke"]
    rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]

structed_llm = llm.with_structured_output(Joke)
for chunk in structed_llm.stream("Tell me a joke about pandas"):
    print(chunk)
```

### 1.4 使用 PydanticOutputParser解析器
> 不是所有的模型都支持.with_structured_output()方法。因此需要在提示中明确告诉模型需要的输出模型，并使用解析器解析输出。


```
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import  PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel,Field
from dotenv import find_dotenv,find_dotenv
# 加载环境变量
load_dotenv(find_dotenv())
# 实例化模型
llm = ChatOpenAI(temperature=0)

# 定义输出模版
class Person(BaseModel):
    """用户信息"""
    name: str = Field(...,description="The person's name")
    height_in_meters: float = Field(
        ..., description="The height of the person expressed in meters."
    )
    
class People(BaseModel):
    peeople : List[Person]

# 解析器
parser = PydanticOutputParser(pydantic_object=People)

# 提示工程
prompt = ChatPromptTemplate.from_messages([
            (
            "system",
            "Answer the user query. Wrap the output in `json` tags\n{format_instructions}",
        ),
        ("human", "{query}"),
]).partial(format_instructions=parser.get_format_instructions())

# 查询字符串
query = "Anna is 23 years old and she is 6 feet tall"
print("提示：",prompt.invoke({"query": query}).to_string())

# 调用链
chain = prompt | llm | parser
print("输出结果".center(50,"="))
chain.invoke({"query": query})
```

### 1.4 自定义解析
> 基于LCEL自定义提示和解析

```
import json
import re
from typing import List
from langchain_core.messages import AIMessage
from langchain_core.prompts import  ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel,Field
from dotenv import find_dotenv,find_dotenv

load_dotenv(find_dotenv())

llm = ChatOpenAI(temperature=0)

class Person(BaseModel):
    """Information about a person."""

    name: str = Field(..., description="The name of the person")
    height_in_meters: float = Field(
        ..., description="The height of the person expressed in meters."
    )


class People(BaseModel):
    """Identifying information about all people in a text."""

    people: List[Person]

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user query. Output your answer as JSON that  "
            "matches the given schema: ```json\n{schema}\n```. "
            "Make sure to wrap the answer in ```json and ``` tags",
        ),
        ("human", "{query}"),
    ]
).partial(schema=People.schema()) # 定义模版

# Custom parser
def extract_json(message: AIMessage) -> List[dict]:
    """Extracts JSON content from a string where JSON is embedded between ```json and ``` tags.

    Parameters:
        text (str): The text containing the JSON content.

    Returns:
        list: A list of extracted JSON strings.
    """
    text = message.content
    # Define the regular expression pattern to match JSON blocks
    pattern = r"```json(.*?)```"

    # Find all non-overlapping matches of the pattern in the string
    matches = re.findall(pattern, text, re.DOTALL)

    # Return the list of matched JSON strings, stripping any leading or trailing whitespace
    try:
        return [json.loads(match.strip()) for match in matches]
    except Exception:
        raise ValueError(f"Failed to parse: {message}")
    

# 查询
query = "Anna is 23 years old and she is 6 feet tall"

# print(prompt.format_prompt(query=query).to_string())

chain = prompt | llm | extract_json

chain.invoke({"query": query})
```
## 2. 模型调用工具

### 2.1 定义工具

#### 2.1.1 Python方法
```
# 方法名称、参数名称、参数类型、返回值类型、方法描述都是工具的必须内容。
def add(a:int,b:int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b
```
#### 2.1.2 Pydantic class 定义工具
```
from langchain_core.pydantic_v1 import BaseModel, Field


class add(BaseModel):
    """Add two integers."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


class multiply(BaseModel):
    """Multiply two integers."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")
```

**完整示例：**
```
from langchain_openai import ChatOpenAI
from dotenv import find_dotenv,load_dotenv

load_dotenv(find_dotenv())

# 实例化模型
llm = ChatOpenAI(model="gpt-4o-mini")

# 方法名称、参数名称、参数类型、返回值类型、方法描述都是工具的必须内容。
def add(a:int,b:int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b

tools = [add,multiply]

# 模型绑定工具
llm_with_tools = llm.bind_tools(tools=tools)
query = "What is the sum of 2 and 3?"
print(llm_with_tools.invoke(query))
print("乘法".center(50,"="))
query = "what is 3 * 12 ?"
# 使用模型生成工具调用需要的参数和选择的工具等
print(llm_with_tools.invoke(query))
```
## 3. 如何流式传输可运行程序

```
from langchain_openai import ChatOpenAI
from dotenv import find_dotenv,load_dotenv

load_dotenv(find_dotenv())

llm = ChatOpenAI(model="gpt-4o-mini")
# 使用 stream 方法

chunks = []

for chunk in llm.stream("what color is the sky?"):
    chunks.append(chunk)
    print(chunk.content, end="|", flush=True)

# 异步执行
async for chunk in llm.astream("what color is the sky?"):
    chunks.append(chunk)
    print(chunk.content, end="|", flush=True)
```

## 4. 模型app的调试

### 4.1 使用 LangSmith 追踪 开启调试模式

```
from langchain_openai import ChatOpenAI
from langchain.agents import  AgentExecutor,create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from dotenv import find_dotenv,load_dotenv
from langchain.globals import set_debug,set_verbose

load_dotenv(find_dotenv())

llm = ChatOpenAI(model="gpt-4o-mini")

tools = [TavilySearchResults(max_results=1)]
prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
])

# 设置调试模式  verbose 更关注重要事件，而 debug 则提供所有事件的完整日志。
set_debug(True)
# set_verbose(False)

# 构建 tools 代理
agent = create_tool_calling_agent(tools=tools, llm=llm, prompt=prompt)
# 创建代理
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
agent_executor.invoke({"input": "Who directed the 2023 film Oppenheimer and what is their age in days?"})
```


