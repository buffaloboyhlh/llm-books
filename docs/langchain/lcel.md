# LangChain Expression Language (LCEL)

LangChain Expression Language (LCEL) 是一种声明式语言，用于轻松组合 LangChain 组件。LCEL 设计的目标是支持从原型到生产环境的过渡，无需更改代码。其主要特点包括以下几个方面：

	1.	流式支持：LCEL 提供了第一流的流式支持，能够最短时间内输出首个标记。这意味着在某些链中，可以直接从语言模型（LLM）流式输出到流式输出解析器，随着 LLM 提供原始标记的速率返回解析后的增量输出。
	2.	异步支持：LCEL 的每个链都可以同步和异步调用，适用于从原型到生产的不同阶段，支持高并发请求处理。
	3.	优化的并行执行：LCEL 自动处理可并行执行的步骤，例如从多个检索器获取文档，以最小化延迟。
	4.	重试和回退机制：LCEL 允许为任何链配置重试和回退，增加了大规模使用的可靠性。
	5.	访问中间结果：对于复杂的链，可以在最终输出产生之前访问中间步骤的结果，这对调试和提高用户体验非常有帮助。
	6.	输入输出模式：每个 LCEL 链都会自动推断输入和输出的 Pydantic 和 JSONSchema 模式，用于验证输入和输出，是 LangServe 的重要组成部分。

LCEL 旨在提供一致性和定制性，使得开发者能够更容易地构建复杂的流程，并且随着模型的多样性增加，定制化变得更加重要。


## 1. 调用链
```
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import  StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv,find_dotenv

# 加载环境变量
load_dotenv(find_dotenv())

# 实例化模型
llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
# 调用链
chain = prompt | llm | StrOutputParser()

print(chain.invoke({"topic":"bears"}))

print("添加评估模块".center(50,"="))

analysis_prompt = ChatPromptTemplate.from_template("is this a funnny joke? {joke}")
composed_chain = {"joke":chain} | analysis_prompt | llm | StrOutputParser()
print(composed_chain.invoke({"topic":"bears"}))
```

## 2. 如何并行调用可运行对象

```
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())

llm = ChatOpenAI()

joke_prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}") 
poem_prompt = ChatPromptTemplate.from_template("write a 2-line poem about {topic}")

joke_chain = joke_prompt | llm 
poem_chain = poem_prompt | llm

map_chain = RunnableParallel(joke=joke_chain,poem=poem_chain)
map_chain.invoke({"topic":"bear"})
```

## 3. 默认调用参数

```
from langchain_openai import ChatOpenAI 
from langchain_core.output_parsers import  StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())

llm = ChatOpenAI(temperature=0)

# 提示
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write out the following equation using algebraic symbols then solve it. Use the format\n\nEQUATION:...\nSOLUTION:...\n\n",
        ),
        ("human", "{equation_statement}"),
    ]
)

runnable = (
    {"equation_statement": RunnablePassthrough()} | prompt | llm | StrOutputParser()
)

print(runnable.invoke("x raised to the third plus seven equals 12"))

print("绑定默认参数".center(50,"="))
# 绑定默认参数
runnable = (
    {"equation_statement": RunnablePassthrough()}
    | prompt
    | llm.bind(stop="SOLUTION")
    | StrOutputParser()
)

print(runnable.invoke("x raised to the third plus seven equals 12"))

# 附带一个工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

print("绑定一个工具".center(50,"="))

model = ChatOpenAI(model="gpt-3.5-turbo-1106").bind(tools=tools)
model.invoke("What's the weather in SF, NYC and LA?")
```

## 4. 运行自定义函数 （How to run custom functions）
### 4.1 使用RunnableLambda 构造器

```
from  operator import  itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())

# 自定义函数
def length_function(text):
    return len(text)


def _multiple_length_function(text1, text2):
    return len(text1) * len(text2)


def multiple_length_function(_dict):
    return _multiple_length_function(_dict["text1"], _dict["text2"])

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("what is {a} + {b}")

chain = {"a":itemgetter("foo") | RunnableLambda(length_function),"b":{"text1": itemgetter("foo"), "text2": itemgetter("bar")}
        | RunnableLambda(multiple_length_function),} | prompt | model

print(chain.invoke({"foo": "bar", "bar": "gah"}))
```

### 4.2 使用装饰器

```
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain

prompt1 = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
prompt2 = ChatPromptTemplate.from_template("What is the subject of this joke: {joke}")


@chain
def custom_chain(text):
    prompt_val1 = prompt1.invoke({"topic": text})
    output1 = ChatOpenAI().invoke(prompt_val1)
    parsed_output1 = StrOutputParser().invoke(output1)
    chain2 = prompt2 | ChatOpenAI() | StrOutputParser()
    return chain2.invoke({"joke": parsed_output1})


custom_chain.invoke("bears")
```

### 4.3 链中的自动类型转换

```
prompt = ChatPromptTemplate.from_template("tell me a story about {topic}")

model = ChatOpenAI()

chain_with_coerced_function = prompt | model | (lambda x: x.content[:5])

chain_with_coerced_function.invoke({"topic": "bears"})
```

## 5. 参数在链中传递 (How to pass through arguments from one step to the next)

```
from langchain_core.runnables import RunnableParallel,RunnablePassthrough

runnable = RunnableParallel(
    passed=RunnablePassthrough(),
    modified=lambda x: x["num"]+1
)

runnable.invoke({"num":1})
```








