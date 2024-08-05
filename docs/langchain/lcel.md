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



