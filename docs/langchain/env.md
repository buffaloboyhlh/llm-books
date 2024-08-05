# LangChain开发环境搭建

## 安装langchain的主要的包

```
conda install -y langchain -c conda-forge
```

## langchain包之间的关系

![](https://python.langchain.com/v0.2/assets/images/ecosystem_packages-32943b32657e7a187770c9b585f22a64.png)

## langchain的社区包 (包含第三方的集成)

```
pip install langchain-community
```

## LangGraph

LangGraph 是一种用于构建复杂语言代理的开源框架。它的设计旨在简化创建多代理系统，特别适合处理复杂任务。以下是 LangGraph 的一些关键功能和特点：

	1.	多代理设计：LangGraph 允许用户创建多个独立的代理，这些代理可以执行特定的任务，并且可以协作或分工。每个代理可以有自己的提示、工具和模型，这种设计提高了系统的灵活性和扩展性 ￼ ￼。
	2.	循环与分支：框架支持实现循环和条件分支，使得开发者能够在图中定义复杂的逻辑流。这对于处理需要多次迭代或条件执行的任务非常有用 ￼。
	3.	状态管理：LangGraph 提供自动状态保存功能，允许在任意时刻暂停和恢复图的执行。这对于错误恢复和长时间运行的任务非常重要 ￼。
	4.	人机交互：框架支持在人类介入的情况下暂停代理的执行，进行人工审查或编辑，从而提高系统的可靠性 ￼。
	5.	集成与扩展：虽然 LangGraph 可以独立使用，但它也能与 LangChain 等其他工具集成，提供更丰富的功能，例如流式输出和持久化 ￼ ￼。

```
pip install langgraph
```

## LangServe
LangServe 是一种用于快速部署和管理基于 LangChain 的应用的工具，特别适用于大型语言模型（LLM）应用的开发。

```
pip install "langserve[all]"
```

## LangChain CLI

langchain 项目的命令行工具，可以快速搭建langchain项目
```
pip install langchain-cli
```



