import os
import time
import json
import logging
from typing import List, Dict, Any, Optional
import requests
from datetime import datetime

# 更新导入路径
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

class AgentSystem:
    def __init__(self, api_key: str, api_base: str, model: str, knowledge_base_path: Optional[str] = None):
        # 初始化配置
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_BASE"] = api_base
        
        # 初始化日志系统
        self.setup_logging()
        
        # 初始化LLM - 移除了api_base参数
        self.llm = ChatOpenAI(
            temperature=0.7,
            api_key=api_key,
            model=model
        )
        
        # 初始化对话记忆
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # 初始化工具集
        self.tools = self.setup_tools()
        
        # 初始化知识库(如果提供)
        self.vector_db = None
        if knowledge_base_path:
            self.setup_knowledge_base(knowledge_base_path)
        
        self.logger.info("Agent系统初始化完成")
    
    def setup_logging(self):
        """设置日志系统"""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("AgentSystem")
    
    def setup_tools(self) -> Dict[str, callable]:
        """设置工具集"""
        return {
            "search": self.search_web,
            "weather": self.get_weather,
            "calculator": self.calculate,
            "time": self.get_current_time
        }
    
    def setup_knowledge_base(self, knowledge_base_path: str):
        """设置知识库"""
        try:
            # 加载文档
            loader = TextLoader(knowledge_base_path)
            documents = loader.load()
            
            # 分割文档
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            
            os.environ["OPENAI_API_BASE"] = "http://192.168.10.137:8200/v1"
            # 创建向量数据库
            embeddings = OpenAIEmbeddings(
                api_key="na",
                model="gpt-4"
            )
            self.vector_db = Chroma.from_documents(chunks, embeddings)
            
            self.logger.info(f"已加载知识库: {knowledge_base_path}, 共{len(chunks)}个文档块")
        except Exception as e:
            self.logger.error(f"知识库加载失败: {str(e)}")
    
    # 百度搜索API实现
    def search_web(self, query: str) -> str:
        """使用百度搜索API查询网络信息"""
        self.logger.info(f"使用百度API执行搜索: {query}")
        
        try:
            # 百度搜索API需要申请，这里使用的是公开API
            url = "https://www.baidu.com/sugrec"
            params = {
                "prod": "pc",
                "wd": query,
                "cb": "jQuery"
            }
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=5)
            
            # 处理百度搜索建议API的响应
            if response.status_code == 200:
                # 提取JSON数据
                content = response.text
                json_str = content.strip("jQuery(").rstrip(");")
                result_json = json.loads(json_str)
                
                # 提取搜索建议
                suggestions = []
                if "g" in result_json and isinstance(result_json["g"], list):
                    for item in result_json["g"]:
                        if "q" in item:
                            suggestions.append(item["q"])
                
                if suggestions:
                    return f"百度搜索结果: 关于'{query}'的信息:\n" + "\n".join([f"- {s}" for s in suggestions])
                else:
                    return f"百度搜索: 未找到关于'{query}'的建议信息"
            else:
                return f"百度搜索请求失败: 状态码 {response.status_code}"
                
        except Exception as e:
            self.logger.error(f"百度搜索API调用出错: {str(e)}")
            return f"百度搜索API调用出错: {str(e)}"
    
    # 墨迹天气API实现
    def get_weather(self, location: str) -> str:
        """使用墨迹天气API获取天气信息"""
        self.logger.info(f"使用墨迹天气API查询天气: {location}")
        
        try:
            # 墨迹天气开放API (注：实际使用需要申请API密钥)
            # 这里使用的是一个公开的天气API替代
            url = "https://restapi.amap.com/v3/weather/weatherInfo?parameters"
            
            # 高德地图天气API参数
            # 注意：实际使用时需要替换为您自己的API密钥
            params = {
                "key": "8074ed8ae34efb78133405c2dfd284ba",  # 替换为您的高德地图API密钥
                "city": location,
                "extensions": "base"
            }
            
            # 如果没有高德地图API密钥，可以模拟响应
            mock_response = False
            
            if mock_response:
                # 模拟天气数据结构但使用真实城市名
                weather_data = {
                    "城市": location,
                    "天气": "晴",
                    "温度": "25°C",
                    "湿度": "45%",
                    "风向": "东南风",
                    "风力": "3级",
                    "更新时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # 格式化返回结果
                weather_str = f"{location}天气:\n"
                for key, value in weather_data.items():
                    if key != "城市":  # 城市名已经在开头显示
                        weather_str += f"- {key}: {value}\n"
                
                return weather_str.strip()
            else:
                # 实际API调用
                response = requests.get(url, params=params, timeout=5)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get("status") == "1" and "lives" in result and result["lives"]:
                        weather = result["lives"][0]
                        
                        weather_str = f"{weather.get('city', location)}天气:\n"
                        weather_str += f"- 天气: {weather.get('weather', '未知')}\n"
                        weather_str += f"- 温度: {weather.get('temperature', '未知')}°C\n"
                        weather_str += f"- 湿度: {weather.get('humidity', '未知')}%\n"
                        weather_str += f"- 风向: {weather.get('winddirection', '未知')}\n"
                        weather_str += f"- 风力: {weather.get('windpower', '未知')}\n"
                        weather_str += f"- 更新时间: {weather.get('reporttime', '未知')}"
                        
                        return weather_str
                    else:
                        return f"获取{location}天气信息失败: {result.get('info', '未知错误')}"
                else:
                    return f"天气API请求失败: 状态码 {response.status_code}"
        
        except Exception as e:
            self.logger.error(f"天气API调用出错: {str(e)}")
            return f"天气API调用出错: {str(e)}"
    
    def calculate(self, expression: str) -> str:
        """计算表达式"""
        self.logger.info(f"计算: {expression}")
        try:
            result = eval(expression)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"
    
    def get_current_time(self, timezone: str = "local") -> str:
        """获取当前时间"""
        self.logger.info(f"查询时间: {timezone}")
        return f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    def query_knowledge_base(self, query: str, k: int = 3) -> List[str]:
        """查询知识库"""
        if not self.vector_db:
            return []
        
        self.logger.info(f"查询知识库: {query}")
        docs = self.vector_db.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    
    def create_prompt(self, user_input: str, retrieved_info: List[str] = None) -> str:
        """创建提示"""
        prompt = f"用户问题: {user_input}\n\n"
        
        # 添加知识库检索结果
        if retrieved_info and len(retrieved_info) > 0:
            prompt += "相关信息:\n"
            for i, info in enumerate(retrieved_info, 1):
                prompt += f"{i}. {info}\n"
            prompt += "\n"
        
        # 添加工具说明
        prompt += "可用工具:\n"
        for tool_name, tool_func in self.tools.items():
            prompt += f"- {tool_name}: {tool_func.__doc__}\n"
        
        prompt += "\n请根据用户问题，使用提供的信息和工具回答。如需使用工具，请使用格式: [工具名称]参数"
        return prompt
    
    def parse_tool_usage(self, response: str) -> Dict[str, Any]:
        """解析回答中的工具使用"""
        tool_results = {}
        
        # 简单的工具调用解析，实际可能需要更复杂的解析
        import re
        tool_calls = re.findall(r'\[([a-zA-Z_]+)\](.*?)(?=\[|$)', response)
        
        for tool_name, params in tool_calls:
            if tool_name in self.tools:
                params = params.strip()
                self.logger.info(f"使用工具: {tool_name}, 参数: {params}")
                result = self.tools[tool_name](params)
                tool_results[tool_name] = result
                # 替换回答中的工具调用为结果
                response = response.replace(f"[{tool_name}]{params}", result)
        
        return {
            "processed_response": response,
            "tool_results": tool_results
        }
    
    def process_query(self, user_input: str) -> Dict[str, Any]:
        """处理用户查询"""
        start_time = time.time()
        self.logger.info(f"收到用户查询: {user_input}")
        
        # 查询知识库
        retrieved_info = self.query_knowledge_base(user_input)
        
        # 创建提示
        prompt = self.create_prompt(user_input, retrieved_info)
        
        # 查询LLM
        try:
            llm_response = self.llm.invoke(prompt).content
            
            # 解析工具使用
            processed_result = self.parse_tool_usage(llm_response)
            final_response = processed_result["processed_response"]
            tools_used = processed_result["tool_results"]
            
            # 更新记忆
            self.memory.save_context(
                {"input": user_input},
                {"output": final_response}
            )
            
            processing_time = time.time() - start_time
            self.logger.info(f"查询处理完成，耗时: {processing_time:.2f}秒")
            
            return {
                "response": final_response,
                "processing_time": processing_time,
                "tools_used": tools_used
            }
        except Exception as e:
            self.logger.error(f"处理查询时出错: {str(e)}")
            return {
                "response": f"处理您的查询时出现错误: {str(e)}",
                "processing_time": time.time() - start_time,
                "error": str(e)
            }

# 使用示例
if __name__ == "__main__":
    # 创建简单的知识库文件
    
    agent = AgentSystem(
        api_key="NULL",
        api_base="http://192.168.10.137:31002/v1",
        model="gpt-3.5-turbo-16k",
        knowledge_base_path="agent_knowledge.txt"
    )
    
    # 模拟对话
    queries = [
        # "什么是AI Agent?",
        "介绍一下北京象新力科技有限公司",
        "武汉现在的天气怎么样?",
        "计算25乘以16",
        "现在几点了?"
    ]
    
    for query in queries:
        print(f"\n用户: {query}")
        result = agent.process_query(query)
        print(f"Agent: {result['response']}")
        print(f"处理时间: {result['processing_time']:.2f}秒")
        if "tools_used" in result and result["tools_used"]:
            print(f"使用的工具: {', '.join(result['tools_used'].keys())}")