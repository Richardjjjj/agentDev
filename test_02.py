import requests
import json

class ToolKit:
    def __init__(self):
        self.tools = {
            "weather": self.get_weather,
            "calculator": self.calculate,
            "search": self.search_web
        }
    
    def get_weather(self, location):
        # 模拟天气API调用
        return f"{location}天气: 晴朗，25°C"
    
    def calculate(self, expression):
        try:
            return str(eval(expression))
        except:
            return "计算错误"
    
    def search_web(self, query):
        # 模拟搜索API
        return f"搜索结果: 关于{query}的信息..."
    
    def use_tool(self, tool_name, params):
        if tool_name in self.tools:
            return self.tools[tool_name](params)
        return "工具不存在"

# 使用示例
toolkit = ToolKit()
weather_result = toolkit.use_tool("weather", "北京")
calc_result = toolkit.use_tool("calculator", "23*7")
print(weather_result)
print(calc_result)