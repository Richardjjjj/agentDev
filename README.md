# Agent API 文档
该项目提供了一套用于与 AI 代理系统交互并管理其知识库的 RESTful API。
## 配置
1. 安装依赖项：
```bash
pip install flask langchain langchain_community langchain_openai werkzeug
```
2. 启动 API 服务器：
```bash
python agent_api.py
```
服务器将在 `http://0.0.0.0:5000` 启动。
## API 端点
### 查询代理
向代理发送查询并获取响应。
**端点：** `POST /api/query`
**请求体：**
```json
{
  "query": "你的问题在这里"
}
```
**响应：**
```json
{
  "response": "代理的响应",
  "processing_time": 1.23,
  "tools_used": ["天气", "计算器"]
}
```
### 上传知识
向知识库上传文件。
**端点：** `POST /api/knowledge/upload`
**请求：**
- 带有一个名为 "file" 的文件字段的表单数据
- 支持的文件类型：txt、pdf、csv、json
**响应：**
```json
{
  "success": true,
  "message": "文件 'example.txt' 处理成功",
  "chunks_added": 10
}
```
### 知识库状态
获取知识库的当前状态。
**端点：** `GET /api/knowledge/status`
**响应：**
```json
{
  "status": "initialized",
  "document_count": 42
}
```
## 知识库容量
用于知识库的 Chroma 向量数据库可以处理大量文档。实际限制取决于：
1. 可用内存：每个文档块都需要内存用于存储和嵌入
2. 磁盘空间：在使用持久化时
3. 查询性能：大型数据库可能会有较慢的查询时间
对于大多数用例，知识库可以轻松处理数千到数万个文档块而不会出现问题。
## 使用示例
### 查询代理
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "北京天气怎么样？"}'
```
### 上传文件
```bash
curl -X POST http://localhost:5000/api/knowledge/upload \
  -F "file=@/path/to/your/document.pdf"
```
### 检查知识库状态
```bash
curl http://localhost:5000/api/knowledge/status
```
