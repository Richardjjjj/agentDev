from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from agent_demo import AgentSystem
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'knowledge_base'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv', 'json'}
KB_PERSIST_DIRECTORY = '/home/user01/software/001_test/agentDev/knowledge_base/chroma_db'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(KB_PERSIST_DIRECTORY, exist_ok=True)

# Initialize agent
agent = AgentSystem(
    api_key="NULL",
    api_base="http://192.168.10.137:31002/v1",
    model="gpt-3.5-turbo-16k",
    knowledge_base_path=None  # We'll manage the knowledge base differently
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_document_loader(file_path):
    """Get the appropriate document loader based on file extension"""
    ext = file_path.rsplit('.', 1)[1].lower()
    if ext == 'txt':
        return TextLoader(file_path)
    elif ext == 'pdf':
        return PyPDFLoader(file_path)
    elif ext == 'csv':
        return CSVLoader(file_path)
    elif ext == 'json':
        return JSONLoader(
            file_path=file_path,
            jq_schema='.',
            text_content=False
        )
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def process_file_to_kb(file_path):
    """Process a file and add it to the knowledge base"""
    try:
        # Load document
        loader = get_document_loader(file_path)
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # Set up embeddings
        os.environ["OPENAI_API_BASE"] = "http://192.168.10.137:8200/v1"
        embeddings = OpenAIEmbeddings(
            api_key="na",
            model="gpt-4"
        )
        
        # Create or update vector database
        if agent.vector_db is None:
            agent.vector_db = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings,
                persist_directory=KB_PERSIST_DIRECTORY
            )
            agent.vector_db.persist()
        else:
            agent.vector_db.add_documents(chunks)
            agent.vector_db.persist()
        
        return len(chunks)
    except Exception as e:
        raise Exception(f"Error processing file: {str(e)}")

@app.route('/api/query', methods=['POST'])
def query_agent():
    """API endpoint to query the agent"""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    if 'query' not in data:
        return jsonify({"error": "Missing 'query' parameter"}), 400
    
    user_query = data['query']
    result = agent.process_query(user_query)
    
    return jsonify({
        "response": result['response'],
        "processing_time": result['processing_time'],
        "tools_used": list(result.get('tools_used', {}).keys())
    })

@app.route('/api/knowledge/upload', methods=['POST'])
def upload_knowledge():
    """API endpoint to upload files to the knowledge base"""
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        # filename = secure_filename(file.filename)
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            chunks_added = process_file_to_kb(file_path)
            return jsonify({
                "success": True,
                "message": f"File '{filename}' processed successfully",
                "chunks_added": chunks_added
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

@app.route('/api/knowledge/status', methods=['GET'])
def knowledge_status():
    """Get status of the knowledge base"""
    if agent.vector_db is None:
        return jsonify({
            "status": "not_initialized",
            "document_count": 0
        })
    
    try:
        collection = agent.vector_db._collection
        count = collection.count()
        return jsonify({
            "status": "initialized",
            "document_count": count
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Load existing knowledge base if available
    try:
        if os.path.exists(KB_PERSIST_DIRECTORY):
            os.environ["OPENAI_API_BASE"] = "http://192.168.10.137:8200/v1"
            embeddings = OpenAIEmbeddings(
                api_key="na",
                model="gpt-4"
            )
            agent.vector_db = Chroma(
                persist_directory=KB_PERSIST_DIRECTORY,
                embedding_function=embeddings
            )
            print(f"Loaded existing knowledge base with {agent.vector_db._collection.count()} documents")
    except Exception as e:
        print(f"Error loading knowledge base: {str(e)}")
    
    app.run(host='0.0.0.0', port=5002, debug=False) 