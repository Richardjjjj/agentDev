#!/usr/bin/env python3
import requests
import json
import sys
import os

API_BASE = "http://localhost:5002/api"

def query_agent(query_text):
    """Send a query to the agent API"""
    url = f"{API_BASE}/query"
    payload = {"query": query_text}
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying agent: {e}")
        return None

def upload_file(file_path):
    """Upload a file to the knowledge base"""
    url = f"{API_BASE}/knowledge/upload"
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'rb') as file:
            files = {'file': (os.path.basename(file_path), file)}
            response = requests.post(url, files=files)
            response.raise_for_status()
            return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error uploading file: {e}")
        return None

def get_kb_status():
    """Get knowledge base status"""
    url = f"{API_BASE}/knowledge/status"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting knowledge base status: {e}")
        return None

def print_help():
    """Print help information"""
    print("Usage:")
    print("  python test_api_client.py query \"Your question here\"")
    print("  python test_api_client.py upload /path/to/file.txt")
    print("  python test_api_client.py status")
    print("  python test_api_client.py help")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "query" and len(sys.argv) >= 3:
        query_text = sys.argv[2]
        print(f"Sending query: {query_text}")
        result = query_agent(query_text)
        if result:
            print("\nResponse:")
            print(result["response"])
            print(f"\nProcessing time: {result['processing_time']:.2f} seconds")
            if result.get("tools_used"):
                print(f"Tools used: {', '.join(result['tools_used'])}")
    
    elif command == "upload" and len(sys.argv) >= 3:
        file_path = sys.argv[2]
        print(f"Uploading file: {file_path}")
        result = upload_file(file_path)
        if result:
            print(json.dumps(result, indent=2))
    
    elif command == "status":
        print("Getting knowledge base status...")
        result = get_kb_status()
        if result:
            print(json.dumps(result, indent=2))
    
    elif command == "help":
        print_help()
    
    else:
        print("Invalid command or missing arguments.")
        print_help()
        sys.exit(1) 