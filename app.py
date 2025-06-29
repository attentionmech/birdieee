from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
import json
import os

app = Flask(__name__, static_folder="static")
DATA_FILE = "posts.json"

# Initialize posts.json if not exists
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        json.dump([], f)

# Load full history (list of snapshots)
def load_history():
    with open(DATA_FILE, "r") as f:
        return json.load(f)

# Get latest snapshot (returns list of flat messages)
def get_latest_state():
    history = load_history()
    return history[-1]["messages"] if history else []

# Save new snapshot of messages
def save_snapshot(messages):
    history = load_history()
    snapshot = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "messages": messages
    }
    history.append(snapshot)
    with open(DATA_FILE, "w") as f:
        json.dump(history, f, indent=2)

@app.route("/api/posts", methods=["GET"])
def get_posts():
    messages = get_latest_state()
    return jsonify(messages)

@app.route("/api/posts", methods=["POST"])
def create_post():
    data = request.json
    content = data.get("content", "").strip()
    created_by = data.get("createdBy", "anonymous").strip()
    parent_id = data.get("parentId")  # can be None or int

    if not content:
        return jsonify({"error": "Content is required"}), 400

    messages = get_latest_state()
    new_id = max((m["id"] for m in messages), default=0) + 1
    now = datetime.utcnow().isoformat() + "Z"

    new_post = {
        "id": new_id,
        "parentId": parent_id,
        "createdBy": created_by or "anonymous",
        "createdWhen": now,
        "updatedWhen": now,
        "content": content,
        "likes": []
    }

    messages.append(new_post)
    save_snapshot(messages)
    return jsonify(new_post), 201

@app.route("/api/history", methods=["GET"])
def get_history():
    return jsonify(load_history())

@app.route("/")
def serve_frontend():
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(debug=True)

