from flask import Flask, request, jsonify, send_from_directory
import json
import os

app = Flask(__name__, static_folder="static")
DATA_FILE = "posts.json"

# Initialize posts.json if not exists
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        json.dump([], f)

def load_posts():
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_posts(posts):
    with open(DATA_FILE, "w") as f:
        json.dump(posts, f, indent=2)

@app.route("/api/posts", methods=["GET"])
def get_posts():
    posts = load_posts()
    return jsonify(posts[::-1])  # reverse for latest first

@app.route("/api/posts", methods=["POST"])
def add_post():
    data = request.json
    content = data.get("content", "").strip()
    if not content:
        return jsonify({"error": "Content is required"}), 400

    posts = load_posts()
    post_id = max([p["id"] for p in posts], default=0) + 1
    post = {"id": post_id, "content": content, "replies": []}
    posts.append(post)
    save_posts(posts)
    return jsonify(post), 201

@app.route("/api/reply", methods=["POST"])
def add_reply():
    data = request.json
    post_id = data.get("post_id")
    content = data.get("content", "").strip()

    if not content:
        return jsonify({"error": "Reply content required"}), 400

    posts = load_posts()
    for post in posts:
        if post["id"] == post_id:
            post["replies"].append(content)
            save_posts(posts)
            return jsonify(post), 200

    return jsonify({"error": "Post not found"}), 404

# Serve frontend
@app.route("/")
def serve_frontend():
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(debug=True)

