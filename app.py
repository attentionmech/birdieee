from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
import json
import os
import random

app = Flask(__name__, static_folder="static")

DATA_FILE = "vault/posts.json"
BOTS_FILE = "static/bots.json"
NOTIF_FILE = "vault/notifications.json"


os.makedirs("vault", exist_ok=True)

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        json.dump([], f)

if not os.path.exists(NOTIF_FILE):
    with open(NOTIF_FILE, "w") as f:
        json.dump([], f)


def load_bots():
    with open(BOTS_FILE, "r") as f:
        return json.load(f)

def load_notifications():
    with open(NOTIF_FILE, "r") as f:
        return json.load(f)

def save_notifications(notifs):
    with open(NOTIF_FILE, "w") as f:
        json.dump(notifs, f, indent=2)

def add_notification(message):
    notifs = load_notifications()
    notif = {
        "id": max((n["id"] for n in notifs), default=0) + 1,
        "message": message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seen": False
    }
    notifs.append(notif)
    save_notifications(notifs)


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

@app.route("/api/notifications", methods=["GET"])
def get_notifications():
    return jsonify(load_notifications())


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

    # Bot reply (only for top-level posts)
    if parent_id is None:
        bots = load_bots()
        bot = random.choice(bots)
        reply = random.choice(bot["replies"])
        bot_post = {
            "id": new_id + 1,
            "parentId": new_id,
            "createdBy": bot["name"],
            "createdWhen": now,
            "updatedWhen": now,
            "content": reply,
            "likes": []
        }
        messages.append(bot_post)
        add_notification(f"{bot['name']} replied to your post")

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

