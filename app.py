from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime, timezone
import json
import os
import threading
import time
import logging
from dotenv import load_dotenv
from llm_integration import LLMIntegration

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")

DATA_FILE = "vault/posts.json"
NOTIF_FILE = "vault/notifications.json"
SETTINGS_FILE = "vault/settings.json"

# Initialize LLM integration
try:
    llm = LLMIntegration()
    logger.info(f"LLM Integration initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    llm = None

# Ensure directories exist
os.makedirs("vault", exist_ok=True)
os.makedirs("static", exist_ok=True)

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        json.dump([], f)

if not os.path.exists(NOTIF_FILE):
    with open(NOTIF_FILE, "w") as f:
        json.dump([], f)

if not os.path.exists(SETTINGS_FILE):
    with open(SETTINGS_FILE, "w") as f:
        json.dump({"notifications_enabled": True}, f)

def load_settings():
    try:
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading settings: {e}")
        return {"notifications_enabled": True}

def save_settings(settings):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving settings: {e}")

def load_notifications():
    try:
        with open(NOTIF_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading notifications: {e}")
        return []

def save_notifications(notifs):
    try:
        with open(NOTIF_FILE, "w") as f:
            json.dump(notifs, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving notifications: {e}")

def add_notification(message, post_id=None):
    notifs = load_notifications()
    notif = {
        "id": max((n["id"] for n in notifs), default=0) + 1,
        "message": message,
        "post_id": post_id,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "seen": False
    }
    notifs.append(notif)
    save_notifications(notifs)
    logger.info(f"Added notification: {message}")

# Load full history (list of snapshots)
def load_history():
    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading history: {e}")
        return []

# Get latest snapshot (returns list of flat messages)
def get_latest_state():
    history = load_history()
    return history[-1]["messages"] if history else []

# Save new snapshot of messages
def save_snapshot(messages):
    try:
        history = load_history()
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "messages": messages
        }
        history.append(snapshot)
        with open(DATA_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving snapshot: {e}")

def get_conversation_context(parent_id, messages):
    """Get conversation context for a thread"""
    if not parent_id:
        return []
    
    # Find the root post and all replies in the thread
    thread_messages = []
    
    # Find root post
    root_post = None
    for msg in messages:
        if msg["id"] == parent_id:
            root_post = msg
            break
        elif msg.get("parentId") == parent_id:
            # This is a reply to the parent
            continue
    
    # If parent_id is actually a reply, find the real root
    if root_post and root_post.get("parentId"):
        return get_conversation_context(root_post["parentId"], messages)
    
    # Collect all messages in this thread
    def collect_thread(post_id):
        thread = []
        for msg in messages:
            if msg["id"] == post_id or msg.get("parentId") == post_id:
                thread.append(msg)
        return thread
    
    if root_post:
        thread_messages = collect_thread(root_post["id"])
    
    return sorted(thread_messages, key=lambda x: x["createdWhen"])

def should_ai_reply(post, parent_id=None):
    """Determine if AI should reply based on new logic"""
    if not llm:
        return False, None
    
    # If this is a reply to an AI post, only that AI should respond
    if parent_id:
        messages = get_latest_state()
        parent_post = next((msg for msg in messages if msg["id"] == parent_id), None)
        if parent_post and parent_post.get("isAI"):
            # Only the same AI personality should respond
            target_personality = parent_post.get("createdBy")
            personality = next((p for p in llm.personalities if p.name == target_personality), None)
            if personality:
                return True, personality
            return False, None
    
    # For new posts or replies to human posts, any AI can respond (but not to other AIs)
    if not post.get("isAI", False):  # Only reply to human posts
        if llm.should_reply_randomly():  # 30% chance is fine for main posts
            return True, llm.get_random_personality()
    
    return False, None

def schedule_ai_reply(post_id, post_content, user_name, parent_id=None):
    """Schedule an AI reply with improved logic"""
    messages = get_latest_state()
    current_post = next((msg for msg in messages if msg["id"] == post_id), None)
    if not current_post:
        return
    
    should_reply, personality = should_ai_reply(current_post, parent_id)
    if not should_reply or not personality:
        return
    
    # Shorter delay for better user experience
    delay = llm.get_shorter_delay()
    logger.info(f"Scheduling AI reply for post {post_id} by {personality.name} in {delay} seconds")
    
    def create_delayed_reply():
        time.sleep(delay)
        try:
            messages = get_latest_state()
            
            # Get conversation context
            context = get_conversation_context(parent_id or post_id, messages)
            
            # Generate reply
            reply_content = llm.generate_reply(
                personality=personality,
                post_content=post_content,
                conversation_context=context,
                user_name=user_name
            )
            
            # Create the AI reply
            new_id = max((m["id"] for m in messages), default=0) + 1
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            
            ai_post = {
                "id": new_id,
                "parentId": parent_id or post_id,
                "createdBy": personality.name,
                "createdWhen": now,
                "updatedWhen": now,
                "content": reply_content,
                "likes": [],
                "isAI": True,
                "aiPersonality": personality.style
            }
            
            messages.append(ai_post)
            save_snapshot(messages)
            
            # Add notification
            add_notification(f"{personality.name} replied to your post", post_id)
            
            logger.info(f"AI reply created by {personality.name}: {reply_content}")
            
        except Exception as e:
            logger.error(f"Error creating delayed AI reply: {e}")
    
    # Start the delayed reply in a separate thread
    thread = threading.Thread(target=create_delayed_reply, daemon=True)
    thread.start()

@app.route("/api/settings", methods=["GET"])
def get_settings():
    return jsonify(load_settings())

@app.route("/api/settings", methods=["POST"])
def update_settings():
    try:
        settings = request.json
        save_settings(settings)
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/notifications", methods=["GET"])
def get_notifications():
    return jsonify(load_notifications())

@app.route("/api/notifications/mark-seen", methods=["POST"])
def mark_notifications_seen():
    """Mark notifications as seen"""
    try:
        notifs = load_notifications()
        for notif in notifs:
            notif["seen"] = True
        save_notifications(notifs)
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error marking notifications as seen: {e}")
        return jsonify({"error": str(e)}), 500

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
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    new_post = {
        "id": new_id,
        "parentId": parent_id,
        "createdBy": created_by or "anonymous",
        "createdWhen": now,
        "updatedWhen": now,
        "content": content,
        "likes": [],
        "isAI": False
    }

    messages.append(new_post)
    save_snapshot(messages)
    
    # Schedule potential AI reply
    if llm:
        schedule_ai_reply(new_id, content, created_by, parent_id)
    
    return jsonify(new_post), 201

@app.route("/api/posts/<int:post_id>/like", methods=["POST"])
def like_post(post_id):
    """Like/unlike a post"""
    data = request.json
    user = data.get("user", "anonymous")
    
    messages = get_latest_state()
    
    for message in messages:
        if message["id"] == post_id:
            likes = message.get("likes", [])
            if user in likes:
                likes.remove(user)
                action = "unliked"
            else:
                likes.append(user)
                action = "liked"
            message["likes"] = likes
            save_snapshot(messages)
            return jsonify({"success": True, "action": action, "likes": len(likes)})
    
    return jsonify({"error": "Post not found"}), 404

@app.route("/api/history", methods=["GET"])
def get_history():
    return jsonify(load_history())

@app.route("/api/ai-status", methods=["GET"])
def get_ai_status():
    """Get AI integration status"""
    return jsonify({
        "enabled": llm is not None,
        "personalities": [p.name for p in llm.personalities] if llm else [],
        "provider": os.getenv("LLM_PROVIDER", "not_configured")
    })

@app.route("/")
def serve_frontend():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    
    logger.info(f"Starting Flask app on port {port}")
    if llm:
        logger.info("AI personalities ready to engage!")
    else:
        logger.warning("AI integration disabled - check your environment variables")
    
    app.run(debug=debug, host="0.0.0.0", port=port)