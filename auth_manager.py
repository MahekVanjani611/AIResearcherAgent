#!/usr/bin/env python3
"""
Authentication & User Management System

Handles:
1. User registration with password hashing
2. User login with password verification
3. Session token management
4. User profile management
"""

import json
import hashlib
import hmac
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import uuid

try:
    import bcrypt
except ImportError:
    bcrypt = None


USERS_FILE = Path("users.json")
SESSIONS_FILE = Path("sessions.json")
MAX_SESSION_VALIDITY = 7 * 24 * 60 * 60  # 7 days in seconds
PBKDF2_ITERATIONS = 390000


class AuthManager:
    """Manages user authentication and sessions"""
    
    def __init__(self):
        self.users_file = USERS_FILE
        self.sessions_file = SESSIONS_FILE
        self._load_users()
        self._load_sessions()
    
    def _load_users(self):
        """Load users from file"""
        if self.users_file.exists():
            with open(self.users_file, 'r') as f:
                self.users = json.load(f)
        else:
            self.users = {}
    
    def _save_users(self):
        """Save users to file"""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    def _load_sessions(self):
        """Load sessions from file"""
        if self.sessions_file.exists():
            with open(self.sessions_file, 'r') as f:
                self.sessions = json.load(f)
        else:
            self.sessions = {}
    
    def _save_sessions(self):
        """Save sessions to file"""
        with open(self.sessions_file, 'w') as f:
            json.dump(self.sessions, f, indent=2)
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        if bcrypt is not None:
            return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        salt = uuid.uuid4().hex
        derived = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            PBKDF2_ITERATIONS,
        )
        return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${derived.hex()}"
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            if hashed.startswith("pbkdf2_sha256$"):
                _, iterations, salt, expected_hash = hashed.split("$", 3)
                derived = hashlib.pbkdf2_hmac(
                    "sha256",
                    password.encode("utf-8"),
                    salt.encode("utf-8"),
                    int(iterations),
                ).hex()
                return hmac.compare_digest(derived, expected_hash)

            if bcrypt is None:
                return False

            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception:
            return False
    
    def register_user(self, username: str, password: str, email: str = "") -> Tuple[bool, str]:
        """
        Register a new user
        
        Args:
            username: Username
            password: Password (min 6 chars)
            email: Optional email
            
        Returns:
            (success: bool, message: str)
        """
        # Validation
        if not username or len(username) < 3:
            return False, "Username must be at least 3 characters"
        
        if not password or len(password) < 6:
            return False, "Password must be at least 6 characters"
        
        if username in self.users:
            return False, "Username already exists"
        
        # Create user
        self.users[username] = {
            "password": self.hash_password(password),
            "email": email,
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "chats": []  # List of chat IDs
        }
        self._save_users()
        
        return True, f"User '{username}' registered successfully"
    
    def login_user(self, username: str, password: str) -> Tuple[bool, str, Optional[str]]:
        """
        Login user
        
        Args:
            username: Username
            password: Password
            
        Returns:
            (success: bool, message: str, session_token: str or None)
        """
        if username not in self.users:
            return False, "User not found", None
        
        user = self.users[username]
        if not self.verify_password(password, user["password"]):
            return False, "Invalid password", None
        
        # Create session token
        session_token = str(uuid.uuid4())
        self.sessions[session_token] = {
            "username": username,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "valid": True
        }
        self._save_sessions()
        
        # Update last login
        user["last_login"] = datetime.now().isoformat()
        self._save_users()
        
        return True, f"Welcome {username}!", session_token
    
    def verify_session(self, session_token: str) -> Tuple[bool, Optional[str]]:
        """
        Verify if session is valid
        
        Returns:
            (is_valid: bool, username: str or None)
        """
        if session_token not in self.sessions:
            return False, None
        
        session = self.sessions[session_token]
        
        # Check if session is still valid
        created_at = datetime.fromisoformat(session["created_at"])
        if (datetime.now() - created_at).total_seconds() > MAX_SESSION_VALIDITY:
            return False, None
        
        if not session.get("valid", False):
            return False, None
        
        return True, session["username"]
    
    def logout_user(self, session_token: str) -> bool:
        """Logout user by invalidating session"""
        if session_token in self.sessions:
            self.sessions[session_token]["valid"] = False
            self._save_sessions()
            return True
        return False
    
    def get_user_chats(self, username: str) -> list:
        """Get all chats for a user"""
        if username in self.users:
            return self.users[username].get("chats", [])
        return []
    
    def add_chat_to_user(self, username: str, chat_id: str):
        """Add chat ID to user's chat list"""
        if username in self.users:
            if chat_id not in self.users[username]["chats"]:
                self.users[username]["chats"].append(chat_id)
                self._save_users()
    
    def get_user_info(self, username: str) -> Optional[Dict]:
        """Get user information"""
        return self.users.get(username)
