"""
Authentication, Permissions & Audit Log.

Provides:
  - Password-based authentication (bcrypt hashed, config-file backed)
  - Session-layer permission model (Board / Analyst / Viewer)
  - Audit logger (JSON lines, every action timestamped)
  - State hash snapshots at login/logout for drift detection

Design:
  - Config file (users.json) maps email → {password_hash, role, name}
  - Passwords are bcrypt-hashed — never stored in plaintext
  - Permission checks are decorative in Streamlit (UI hiding) but
    enforced at the function level via require_permission()
  - Audit log is append-only; one JSON object per line
  - State snapshots use SHA-256 of serialized asset_state + tranches

Roles:
  - board:   read + write + sandbox + Board override
  - analyst: read + write + sandbox (no override)
  - viewer:  read + sandbox (no write)

Sandbox is available to ALL roles — it never writes to state.

Usage in app.py:
    from src.auth import init_auth, login_gate, require_permission, log_action, get_current_user
    init_auth(data_dir)
    if not login_gate():
        st.stop()
    # ... user is authenticated, proceed
    require_permission("write")  # raises if viewer
    log_action("run_mc", {"n_sims": 1000})
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import bcrypt
    HAS_BCRYPT = True
except ImportError:
    HAS_BCRYPT = False

try:
    import streamlit as st
    HAS_ST = True
except ImportError:
    HAS_ST = False


# ── Role definitions ──────────────────────────────────────────────────────

ROLES = {
    "board": {
        "label": "Board (BoD Override)",
        "permissions": {"read", "write", "sandbox", "override", "admin"},
        "level": 3,
    },
    "analyst": {
        "label": "Analyst (Read/Write)",
        "permissions": {"read", "write", "sandbox"},
        "level": 2,
    },
    "viewer": {
        "label": "Viewer (Read-Only)",
        "permissions": {"read", "sandbox"},
        "level": 1,
    },
}


# ── Password hashing ─────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    """Hash a plaintext password with bcrypt (or SHA-256 fallback)."""
    if HAS_BCRYPT:
        return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()
    # Fallback: SHA-256 with static salt (less secure, for environments without bcrypt)
    salted = f"discovery_salt_v1:{plain}"
    return "sha256:" + hashlib.sha256(salted.encode()).hexdigest()


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a stored hash."""
    if hashed.startswith("sha256:"):
        salted = f"discovery_salt_v1:{plain}"
        return "sha256:" + hashlib.sha256(salted.encode()).hexdigest() == hashed
    if HAS_BCRYPT:
        try:
            return bcrypt.checkpw(plain.encode(), hashed.encode())
        except Exception:
            return False
    return False


# ── User config file ──────────────────────────────────────────────────────

def _users_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / "users.json"


def load_users(data_dir: str | Path) -> dict:
    """Load user config. Returns {email: {password_hash, role, name}}."""
    path = _users_path(data_dir)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_users(data_dir: str | Path, users: dict):
    """Save user config."""
    path = _users_path(data_dir)
    with open(path, "w") as f:
        json.dump(users, f, indent=2)


def create_default_users(data_dir: str | Path):
    """Create a default users.json if none exists."""
    path = _users_path(data_dir)
    if path.exists():
        return
    default_users = {
        "admin@discoverybiotech.com": {
            "name": "Admin",
            "role": "board",
            "password_hash": hash_password("changeme"),
        },
    }
    save_users(data_dir, default_users)


def add_user(data_dir: str | Path, email: str, name: str,
             role: str, password: str):
    """Add a user to the config file."""
    if role not in ROLES:
        raise ValueError(f"Invalid role: {role}. Must be one of {list(ROLES.keys())}")
    users = load_users(data_dir)
    users[email] = {
        "name": name,
        "role": role,
        "password_hash": hash_password(password),
    }
    save_users(data_dir, users)


def remove_user(data_dir: str | Path, email: str):
    """Remove a user from the config file."""
    users = load_users(data_dir)
    if email in users:
        del users[email]
        save_users(data_dir, users)


# ── State hashing ─────────────────────────────────────────────────────────

def compute_state_hash(asset_state: pd.DataFrame, tranches: pd.DataFrame) -> str:
    """Compute SHA-256 hash of portfolio state for drift detection."""
    content = asset_state.to_csv(index=False) + "||" + tranches.to_csv(index=False)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ── Audit log ─────────────────────────────────────────────────────────────

def _audit_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / "audit_log.jsonl"


def log_action(
    data_dir: str | Path,
    user_email: str,
    action: str,
    details: dict | None = None,
    state_hash: str | None = None,
):
    """Append an audit entry to the log file."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user": user_email,
        "action": action,
        "details": details or {},
    }
    if state_hash:
        entry["state_hash"] = state_hash
    path = _audit_path(data_dir)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_audit_log(data_dir: str | Path, last_n: int = 50) -> list[dict]:
    """Read the last N audit log entries."""
    path = _audit_path(data_dir)
    if not path.exists():
        return []
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries[-last_n:]


# ── Streamlit session integration ─────────────────────────────────────────

def init_auth(data_dir: str | Path):
    """Initialize auth system. Call once at app startup."""
    create_default_users(data_dir)
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user_email = None
        st.session_state.user_name = None
        st.session_state.user_role = None
        st.session_state.login_time = None
        st.session_state.login_state_hash = None


def login_gate(data_dir: str | Path, asset_state=None, tranches=None) -> bool:
    """
    Show login form if not authenticated. Returns True if user is logged in.

    Call at the top of app.py:
        if not login_gate(DATA_DIR):
            st.stop()
    """
    if st.session_state.get("authenticated"):
        return True

    st.markdown(
        '<div style="max-width:400px;margin:60px auto;padding:30px;'
        'border:1px solid #E2E8F0;border-radius:12px;">'
        '<h2 style="text-align:center;color:#4A5568;">Discovery Portfolio Engine</h2>'
        '<p style="text-align:center;color:#718096;">Please sign in to continue</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="you@discoverybiotech.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

        if submitted:
            users = load_users(data_dir)
            if email in users and verify_password(password, users[email]["password_hash"]):
                st.session_state.authenticated = True
                st.session_state.user_email = email
                st.session_state.user_name = users[email]["name"]
                st.session_state.user_role = users[email]["role"]
                st.session_state.login_time = datetime.now().isoformat()

                # State hash at login
                state_hash = None
                if asset_state is not None and tranches is not None:
                    state_hash = compute_state_hash(asset_state, tranches)
                    st.session_state.login_state_hash = state_hash

                log_action(data_dir, email, "login", {
                    "role": users[email]["role"],
                    "name": users[email]["name"],
                }, state_hash=state_hash)

                st.rerun()
            elif submitted:
                st.error("Invalid email or password")
                log_action(data_dir, email or "unknown", "login_failed")

    return False


def logout(data_dir: str | Path, asset_state=None, tranches=None):
    """Log out the current user."""
    email = st.session_state.get("user_email", "unknown")

    state_hash = None
    if asset_state is not None and tranches is not None:
        state_hash = compute_state_hash(asset_state, tranches)

    # Detect state drift
    login_hash = st.session_state.get("login_state_hash")
    drift = None
    if login_hash and state_hash:
        drift = "CHANGED" if login_hash != state_hash else "unchanged"

    log_action(data_dir, email, "logout", {
        "session_duration": _session_duration(),
        "state_drift": drift,
    }, state_hash=state_hash)

    for key in ["authenticated", "user_email", "user_name", "user_role",
                "login_time", "login_state_hash"]:
        if key in st.session_state:
            del st.session_state[key]

    st.rerun()


def _session_duration() -> str:
    """Compute human-readable session duration."""
    login = st.session_state.get("login_time")
    if not login:
        return "unknown"
    try:
        start = datetime.fromisoformat(login)
        elapsed = datetime.now() - start
        mins = int(elapsed.total_seconds() / 60)
        return f"{mins}m"
    except Exception:
        return "unknown"


def get_current_user() -> dict:
    """Get current user info from session state."""
    return {
        "email": st.session_state.get("user_email"),
        "name": st.session_state.get("user_name"),
        "role": st.session_state.get("user_role"),
        "authenticated": st.session_state.get("authenticated", False),
    }


def has_permission(permission: str) -> bool:
    """Check if current user has a specific permission."""
    role = st.session_state.get("user_role")
    if not role or role not in ROLES:
        return False
    return permission in ROLES[role]["permissions"]


def require_permission(permission: str, silent: bool = False) -> bool:
    """
    Check permission. If silent=True, returns bool. Otherwise shows error.
    Use in app.py:
        if require_permission("write", silent=True):
            # show write controls
    """
    if has_permission(permission):
        return True
    if not silent:
        role_label = ROLES.get(st.session_state.get("user_role", ""), {}).get("label", "Unknown")
        st.error(f"Permission denied: '{permission}' requires a higher access level. "
                 f"Your role: {role_label}")
    return False


# ── User sidebar widget ──────────────────────────────────────────────────

def user_sidebar(data_dir: str | Path, asset_state=None, tranches=None):
    """
    Render user info + logout + admin panel in sidebar.
    Call inside `with st.sidebar:` block.
    """
    user = get_current_user()
    if not user["authenticated"]:
        return

    role_info = ROLES.get(user["role"], {})
    role_label = role_info.get("label", user["role"])

    st.markdown(
        f'<div style="background:#EDF2F7;padding:10px 14px;border-radius:8px;margin-bottom:8px;">'
        f'<div style="font-weight:600;color:#2D3748;">{user["name"]}</div>'
        f'<div style="font-size:12px;color:#718096;">{user["email"]}</div>'
        f'<div style="font-size:11px;color:#4A5568;margin-top:2px;">{role_label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if st.button("Sign Out", use_container_width=True, key="logout_btn"):
        logout(data_dir, asset_state, tranches)

    # Admin panel (board only)
    if has_permission("admin"):
        with st.expander("Admin Panel"):
            st.markdown("**User Management**")
            users = load_users(data_dir)
            for email, info in users.items():
                st.text(f"{info['name']} ({email}) — {info['role']}")

            st.markdown("---")
            st.markdown("**Add User**")
            new_email = st.text_input("Email", key="admin_new_email",
                                      placeholder="user@discoverybiotech.com")
            new_name = st.text_input("Name", key="admin_new_name")
            new_role = st.selectbox("Role", ["viewer", "analyst", "board"],
                                   key="admin_new_role")
            new_pw = st.text_input("Password", type="password", key="admin_new_pw")
            if st.button("Add User", key="admin_add_btn"):
                if new_email and new_name and new_pw:
                    add_user(data_dir, new_email, new_name, new_role, new_pw)
                    log_action(data_dir, user["email"], "add_user", {
                        "target_email": new_email, "target_role": new_role})
                    st.success(f"Added {new_email}")
                    st.rerun()
                else:
                    st.warning("Fill in all fields")

            st.markdown("---")
            st.markdown("**Audit Log (last 20)**")
            logs = read_audit_log(data_dir, last_n=20)
            if logs:
                log_df = pd.DataFrame(logs)
                display_cols = ["timestamp", "user", "action"]
                if "state_hash" in log_df.columns:
                    display_cols.append("state_hash")
                st.dataframe(log_df[display_cols], use_container_width=True,
                             hide_index=True)
            else:
                st.caption("No log entries yet")


# ── Confidentiality footer ────────────────────────────────────────────────

def confidentiality_footer():
    """Render sticky confidentiality footer on every page."""
    st.markdown(
        '<div style="position:fixed;bottom:0;left:0;width:100%;background:#2D3748;'
        'color:#A0AEC0;text-align:center;padding:6px 0;font-size:11px;z-index:999;'
        'border-top:1px solid #4A5568;">'
        'Discovery Biotech — Confidential  |  Authorized Recipients Only'
        '</div>',
        unsafe_allow_html=True,
    )
    # Add bottom padding so content isn't hidden behind footer
    st.markdown('<div style="height:40px;"></div>', unsafe_allow_html=True)
