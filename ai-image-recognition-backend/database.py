import os
import sqlite3
import time
from pathlib import Path
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import DatabaseError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 数据库配置 - 从环境变量读取
def _default_database_url() -> str:
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_file = os.path.join(base_dir, "auto_annotate.db")
    db_file_posix = Path(db_file).as_posix()
    if db_file_posix.startswith("/"):
        return f"sqlite:////{db_file_posix.lstrip('/')}"
    return f"sqlite:///{db_file_posix}"

DATABASE_URL = _default_database_url()

def _sqlite_db_file_from_url(database_url: str) -> Optional[str]:
    if not database_url.startswith("sqlite:"):
        return None
    try:
        url = make_url(database_url)
        db_file = url.database
    except Exception:
        return None
    if not db_file or db_file == ":memory:":
        return None
    if not os.path.isabs(db_file):
        db_file = os.path.abspath(db_file)
    return db_file

def _move_as_corrupt(db_file: str, ts: str) -> None:
    backup_path = f"{db_file}.corrupt_{ts}"
    try:
        os.replace(db_file, backup_path)
    except Exception:
        try:
            os.remove(db_file)
        except Exception:
            pass

    for suffix in ("-wal", "-shm"):
        p = f"{db_file}{suffix}"
        if not os.path.exists(p):
            continue
        try:
            os.replace(p, f"{backup_path}{suffix}")
        except Exception:
            try:
                os.remove(p)
            except Exception:
                pass

def _maybe_recover_sqlite_db(database_url: str, force: bool = False) -> bool:
    db_file = _sqlite_db_file_from_url(database_url)
    if not db_file:
        return False
    parent = os.path.dirname(db_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if not os.path.exists(db_file):
        return False

    ok = False
    if not force:
        try:
            conn = sqlite3.connect(db_file)
            try:
                row = conn.execute("PRAGMA integrity_check;").fetchone()
                ok = bool(row) and row[0] == "ok"
            finally:
                conn.close()
        except Exception:
            ok = False

    if ok:
        return False

    ts = time.strftime("%Y%m%d_%H%M%S")
    _move_as_corrupt(db_file, ts)
    return True

def _is_sqlite_malformed_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("database disk image is malformed" in msg) or ("disk image is malformed" in msg)

def recover_sqlite_db(force: bool = False) -> bool:
    if not DATABASE_URL.startswith("sqlite:"):
        return False
    engine.dispose()
    recovered = _maybe_recover_sqlite_db(DATABASE_URL, force=force)
    Base.metadata.create_all(bind=engine)
    return recovered

try:
    url = make_url(DATABASE_URL)
    print(f"Database URL: {url.render_as_string(hide_password=True)}")
    db_file = _sqlite_db_file_from_url(DATABASE_URL)
    if db_file:
        print(f"SQLite DB file: {db_file}")
except Exception:
    pass

_maybe_recover_sqlite_db(DATABASE_URL)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} # Needed for SQLite
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        try:
            db.execute(text("SELECT 1"))
        except DatabaseError as e:
            if _is_sqlite_malformed_error(e):
                db.close()
                recover_sqlite_db(force=True)
                db = SessionLocal()
                db.execute(text("SELECT 1"))
            else:
                raise
        yield db
    finally:
        db.close()
