import os
import re
import shutil
import subprocess
import tarfile
import threading
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

PROJECT_ID_PATTERN = re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$", re.I)

AUTODL_HOST = os.getenv("AUTODL_HOST", "autodl-backend")
AUTODL_PROJECTS_ROOT = os.getenv("AUTODL_PROJECTS_ROOT", "/root/autodl-tmp/projects")
CONTROL_PATH = os.getenv("AUTODL_SSH_CONTROL_PATH", "/tmp/autodl-gw-%r@%h:%p")
CONTROL_PERSIST = os.getenv("AUTODL_SSH_CONTROL_PERSIST", "600")

SSH_BASE = [
    "ssh",
    "-o",
    "StrictHostKeyChecking=accept-new",
    "-o",
    f"ControlPath={CONTROL_PATH}",
    "-o",
    "ControlMaster=auto",
    "-o",
    f"ControlPersist={CONTROL_PERSIST}",
    AUTODL_HOST,
]

_master_lock = threading.Lock()
_master_started = False


def _require_safe_project_id(project_id: str) -> str:
    if not PROJECT_ID_PATTERN.match(project_id or ""):
        raise HTTPException(status_code=400, detail="Invalid project_id")
    return project_id


def _ensure_master() -> None:
    global _master_started
    if _master_started:
        return
    with _master_lock:
        if _master_started:
            return
        subprocess.run(
            SSH_BASE[:-1]
            + [
                "-Nf",
                "-o",
                "ExitOnForwardFailure=yes",
                AUTODL_HOST,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        _master_started = True


def _remote_mkdir(path: str) -> None:
    _ensure_master()
    cmd = SSH_BASE + ["mkdir", "-p", path]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise HTTPException(status_code=502, detail=f"AutoDL mkdir failed: {p.stderr.strip() or p.stdout.strip()}")


def _remote_write_file(remote_path: str, src_file) -> None:
    _ensure_master()
    cmd = SSH_BASE + ["bash", "-lc", f"cat > {shlex_quote(remote_path)}"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        shutil.copyfileobj(src_file, p.stdin, length=1024 * 1024)
        try:
            p.stdin.close()
        except Exception:
            pass
        p.wait()
        out = p.stdout.read() if p.stdout is not None else b""
        err = p.stderr.read() if p.stderr is not None else b""
    except Exception:
        try:
            p.kill()
        except Exception:
            pass
        raise
    if p.returncode != 0:
        raise HTTPException(status_code=502, detail=f"AutoDL write failed: {(err or b'').decode(errors='ignore')}")

def _seekable_file_size(fileobj) -> int:
    try:
        cur = fileobj.tell()
        fileobj.seek(0, os.SEEK_END)
        size = int(fileobj.tell())
        fileobj.seek(cur, os.SEEK_SET)
        return size
    except Exception:
        return -1


def _file_size(fileobj) -> int:
    try:
        fileno = fileobj.fileno()
        return int(os.fstat(fileno).st_size)
    except Exception:
        return _seekable_file_size(fileobj)


def _remote_write_tar(remote_dir: str, uploads: List[UploadFile], names: List[str]) -> None:
    _ensure_master()
    _remote_mkdir(remote_dir)
    cmd = SSH_BASE + ["tar", "-x", "--no-same-owner", "--no-same-permissions", "-C", remote_dir, "-f", "-"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        if p.stdin is None:
            raise RuntimeError("stdin not available")
        with tarfile.open(fileobj=p.stdin, mode="w|") as tar:
            for upload, filename in zip(uploads, names):
                fileobj = upload.file
                try:
                    fileobj.seek(0, os.SEEK_SET)
                except Exception:
                    pass
                size = _file_size(fileobj)
                if size < 0:
                    raise HTTPException(status_code=400, detail=f"File not seekable: {filename}")
                try:
                    fileobj.seek(0, os.SEEK_SET)
                except Exception:
                    pass
                info = tarfile.TarInfo(name=filename)
                info.size = size
                tar.addfile(info, fileobj=fileobj)
        p.wait()
        err = p.stderr.read() if p.stderr is not None else b""
        if p.returncode != 0:
            raise HTTPException(status_code=502, detail=f"AutoDL tar extract failed: {(err or b'').decode(errors='ignore')}")
    finally:
        try:
            if p.stdin:
                p.stdin.close()
        except Exception:
            pass
        try:
            if p.stdout:
                p.stdout.close()
        except Exception:
            pass
        try:
            if p.stderr:
                p.stderr.close()
        except Exception:
            pass
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass


def shlex_quote(s: str) -> str:
    s = str(s)
    if not s:
        return "''"
    if re.fullmatch(r"[a-zA-Z0-9_./:-]+", s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _sanitize_filename(name: str) -> str:
    name = os.path.basename(name or "").strip()
    if not name:
        return "image.jpg"
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def _unique_names(files: List[UploadFile], *, overwrite: bool) -> List[str]:
    used = set()
    out = []
    for f in files:
        base = _sanitize_filename(f.filename)
        stem, ext = os.path.splitext(base)
        ext = ext or ".jpg"
        safe_stem = stem or "image"
        if overwrite:
            candidate = f"{safe_stem}{ext}"
            if candidate in used:
                candidate = f"{safe_stem}_{uuid.uuid4().hex[:8]}{ext}"
        else:
            candidate = f"{safe_stem}_{uuid.uuid4().hex[:8]}{ext}"
            while candidate in used:
                candidate = f"{safe_stem}_{uuid.uuid4().hex[:8]}{ext}"
        used.add(candidate)
        out.append(candidate)
    return out


app = FastAPI(title="Upload Gateway", version="1.0.0")


@app.post("/upload-api/projects/{project_id}/staging/images")
async def upload_to_autodl_staging(
    project_id: str,
    images: List[UploadFile] = File(...),
    overwrite: int = Form(0),
):
    project_id = _require_safe_project_id(project_id)
    if not images:
        raise HTTPException(status_code=400, detail="No images")

    remote_dir = str(Path(AUTODL_PROJECTS_ROOT) / project_id / "staging" / "images")
    do_overwrite = bool(int(overwrite or 0))
    names = _unique_names(images, overwrite=do_overwrite)

    stored = []
    _remote_write_tar(remote_dir, images, names)

    for filename, upload in zip(names, images):
        try:
            upload.file.close()
        except Exception:
            pass
        stored.append({"name": filename, "url_path": f"/api/projects/{project_id}/staging/images/{filename}"})
    return JSONResponse(content={"success": True, "count": len(stored), "images": stored})
