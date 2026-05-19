"use client";

import { Button, Input, Modal, Select, Tag, message } from "antd";
import { useEffect, useMemo, useRef, useState } from "react";

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 3;

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function pctFromClient(e, rect) {
  const x = ((e.clientX - rect.left) / rect.width) * 100;
  const y = ((e.clientY - rect.top) / rect.height) * 100;
  return { x, y };
}

function normalizeBox(b) {
  const x = Number(b?.x || 0);
  const y = Number(b?.y || 0);
  const w = Number(b?.width || 0);
  const h = Number(b?.height || 0);
  return {
    x: Math.min(x, x + w),
    y: Math.min(y, y + h),
    width: Math.abs(w),
    height: Math.abs(h),
    angle: b?.angle != null ? Number(b.angle) : 0,
  };
}

function pointInAABB(p, b) {
  const bb = normalizeBox(b);
  return p.x >= bb.x && p.x <= bb.x + bb.width && p.y >= bb.y && p.y <= bb.y + bb.height;
}

function rotatePoint(px, py, cx, cy, angleRad) {
  const dx = px - cx;
  const dy = py - cy;
  const cos = Math.cos(angleRad);
  const sin = Math.sin(angleRad);
  return { x: cx + dx * cos - dy * sin, y: cy + dx * sin + dy * cos };
}

function pointInOBB(p, b) {
  const bb = normalizeBox(b);
  const angle = Number(bb.angle || 0);
  const cx = bb.x + bb.width / 2;
  const cy = bb.y + bb.height / 2;
  const inv = rotatePoint(p.x, p.y, cx, cy, -angle);
  return pointInAABB(inv, { x: bb.x, y: bb.y, width: bb.width, height: bb.height });
}

function getObbRotateHandle(b) {
  const bb = normalizeBox(b);
  const angle = Number(bb.angle || 0);
  const cx = bb.x + bb.width / 2;
  const cy = bb.y + bb.height / 2;
  const top = { x: cx, y: bb.y };
  const handle = rotatePoint(top.x, top.y - 6, cx, cy, angle);
  return { cx, cy, handle };
}

function getResizeHandle(p, bbox, isObb) {
  const bb = normalizeBox(bbox);
  const angle = isObb ? Number(bb.angle || 0) : 0;
  const cx = bb.x + bb.width / 2;
  const cy = bb.y + bb.height / 2;
  const raw = [
    { key: "tl", x: bb.x, y: bb.y },
    { key: "tr", x: bb.x + bb.width, y: bb.y },
    { key: "bl", x: bb.x, y: bb.y + bb.height },
    { key: "br", x: bb.x + bb.width, y: bb.y + bb.height },
  ].map((c) => (angle ? { ...c, ...rotatePoint(c.x, c.y, cx, cy, angle) } : c));
  const TH = 2.2;
  for (const c of raw) {
    if (Math.hypot(p.x - c.x, p.y - c.y) <= TH) return c.key;
  }
  return "";
}

export default function AnnotationCanvas({
  imageUrl,
  imageName,
  annotations,
  selectedTool,
  selectedAnnotationIndex,
  onSelectAnnotationIndex,
  onChangeAnnotations,
}) {
  const wrapperRef = useRef(null);
  const innerRef = useRef(null);
  const imgRef = useRef(null);
  const canvasRef = useRef(null);

  const [imageLoaded, setImageLoaded] = useState(false);

  const [zoomLevel, setZoomLevel] = useState(1);
  const [panX, setPanX] = useState(0);
  const [panY, setPanY] = useState(0);
  const [isSpacePressed, setIsSpacePressed] = useState(false);
  const [isPanning, setIsPanning] = useState(false);
  const panStartRef = useRef({ x: 0, y: 0, panX: 0, panY: 0 });

  const [draftBox, setDraftBox] = useState(null);
  const [drawing, setDrawing] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [resizing, setResizing] = useState(false);
  const [rotating, setRotating] = useState(false);
  const dragStateRef = useRef(null);

  const [labelModalOpen, setLabelModalOpen] = useState(false);
  const [editingLabel, setEditingLabel] = useState("");
  const [pendingIndex, setPendingIndex] = useState(-1);

  const [contextOpen, setContextOpen] = useState(false);
  const [contextPos, setContextPos] = useState({ x: 0, y: 0 });
  const [contextIndex, setContextIndex] = useState(-1);

  const categoryPresets = useMemo(
    () => ["person", "car", "dog", "cat", "bicycle", "motorcycle", "bus", "truck", "bird", "other"],
    []
  );

  const innerStyle = useMemo(
    () => ({
      transform: `translate(${panX}px, ${panY}px) scale(${zoomLevel})`,
      transformOrigin: "0 0",
    }),
    [panX, panY, zoomLevel]
  );

  const draw = () => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !imageLoaded) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const drawBox = (ann, idx, isDraft = false) => {
      if (!ann?.bbox) return;
      const bb = normalizeBox(ann.bbox);
      const x = (bb.x / 100) * rect.width * scaleX;
      const y = (bb.y / 100) * rect.height * scaleY;
      const w = (bb.width / 100) * rect.width * scaleX;
      const h = (bb.height / 100) * rect.height * scaleY;
      const angle = Number(bb.angle || 0);

      const isSelected = idx === selectedAnnotationIndex;
      const stroke = isDraft ? "#1677ff" : isSelected ? "#1677ff" : ann.type === "obb" ? "#fa8c16" : "#52c41a";
      ctx.save();
      if (ann.type === "obb" && angle) {
        const cx = x + w / 2;
        const cy = y + h / 2;
        ctx.translate(cx, cy);
        ctx.rotate(angle);
        ctx.translate(-cx, -cy);
      }
      ctx.lineWidth = isSelected ? 2.5 : 2;
      ctx.strokeStyle = stroke;
      ctx.fillStyle = "rgba(22,119,255,0.06)";
      ctx.strokeRect(x, y, w, h);
      if (isSelected) ctx.fillRect(x, y, w, h);

      ctx.fillStyle = stroke;
      const hs = 6;
      ctx.fillRect(x - hs / 2, y - hs / 2, hs, hs);
      ctx.fillRect(x + w - hs / 2, y - hs / 2, hs, hs);
      ctx.fillRect(x - hs / 2, y + h - hs / 2, hs, hs);
      ctx.fillRect(x + w - hs / 2, y + h - hs / 2, hs, hs);

      if (ann.type === "obb") {
        const handleX = x + w / 2;
        const handleY = y - 14;
        ctx.beginPath();
        ctx.strokeStyle = stroke;
        ctx.moveTo(x + w / 2, y);
        ctx.lineTo(handleX, handleY);
        ctx.stroke();
        ctx.beginPath();
        ctx.fillStyle = "#34d399";
        ctx.arc(handleX, handleY, 6, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.restore();

      if (ann.label) {
        ctx.save();
        ctx.fillStyle = "rgba(15,23,42,0.85)";
        ctx.font = "12px ui-sans-serif, system-ui";
        ctx.fillText(String(ann.label), x + 4, Math.max(12, y - 6));
        ctx.restore();
      }
    };

    const list = Array.isArray(annotations) ? annotations : [];
    list.forEach((ann, idx) => drawBox(ann, idx, false));
    if (draftBox) drawBox(draftBox, -2, true);
  };

  useEffect(() => {
    draw();
  }, [annotations, draftBox, selectedAnnotationIndex, imageLoaded, zoomLevel, panX, panY]);

  useEffect(() => {
    const onKeyDown = (e) => {
      const tag = String(e?.target?.tagName || "").toLowerCase();
      const editing = tag === "input" || tag === "textarea" || e?.target?.isContentEditable;
      if (editing) return;
      if (e.code === "Space") setIsSpacePressed(true);
      if (e.key === "Escape") {
        setContextOpen(false);
        setDraftBox(null);
        setDrawing(false);
        setDragging(false);
        setResizing(false);
        setRotating(false);
        dragStateRef.current = null;
      }
    };
    const onKeyUp = (e) => {
      if (e.code === "Space") setIsSpacePressed(false);
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, []);

  const resizeCanvasToImage = () => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
    setZoomLevel(1);
    setPanX(0);
    setPanY(0);
    setImageLoaded(true);
  };

  useEffect(() => {
    setImageLoaded(false);
  }, [imageUrl]);

  useEffect(() => {
    const onResize = () => {
      if (!imageLoaded) return;
      resizeCanvasToImage();
      draw();
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [imageLoaded]);

  const handleWheel = (e) => {
    if (!e.ctrlKey) return;
    e.preventDefault();
    const wrapper = wrapperRef.current;
    if (!wrapper) return;
    const rect = wrapper.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    setZoomLevel((prev) => {
      const next = clamp(prev * factor, MIN_ZOOM, MAX_ZOOM);
      setPanX((px) => mouseX - (mouseX - px) * (next / prev));
      setPanY((py) => mouseY - (mouseY - py) * (next / prev));
      return next;
    });
  };

  const zoomIn = () => setZoomLevel((z) => clamp(z + 0.25, MIN_ZOOM, MAX_ZOOM));
  const zoomOut = () => setZoomLevel((z) => clamp(z - 0.25, MIN_ZOOM, MAX_ZOOM));
  const resetZoomPan = () => {
    setZoomLevel(1);
    setPanX(0);
    setPanY(0);
  };

  const findHit = (p) => {
    const list = Array.isArray(annotations) ? annotations : [];
    for (let i = list.length - 1; i >= 0; i -= 1) {
      const ann = list[i];
      if (!ann?.bbox) continue;
      const resizeKey = getResizeHandle(p, ann.bbox, ann.type === "obb");
      if (resizeKey) return { index: i, hit: `resize:${resizeKey}` };
      if (ann.type === "obb") {
        const bb = normalizeBox(ann.bbox);
        const { handle } = getObbRotateHandle(bb);
        const dist = Math.hypot(p.x - handle.x, p.y - handle.y);
        if (dist < 3) return { index: i, hit: "rotate" };
        if (pointInOBB(p, bb)) return { index: i, hit: "body" };
      } else {
        if (pointInAABB(p, ann.bbox)) return { index: i, hit: "body" };
      }
    }
    return { index: -1, hit: "" };
  };

  const beginLabelForIndex = (idx) => {
    const ann = annotations?.[idx];
    setPendingIndex(idx);
    setEditingLabel(String(ann?.label || ""));
    setLabelModalOpen(true);
  };

  const commitLabel = () => {
    const value = String(editingLabel || "").trim();
    if (!value) {
      message.warning("请输入标签名称");
      return;
    }
    const list = Array.isArray(annotations) ? [...annotations] : [];
    if (pendingIndex < 0 || pendingIndex >= list.length) return;
    list[pendingIndex] = { ...list[pendingIndex], label: value };
    onChangeAnnotations?.(list);
    setLabelModalOpen(false);
    setPendingIndex(-1);
  };

  const cancelLabel = () => {
    setLabelModalOpen(false);
    if (pendingIndex >= 0 && pendingIndex < (annotations || []).length) {
      const list = Array.isArray(annotations) ? [...annotations] : [];
      if (!list[pendingIndex]?.label) {
        list.splice(pendingIndex, 1);
        onChangeAnnotations?.(list);
      }
    }
    setPendingIndex(-1);
  };

  const onMouseDown = (e) => {
    if (!canvasRef.current) return;
    setContextOpen(false);
    if (e.button === 2) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const p = pctFromClient(e, rect);

    if (isSpacePressed) {
      setIsPanning(true);
      panStartRef.current = { x: e.clientX, y: e.clientY, panX, panY };
      return;
    }

    const { index, hit } = findHit(p);
    if (index >= 0) {
      onSelectAnnotationIndex?.(index);
      dragStateRef.current = { index, start: p, origin: normalizeBox(annotations[index].bbox), hit };
      if (hit === "rotate") setRotating(true);
      else if (String(hit || "").startsWith("resize:")) setResizing(true);
      else setDragging(true);
      return;
    }

    if (selectedTool !== "object_detection" && selectedTool !== "object_detection_obb") return;

    setDrawing(true);
    const type = selectedTool === "object_detection_obb" ? "obb" : "bounding_box";
    const newAnn = { type, label: "", bbox: { x: p.x, y: p.y, width: 0, height: 0, angle: 0 } };
    setDraftBox(newAnn);
  };

  const onMouseMove = (e) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const p = pctFromClient(e, rect);

    if (isPanning) {
      const s = panStartRef.current;
      setPanX(s.panX + (e.clientX - s.x));
      setPanY(s.panY + (e.clientY - s.y));
      return;
    }

    if (drawing && draftBox?.bbox) {
      const b0 = draftBox.bbox;
      setDraftBox({
        ...draftBox,
        bbox: { ...b0, width: p.x - b0.x, height: p.y - b0.y },
      });
      return;
    }

    if ((dragging || rotating || resizing) && dragStateRef.current) {
      const { index, start, origin, hit } = dragStateRef.current;
      const list = Array.isArray(annotations) ? [...annotations] : [];
      const ann = list[index];
      if (!ann?.bbox) return;

      if (rotating && ann.type === "obb") {
        const bb = origin;
        const cx = bb.x + bb.width / 2;
        const cy = bb.y + bb.height / 2;
        const a0 = Math.atan2(start.y - cy, start.x - cx);
        const a1 = Math.atan2(p.y - cy, p.x - cx);
        const angle = (bb.angle || 0) + (a1 - a0);
        list[index] = { ...ann, bbox: { ...bb, angle } };
        onChangeAnnotations?.(list);
        return;
      }

      if (resizing && String(hit || "").startsWith("resize:")) {
        const key = String(hit).split(":")[1] || "";
        const bb = origin;
        const angle = ann.type === "obb" ? Number(bb.angle || 0) : 0;
        const cx = bb.x + bb.width / 2;
        const cy = bb.y + bb.height / 2;
        const pLocal = angle ? rotatePoint(p.x, p.y, cx, cy, -angle) : p;
        const startLocal = angle ? rotatePoint(start.x, start.y, cx, cy, -angle) : start;
        const dx = pLocal.x - startLocal.x;
        const dy = pLocal.y - startLocal.y;
        let x1 = bb.x;
        let y1 = bb.y;
        let x2 = bb.x + bb.width;
        let y2 = bb.y + bb.height;
        if (key === "tl") {
          x1 += dx;
          y1 += dy;
        } else if (key === "tr") {
          x2 += dx;
          y1 += dy;
        } else if (key === "bl") {
          x1 += dx;
          y2 += dy;
        } else if (key === "br") {
          x2 += dx;
          y2 += dy;
        }
        const minSize = 0.6;
        const nx = Math.min(x1, x2);
        const ny = Math.min(y1, y2);
        const nw = Math.max(minSize, Math.abs(x2 - x1));
        const nh = Math.max(minSize, Math.abs(y2 - y1));
        list[index] = { ...ann, bbox: { ...bb, x: nx, y: ny, width: nw, height: nh, angle } };
        onChangeAnnotations?.(list);
        return;
      }

      if (dragging) {
        const dx = p.x - start.x;
        const dy = p.y - start.y;
        list[index] = { ...ann, bbox: { ...origin, x: origin.x + dx, y: origin.y + dy } };
        onChangeAnnotations?.(list);
      }
    }
  };

  const onMouseUp = () => {
    if (isPanning) {
      setIsPanning(false);
      return;
    }

    if (drawing && draftBox?.bbox) {
      const bb = normalizeBox(draftBox.bbox);
      if (bb.width < 0.5 || bb.height < 0.5) {
        setDraftBox(null);
        setDrawing(false);
        return;
      }
      const list = Array.isArray(annotations) ? [...annotations] : [];
      const finalAnn = { ...draftBox, bbox: bb };
      const idx = list.length;
      list.push(finalAnn);
      onChangeAnnotations?.(list);
      setDraftBox(null);
      setDrawing(false);
      setTimeout(() => beginLabelForIndex(idx), 0);
      return;
    }

    setDragging(false);
    setResizing(false);
    setRotating(false);
    dragStateRef.current = null;
  };

  const onContextMenu = (e) => {
    e.preventDefault();
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const p = pctFromClient(e, rect);
    const hit = findHit(p);
    if (hit.index < 0) return;
    setContextIndex(hit.index);
    setContextPos({ x: e.clientX, y: e.clientY });
    setContextOpen(true);
  };

  useEffect(() => {
    const onGlobalClick = () => setContextOpen(false);
    document.addEventListener("click", onGlobalClick);
    return () => document.removeEventListener("click", onGlobalClick);
  }, []);

  const deleteAnnotationAt = (idx) => {
    const list = Array.isArray(annotations) ? [...annotations] : [];
    if (idx < 0 || idx >= list.length) return;
    list.splice(idx, 1);
    onChangeAnnotations?.(list);
  };

  return (
    <div style={{ position: "relative" }}>
      <div
        ref={wrapperRef}
        onWheel={handleWheel}
        style={{
          borderRadius: 14,
          border: "1px solid rgba(15,23,42,0.08)",
          background: "#fff",
          overflow: "hidden",
          position: "relative",
          minHeight: 420,
        }}
      >
        <div ref={innerRef} style={innerStyle}>
          <div style={{ position: "relative" }}>
            <img
              ref={imgRef}
              src={imageUrl}
              alt={imageName || "image"}
              style={{ display: "block", maxWidth: "100%" }}
              onLoad={() => {
                resizeCanvasToImage();
                setTimeout(() => draw(), 0);
              }}
            />
            <canvas
              ref={canvasRef}
              style={{ position: "absolute", inset: 0, width: "100%", height: "100%", cursor: isSpacePressed ? "grab" : "crosshair" }}
              onMouseDown={onMouseDown}
              onMouseMove={onMouseMove}
              onMouseUp={onMouseUp}
              onMouseLeave={onMouseUp}
              onContextMenu={onContextMenu}
            />

            {imageLoaded ? (
              <div style={{ position: "absolute", left: 12, top: 12, padding: "6px 10px", borderRadius: 999, background: "rgba(255,255,255,0.85)", border: "1px solid rgba(15,23,42,0.08)", fontSize: 12, color: "rgba(15,23,42,0.7)" }}>
                {selectedTool === "object_detection"
                  ? "矩形框：左键拖拽绘制；右键可编辑/删除；Ctrl+滚轮缩放；空格+拖拽平移"
                  : selectedTool === "object_detection_obb"
                  ? "旋转框(OBB)：左键拖拽绘制；绿色圆点旋转；右键可编辑/删除；Ctrl+滚轮缩放；空格+拖拽平移"
                  : "当前工具仅用于 AI 自动标注/展示"}
              </div>
            ) : null}
          </div>
        </div>
      </div>

      {imageLoaded ? (
        <div style={{ position: "absolute", right: 14, top: 14, display: "flex", alignItems: "center", gap: 8, padding: "8px 10px", borderRadius: 12, background: "rgba(255,255,255,0.92)", border: "1px solid rgba(15,23,42,0.08)" }}>
          <span style={{ fontSize: 12, color: "rgba(15,23,42,0.65)" }}>画布缩放</span>
          <Button size="small" onClick={zoomOut} disabled={zoomLevel <= MIN_ZOOM}>
            −
          </Button>
          <span style={{ fontSize: 12, width: 52, textAlign: "center" }}>{Math.round(zoomLevel * 100)}%</span>
          <Button size="small" onClick={zoomIn} disabled={zoomLevel >= MAX_ZOOM}>
            +
          </Button>
          <Button size="small" onClick={resetZoomPan}>
            1:1
          </Button>
        </div>
      ) : null}

      <Modal
        open={labelModalOpen}
        onCancel={cancelLabel}
        onOk={commitLabel}
        title="输入标签"
        okText="确定"
        cancelText="取消"
        destroyOnHidden
      >
        <div style={{ display: "grid", gap: 10 }}>
          <div>
            <div style={{ marginBottom: 6 }}>标签名称</div>
            <Input value={editingLabel} onChange={(e) => setEditingLabel(e.target.value)} onPressEnter={commitLabel} />
          </div>
          <div>
            <div style={{ marginBottom: 6 }}>快速选择</div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              {categoryPresets.map((c) => (
                <Tag key={c} style={{ cursor: "pointer" }} onClick={() => setEditingLabel(c)}>
                  {c}
                </Tag>
              ))}
            </div>
          </div>
          <div>
            <div style={{ marginBottom: 6 }}>选择/创建类别</div>
            <Select
              showSearch
              allowClear
              value={editingLabel || undefined}
              onChange={(v) => setEditingLabel(String(v || ""))}
              options={categoryPresets.map((c) => ({ label: c, value: c }))}
              placeholder="可输入新类别"
              style={{ width: "100%" }}
            />
          </div>
        </div>
      </Modal>

      {contextOpen ? (
        <div
          style={{
            position: "fixed",
            left: contextPos.x,
            top: contextPos.y,
            background: "#fff",
            border: "1px solid rgba(15,23,42,0.12)",
            borderRadius: 10,
            boxShadow: "0 10px 30px rgba(15,23,42,0.12)",
            zIndex: 9999,
            overflow: "hidden",
            minWidth: 120,
          }}
        >
          <div
            onClick={() => {
              setContextOpen(false);
              beginLabelForIndex(contextIndex);
            }}
            style={{ padding: "10px 12px", cursor: "pointer" }}
          >
            编辑标签
          </div>
          <div
            onClick={() => {
              setContextOpen(false);
              deleteAnnotationAt(contextIndex);
              message.success("已删除标注");
            }}
            style={{ padding: "10px 12px", cursor: "pointer", color: "#cf1322" }}
          >
            删除标注
          </div>
        </div>
      ) : null}
    </div>
  );
}
