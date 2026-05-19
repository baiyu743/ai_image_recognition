"use client";

import { Button, Card } from "antd";
import { DownOutlined, UpOutlined } from "@ant-design/icons";
import { useMemo, useState } from "react";

const TYPE_NAME = {
  rectanglelabels: "边界框",
  polygonlabels: "多边形",
  keypointlabels: "关键点",
  bbox: "目标检测",
  bounding_box: "矩形框",
  obb: "旋转框(OBB)",
  polygon: "图像分割",
  classification: "图像分类",
};

function shouldShowAngle(annotation) {
  if (!annotation?.bbox) return false;
  if (annotation.type === "obb") return true;
  return annotation.bbox.angle != null && annotation.bbox.angle !== 0;
}

function formatAngle(rad) {
  const n = Number(rad);
  if (!Number.isFinite(n)) return "0";
  return ((n * 180) / Math.PI).toFixed(1);
}

export default function AnnotationList({ annotations, initialVisible = true, selectedIndex = -1, onSelect }) {
  const [visible, setVisible] = useState(initialVisible);

  const items = useMemo(() => (Array.isArray(annotations) ? annotations : []), [annotations]);

  return (
    <Card
      size="small"
      className="app-card"
      title="标注结果"
      extra={
        <Button type="text" onClick={() => setVisible((v) => !v)} icon={visible ? <UpOutlined /> : <DownOutlined />} />
      }
      style={{ height: "fit-content" }}
      styles={{ body: { display: visible ? "block" : "none" } }}
    >
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(min(100%, 260px), 1fr))", gap: 10 }}>
        {items.map((ann, index) => {
          const active = selectedIndex === index;
          return (
            <div
              key={index}
              onClick={() => onSelect?.(index)}
              style={{
                border: `1px solid ${active ? "#1677ff" : "rgba(15,23,42,0.10)"}`,
                background: active ? "rgba(22,119,255,0.06)" : "#fff",
                borderRadius: 10,
                padding: 10,
                cursor: "pointer",
              }}
            >
              <div style={{ fontWeight: 800, color: "#1677ff", marginBottom: 6 }}>{TYPE_NAME[ann?.type] || ann?.type}</div>
              <div style={{ fontSize: 13, color: "rgba(15,23,42,0.72)", lineHeight: 1.6 }}>
                {ann?.type === "bbox" || ann?.type === "bounding_box" || ann?.type === "obb" ? (
                  <>
                    类型：{ann.type === "obb" ? "旋转框(OBB)" : "矩形框"}
                    <br />
                    标签：{ann.label}
                    <br />
                    {ann.confidence != null ? (
                      <>
                        置信度：{(Number(ann.confidence) * 100).toFixed(2)}%<br />
                      </>
                    ) : null}
                    位置：x={(Number(ann.bbox?.x ?? 0)).toFixed(2)}%，y={(Number(ann.bbox?.y ?? 0)).toFixed(2)}%
                    <br />
                    尺寸：{(Number(ann.bbox?.width ?? 0)).toFixed(2)}% × {(Number(ann.bbox?.height ?? 0)).toFixed(2)}%
                    {shouldShowAngle(ann) ? (
                      <>
                        <br />
                        旋转：{formatAngle(ann.bbox?.angle)}°
                      </>
                    ) : null}
                  </>
                ) : ann?.type === "polygon" ? (
                  <>
                    标签：{ann.label}
                    <br />
                    置信度：{(Number(ann.confidence || 0) * 100).toFixed(2)}%
                    <br />
                    顶点数：{Array.isArray(ann.points) ? ann.points.length : 0}
                  </>
                ) : ann?.type === "classification" ? (
                  <>
                    标签：{ann.label}
                    <br />
                    置信度：{(Number(ann.confidence || 0) * 100).toFixed(2)}%
                  </>
                ) : (
                  <span>该标注类型的详情展示待完善</span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </Card>
  );
}
