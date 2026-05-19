"use client";

import { Button, Checkbox, Select, Space, Tag } from "antd";
import { CloseOutlined } from "@ant-design/icons";
import { useEffect, useMemo, useState } from "react";

export default function ImageGallery({
  images,
  currentIndex,
  annotationStats,
  filterOption,
  onFilterChange,
  onSelect,
  onRemove,
  onBatchAnnotate,
  onBatchExport,
  onBatchClearAnnotations,
  onBatchDelete,
}) {
  const [selectAll, setSelectAll] = useState(false);
  const [selectedIndices, setSelectedIndices] = useState([]);

  const getStats = (idx) => annotationStats?.[idx] || { count: 0, saved: false };

  const annotatedCount = useMemo(
    () => (annotationStats || []).filter((s) => (s?.count || 0) > 0).length,
    [annotationStats]
  );

  const displayedIndices = useMemo(() => {
    if (!Array.isArray(images)) return [];
    if (filterOption === "annotated") return images.map((_, i) => i).filter((i) => (getStats(i).count || 0) > 0);
    if (filterOption === "unannotated") return images.map((_, i) => i).filter((i) => (getStats(i).count || 0) === 0);
    return images.map((_, i) => i);
  }, [images, filterOption, annotationStats]);

  const displayedItems = useMemo(() => displayedIndices.map((i) => ({ index: i, image: images[i] })), [displayedIndices, images]);

  const updateSelected = () => {
    const sel = (images || []).map((img, i) => (img?.selected ? i : -1)).filter((i) => i >= 0);
    setSelectedIndices(sel);
  };

  useEffect(() => {
    (images || []).forEach((img) => {
      if (img && typeof img.selected === "undefined") img.selected = false;
    });
    updateSelected();
  }, [images]);

  const handleSelectAll = (checked) => {
    displayedItems.forEach(({ index }) => {
      if (images[index]) images[index].selected = checked;
    });
    setSelectAll(checked);
    updateSelected();
  };

  const handleToggleOne = (index, checked) => {
    if (images[index]) images[index].selected = checked;
    updateSelected();
    setSelectAll(displayedItems.length > 0 && displayedItems.every((it) => images[it.index]?.selected));
  };

  const clickItem = (index, e) => {
    if (e?.ctrlKey) {
      const next = !images[index]?.selected;
      handleToggleOne(index, next);
      return;
    }
    onSelect?.(index);
  };

  return (
    <div style={{ padding: 10, maxHeight: 300, overflowY: "auto" }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
        <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
          <b>已上传的图片 · 共 {images.length} 张，已标注 {annotatedCount} 张</b>
          <Select
            size="small"
            style={{ width: 120 }}
            value={filterOption}
            onChange={(v) => onFilterChange?.(v)}
            options={[
              { label: "全部", value: "all" },
              { label: "已标注", value: "annotated" },
              { label: "未标注", value: "unannotated" },
            ]}
          />
        </div>

        {images.length ? (
          <Space wrap>
            <Checkbox checked={selectAll} onChange={(e) => handleSelectAll(e.target.checked)}>
              全选
            </Checkbox>
            <Button size="small" type="primary" disabled={!selectedIndices.length} onClick={() => onBatchAnnotate?.(selectedIndices)}>
              批量标注
            </Button>
            <Button size="small" type="primary" ghost disabled={!selectedIndices.length} onClick={() => onBatchExport?.(selectedIndices)}>
              批量导出
            </Button>
            <Button size="small" disabled={!selectedIndices.length} onClick={() => onBatchClearAnnotations?.(selectedIndices)}>
              批量清除标注
            </Button>
            <Button size="small" danger disabled={!selectedIndices.length} onClick={() => onBatchDelete?.(selectedIndices)}>
              批量删除
            </Button>
          </Space>
        ) : null}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", gap: 12, marginTop: 12 }}>
        {displayedItems.map(({ index, image }) => {
          const stats = getStats(index);
          const active = index === currentIndex;
          const selected = !!image?.selected;
          return (
            <div
              key={index}
              onClick={(e) => clickItem(index, e)}
              style={{
                position: "relative",
                borderRadius: 10,
                overflow: "hidden",
                cursor: "pointer",
                border: `2px solid ${active && selected ? "#faad14" : active ? "#1677ff" : selected ? "#52c41a" : "transparent"}`,
                boxShadow: active || selected ? "0 10px 20px rgba(15,23,42,0.10)" : "none",
                background: "#fff",
              }}
            >
              <div style={{ position: "absolute", top: 6, left: 6, zIndex: 2, background: "rgba(255,255,255,0.75)", borderRadius: 6, padding: "2px 4px" }}>
                <Checkbox
                  checked={selected}
                  onChange={(e) => handleToggleOne(index, e.target.checked)}
                  onClick={(e) => e.stopPropagation()}
                />
              </div>
              <img
                src={image?.thumbnail || image?.url}
                alt={image?.name}
                style={{ width: "100%", height: 120, objectFit: "cover", display: "block" }}
              />
              <div style={{ position: "absolute", left: 0, right: 0, bottom: 0, padding: 6, background: "rgba(0,0,0,0.55)", color: "#fff", display: "flex", justifyContent: "space-between", gap: 8, alignItems: "center" }}>
                <div style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontSize: 12, maxWidth: "70%" }}>
                  {image?.name}
                </div>
                {!image?.isRemote ? (
                  <Button
                    size="small"
                    danger
                    shape="circle"
                    icon={<CloseOutlined />}
                    onClick={(e) => {
                      e.stopPropagation();
                      onRemove?.(index);
                      updateSelected();
                    }}
                  />
                ) : null}
              </div>
              {stats.count > 0 ? (
                <div style={{ position: "absolute", top: 30, right: 6, zIndex: 2, display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 4 }}>
                  <Tag color="green">{stats.count}个框</Tag>
                  {stats.saved ? <Tag color="blue">已保存</Tag> : null}
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}
