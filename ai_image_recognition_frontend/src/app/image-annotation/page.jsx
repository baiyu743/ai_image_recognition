"use client";

import {
  Alert,
  Button,
  Card,
  Collapse,
  Input,
  Modal,
  Radio,
  Select,
  Slider,
  Space,
  Table,
  Tag,
  Typography,
  message,
} from "antd";
import { UploadOutlined, PlusOutlined, FileTextOutlined } from "@ant-design/icons";
import { useEffect, useMemo, useRef, useState } from "react";
import JSZip from "jszip";

import AnnotationCanvas from "@/components/annotation/AnnotationCanvas";
import AnnotationList from "@/components/annotation/AnnotationList";
import ImageGallery from "@/components/annotation/ImageGallery";

import {
  annotateImage,
  annotateProjectFile,
  exportAnnotationData,
  getProjectDatasetState,
  importStagingToProjectDataset,
  saveBatchAnnotations,
  uploadProjectStagingImages,
} from "@/api/annotation";
import { runAugmentation } from "@/api/augmentation";
import { visioFirmAPI } from "@/api/visiofirm";
import { getApiUrl } from "@/lib/api";
import { getCurrentProject } from "@/lib/projectManager";
import { useAnnotationStore } from "@/store/annotationStore";

const MAX_HISTORY = 20;

function baseNameNoExt(name) {
  const n = String(name || "");
  const dot = n.lastIndexOf(".");
  return dot >= 0 ? n.slice(0, dot) : n;
}

function parseYoloTxtToAnnotations(txt, labelForId = (id) => `class_${id}`) {
  const lines = String(txt || "")
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);
  const anns = [];
  for (const line of lines) {
    const parts = line.split(/\s+/);
    if (parts.length < 5) continue;
    const cls = parts[0];
    const cx = Number(parts[1]);
    const cy = Number(parts[2]);
    const w = Number(parts[3]);
    const h = Number(parts[4]);
    if (![cx, cy, w, h].every((n) => Number.isFinite(n))) continue;
    const x = (cx - w / 2) * 100;
    const y = (cy - h / 2) * 100;
    anns.push({
      type: "bbox",
      label: labelForId(cls),
      bbox: { x, y, width: w * 100, height: h * 100 },
      confidence: 1.0,
    });
  }
  return anns;
}

function parseVocXmlToAnnotations(xmlText) {
  const parser = new DOMParser();
  const doc = parser.parseFromString(String(xmlText || ""), "application/xml");
  const errorNode = doc.querySelector("parsererror");
  if (errorNode) throw new Error("XML 解析失败");
  const sizeW = Number(doc.querySelector("annotation > size > width")?.textContent || 0);
  const sizeH = Number(doc.querySelector("annotation > size > height")?.textContent || 0);
  const objects = Array.from(doc.querySelectorAll("annotation > object"));
  const anns = [];
  for (const obj of objects) {
    const label = String(obj.querySelector("name")?.textContent || "unknown").trim() || "unknown";
    const xmin = Number(obj.querySelector("bndbox > xmin")?.textContent || 0);
    const ymin = Number(obj.querySelector("bndbox > ymin")?.textContent || 0);
    const xmax = Number(obj.querySelector("bndbox > xmax")?.textContent || 0);
    const ymax = Number(obj.querySelector("bndbox > ymax")?.textContent || 0);
    const w = Math.max(0, xmax - xmin);
    const h = Math.max(0, ymax - ymin);
    const xPct = sizeW ? (xmin / sizeW) * 100 : 0;
    const yPct = sizeH ? (ymin / sizeH) * 100 : 0;
    const wPct = sizeW ? (w / sizeW) * 100 : 0;
    const hPct = sizeH ? (h / sizeH) * 100 : 0;
    anns.push({ type: "bbox", label, bbox: { x: xPct, y: yPct, width: wPct, height: hPct }, confidence: 1.0 });
  }
  return anns;
}

function base64ToFile(b64, filename, mime = "image/png") {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
  const blob = new Blob([bytes], { type: mime });
  return new File([blob], filename, { type: mime });
}

async function dataUrlToFile(dataUrl, filename) {
  const res = await fetch(dataUrl);
  const blob = await res.blob();
  return new File([blob], filename, { type: blob.type || "application/octet-stream" });
}

async function getImageDimensions(url) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve({ width: img.naturalWidth || 0, height: img.naturalHeight || 0 });
    img.onerror = () => resolve({ width: 0, height: 0 });
    img.src = url;
  });
}

function toolToTask(tool) {
  if (tool === "image_classification") return "classification";
  if (tool === "image_segmentation") return "segmentation";
  return "detection";
}

function normalizeAiAnnotations(raw) {
  const anns = Array.isArray(raw) ? raw : [];
  return anns.map((ann) => {
    const a = { ...(ann || {}) };
    if (a.type === "rectanglelabels" && a.value) {
      return {
        type: "bbox",
        label: (a.value.rectanglelabels && a.value.rectanglelabels[0]) || "unknown",
        bbox: {
          x: parseFloat(a.value.x ?? 0),
          y: parseFloat(a.value.y ?? 0),
          width: parseFloat(a.value.width ?? 0),
          height: parseFloat(a.value.height ?? 0),
        },
        confidence: a.confidence ?? 1.0,
      };
    }

    const hasBbox =
      a.bbox &&
      (typeof a.bbox === "object"
        ? a.bbox.width != null && a.bbox.height != null
        : Array.isArray(a.bbox) && a.bbox.length >= 4);
    const hasPoints = Array.isArray(a.points) && a.points.length >= 3;

    if (!hasBbox && !hasPoints && a.class_name != null && a.confidence != null) {
      return { type: "classification", label: a.class_name, confidence: parseFloat(a.confidence) };
    }

    if (hasPoints) {
      return {
        type: "polygon",
        label: a.class_name ?? a.label ?? "unknown",
        confidence: a.confidence != null ? parseFloat(a.confidence) : 1.0,
        points: a.points,
      };
    }

    if (a.type === undefined && (a.bbox != null || a.value != null)) {
      a.type = "bbox";
    }

    if (a.bbox && typeof a.bbox === "object" && !Array.isArray(a.bbox)) {
      const bbox = {
        x: parseFloat(a.bbox.x || 0),
        y: parseFloat(a.bbox.y || 0),
        width: parseFloat(a.bbox.width || 0),
        height: parseFloat(a.bbox.height || 0),
        angle: a.bbox.angle != null ? parseFloat(a.bbox.angle || 0) : 0,
      };
      a.bbox = bbox;
    } else if (Array.isArray(a.bbox) && a.bbox.length >= 4) {
      const arr = a.bbox.map((v) => parseFloat(v || 0));
      a.bbox = { x: arr[0], y: arr[1], width: arr[2], height: arr[3] };
    }

    return a;
  });
}

export default function ImageAnnotationPage() {
  const store = useAnnotationStore();

  const uploadedImages = store.uploadedImages;
  const currentImageIndex = store.currentImageIndex;
  const annotationMode = store.annotationMode;
  const selectedTool = store.selectedTool;
  const selectedModel = store.selectedModel;
  const exportFormat = store.exportFormat;
  const galleryFilter = store.galleryFilter;

  const imageAnnotations = store.imageAnnotations;
  const savedAnnotations = store.savedAnnotations;

  const setState = store.setState;

  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelOptions, setModelOptions] = useState([]);

  const [isAutoAnnotating, setIsAutoAnnotating] = useState(false);
  const [selectedAnnotationIndex, setSelectedAnnotationIndex] = useState(-1);
  const [annotationsVisible, setAnnotationsVisible] = useState(true);

  const [historyPast, setHistoryPast] = useState([]);
  const [historyFuture, setHistoryFuture] = useState([]);

  const [importModelOpen, setImportModelOpen] = useState(false);
  const [importModelUploading, setImportModelUploading] = useState(false);
  const [importModelForm, setImportModelForm] = useState({ file: null, name: "", task: "detection" });

  const uploadInputRef = useRef(null);
  const importAnnoRef = useRef(null);
  const batchImportRef = useRef(null);

  const [datasetImportOpen, setDatasetImportOpen] = useState(false);
  const [datasetSplitMode, setDatasetSplitMode] = useState("auto");
  const [datasetValRatio, setDatasetValRatio] = useState(0.2);
  const [datasetRandomSeed, setDatasetRandomSeed] = useState(42);
  const [datasetSplits, setDatasetSplits] = useState({});
  const [datasetImportLoading, setDatasetImportLoading] = useState(false);

  const [augmentationMode, setAugmentationMode] = useState("ai");
  const [augmentationInstruction, setAugmentationInstruction] = useState("");
  const [augmentationApiKey, setAugmentationApiKey] = useState("");
  const [classicOptions, setClassicOptions] = useState([]);
  const [augmentationProviderPreset, setAugmentationProviderPreset] = useState("");
  const [augmentationApiStyle, setAugmentationApiStyle] = useState("");
  const [augmentationBaseUrl, setAugmentationBaseUrl] = useState("");
  const [augmentationImageUrl, setAugmentationImageUrl] = useState("");
  const [augmentationTextModel, setAugmentationTextModel] = useState("");
  const [augmentationImageModel, setAugmentationImageModel] = useState("");
  const [isAugmenting, setIsAugmenting] = useState(false);

  const currentImage = useMemo(() => (currentImageIndex >= 0 ? uploadedImages[currentImageIndex] : null), [uploadedImages, currentImageIndex]);
  const currentImageUrl = currentImage?.url || "";

  const currentAnnotations = useMemo(() => {
    if (currentImageIndex < 0) return [];
    if (annotationMode === "manual") {
      return savedAnnotations[currentImageIndex] || [];
    }
    return imageAnnotations[currentImageIndex] || [];
  }, [annotationMode, currentImageIndex, imageAnnotations, savedAnnotations]);

  const annotationStats = useMemo(() => {
    return uploadedImages.map((_, idx) => {
      const anns = (savedAnnotations[idx] || imageAnnotations[idx] || []);
      return { count: Array.isArray(anns) ? anns.length : 0, saved: !!savedAnnotations[idx] };
    });
  }, [uploadedImages, savedAnnotations, imageAnnotations]);

  const projectStats = useMemo(() => {
    const totalImages = uploadedImages.length;
    let annotatedCount = 0;
    let totalBoxes = 0;
    let unannotatedCount = 0;
    for (let i = 0; i < uploadedImages.length; i += 1) {
      const count = annotationStats[i]?.count || 0;
      totalBoxes += count;
      if (count > 0) annotatedCount += 1;
      else unannotatedCount += 1;
    }
    return { totalImages, annotatedCount, totalBoxes, unannotatedCount };
  }, [uploadedImages, annotationStats]);

  const goToFirstUnannotated = () => {
    const idx = annotationStats.findIndex((s) => (s?.count || 0) === 0);
    if (idx >= 0) setState({ currentImageIndex: idx });
  };

  const ensureModelsLoaded = async () => {
    setModelsLoading(true);
    try {
      const list = await visioFirmAPI.getModels();
      const task = toolToTask(selectedTool === "object_detection_obb" ? "object_detection" : selectedTool);
      const options = (list || [])
        .filter((m) => !m.task || m.task === task)
        .map((m) => ({ label: m.name || m.id, value: m.id, task: m.task, isLocal: m.isLocal, source: m.source }));
      setModelOptions(options);
    } catch (e) {
      message.error(`获取模型列表失败：${e?.message || e}`);
      setModelOptions([]);
    } finally {
      setModelsLoading(false);
    }
  };

  useEffect(() => {
    ensureModelsLoaded();
  }, []);

  const loadProjectDataset = async () => {
    const proj = getCurrentProject();
    if (!proj?.id) {
      message.warning("请先在首页创建/选择项目");
      return;
    }
    try {
      const resp = await getProjectDatasetState(proj.id, { include_thumbnails: true, thumb_size: 180, thumb_quality: 70 });
      const images = Array.isArray(resp.data?.images) ? resp.data.images : [];
      const nextImages = images.map((it) => ({
        name: it.name,
        url: `${getApiUrl()}${it.url_path}`,
        thumbnail: it.thumbnail || "",
        selected: false,
        isRemote: true,
        remoteSource: "dataset",
        remoteName: it.name,
        split: it.split,
      }));

      const nextSaved = {};
      nextImages.forEach((it, idx) => {
        const src = images[idx];
        const anns = Array.isArray(src?.annotations) ? src.annotations : [];
        nextSaved[idx] = anns;
      });

      setSelectedAnnotationIndex(-1);
      setHistoryPast([]);
      setHistoryFuture([]);
      setState({
        uploadedImages: nextImages,
        currentImageIndex: nextImages.length ? 0 : -1,
        annotationMode: "manual",
        selectedTool: "object_detection",
        savedAnnotations: nextSaved,
        imageAnnotations: {},
      });
      if (!nextImages.length) message.info("项目数据集暂无图片");
      else message.success(`已加载项目数据集：${nextImages.length} 张图片`);
    } catch (e) {
      message.error(`加载项目数据集失败：${e?.response?.data?.detail || e.message}`);
    }
  };

  useEffect(() => {
    const proj = getCurrentProject();
    if (proj?.id && uploadedImages.length === 0) {
      loadProjectDataset();
    }
  }, []);

  useEffect(() => {
    ensureModelsLoaded();
    if (annotationMode === "auto") {
      setState({ selectedModel: "" });
    }
  }, [selectedTool]);

  const pushHistory = (next) => {
    setHistoryPast((past) => {
      const out = [...past, currentAnnotations];
      if (out.length > MAX_HISTORY) out.splice(0, out.length - MAX_HISTORY);
      return out;
    });
    setHistoryFuture([]);
    if (annotationMode === "manual") {
      setState({ savedAnnotations: { ...savedAnnotations, [currentImageIndex]: next } });
    } else {
      setState({ imageAnnotations: { ...imageAnnotations, [currentImageIndex]: next } });
    }
  };

  const undoAnnotation = () => {
    if (!historyPast.length) return;
    const prev = historyPast[historyPast.length - 1];
    setHistoryPast((p) => p.slice(0, -1));
    setHistoryFuture((f) => [currentAnnotations, ...f]);
    if (annotationMode === "manual") setState({ savedAnnotations: { ...savedAnnotations, [currentImageIndex]: prev } });
    else setState({ imageAnnotations: { ...imageAnnotations, [currentImageIndex]: prev } });
  };

  const redoAnnotation = () => {
    if (!historyFuture.length) return;
    const next = historyFuture[0];
    setHistoryFuture((f) => f.slice(1));
    setHistoryPast((p) => [...p, currentAnnotations].slice(-MAX_HISTORY));
    if (annotationMode === "manual") setState({ savedAnnotations: { ...savedAnnotations, [currentImageIndex]: next } });
    else setState({ imageAnnotations: { ...imageAnnotations, [currentImageIndex]: next } });
  };

  const handleUploadImages = async (files) => {
    const list = Array.from(files || []).filter((f) => f && f.type && f.type.startsWith("image/"));
    if (!list.length) return;
    const nextImages = list.map((file) => ({
      name: file.name,
      url: URL.createObjectURL(file),
      file,
      selected: false,
      isRemote: false,
      remoteSource: "",
      remoteName: "",
      stagingStatus: "pending",
      stagingProgress: 0,
    }));
    const merged = [...uploadedImages, ...nextImages];
    const nextIndex = currentImageIndex >= 0 ? currentImageIndex : 0;
    setState({ uploadedImages: merged, currentImageIndex: nextIndex });
    message.success(`已添加 ${list.length} 张图片`);
  };

  const handleImportAnnotations = async (files) => {
    const list = Array.from(files || []);
    const importFiles = list.filter((f) => /\.(json|txt|xml)$/i.test(String(f.name || "")));
    if (!importFiles.length) {
      message.warning("请选择 .json / .txt(YOLO) / .xml(VOC) 标注文件");
      return;
    }
    for (const f of importFiles) {
      try {
        const text = await f.text();
        const lower = String(f.name || "").toLowerCase();
        let targetIndex = currentImageIndex;
        const base = baseNameNoExt(f.name);
        const matched = uploadedImages.findIndex((img) => baseNameNoExt(img?.name) === base);
        if (matched >= 0) targetIndex = matched;

        if (lower.endsWith(".json")) {
          const data = JSON.parse(text);
          if (Array.isArray(data.annotations)) {
            if (targetIndex >= 0) {
              setState({ savedAnnotations: { ...savedAnnotations, [targetIndex]: data.annotations } });
              message.success(`已导入标注：${f.name}`);
            }
          } else if (data && typeof data === "object" && !Array.isArray(data)) {
            const nextSaved = { ...savedAnnotations };
            for (const [imgName, anns] of Object.entries(data)) {
              const idx = uploadedImages.findIndex((img) => img?.name === imgName);
              if (idx >= 0 && Array.isArray(anns)) nextSaved[idx] = anns;
            }
            setState({ savedAnnotations: nextSaved });
            message.success(`已按文件名映射导入标注：${f.name}`);
          } else {
            message.warning(`不支持的 JSON 结构：${f.name}`);
          }
        } else if (lower.endsWith(".txt")) {
          if (targetIndex < 0) {
            message.warning(`未找到对应图片（按文件名匹配）：${f.name}`);
            continue;
          }
          const anns = parseYoloTxtToAnnotations(text);
          setState({ savedAnnotations: { ...savedAnnotations, [targetIndex]: anns } });
          message.success(`已导入 YOLO 标注：${f.name}`);
        } else if (lower.endsWith(".xml")) {
          if (targetIndex < 0) {
            message.warning(`未找到对应图片（按文件名匹配）：${f.name}`);
            continue;
          }
          const anns = parseVocXmlToAnnotations(text);
          setState({ savedAnnotations: { ...savedAnnotations, [targetIndex]: anns } });
          message.success(`已导入 VOC 标注：${f.name}`);
        }
      } catch (e) {
        message.error(`导入失败：${f.name}（${e?.message || e}）`);
      }
    }
  };

  const importAnnotatedImages = async (files) => {
    const list = Array.from(files || []);
    const imageFiles = list.filter((f) => f.type && f.type.startsWith("image/"));
    const annoFiles = list.filter((f) => String(f.name || "").toLowerCase().endsWith(".json"));
    await handleUploadImages(imageFiles);
    if (annoFiles.length) {
      await handleImportAnnotations(annoFiles);
    }
  };

  const selectImage = (index) => {
    if (index < 0 || index >= uploadedImages.length) return;
    setSelectedAnnotationIndex(-1);
    setHistoryPast([]);
    setHistoryFuture([]);
    setState({ currentImageIndex: index });
  };

  const removeImage = (index) => {
    if (index < 0 || index >= uploadedImages.length) return;
    const img = uploadedImages[index];
    if (img?.url?.startsWith("blob:")) {
      try { URL.revokeObjectURL(img.url); } catch {}
    }
    const nextImages = uploadedImages.filter((_, i) => i !== index);
    const nextSaved = { ...savedAnnotations };
    const nextAuto = { ...imageAnnotations };
    delete nextSaved[index];
    delete nextAuto[index];
    const shiftKeys = (obj) => {
      const out = {};
      for (const [k, v] of Object.entries(obj)) {
        const i = Number(k);
        if (!Number.isFinite(i)) continue;
        if (i < index) out[i] = v;
        else if (i > index) out[i - 1] = v;
      }
      return out;
    };
    const finalSaved = shiftKeys(nextSaved);
    const finalAuto = shiftKeys(nextAuto);
    let nextIndex = currentImageIndex;
    if (index === currentImageIndex) nextIndex = Math.min(index, nextImages.length - 1);
    else if (index < currentImageIndex) nextIndex = currentImageIndex - 1;
    setState({ uploadedImages: nextImages, savedAnnotations: finalSaved, imageAnnotations: finalAuto, currentImageIndex: nextIndex });
  };

  const autoAnnotate = async () => {
    const img = currentImage;
    if (!img) {
      message.warning("请先选择一张图片");
      return;
    }
    if (!selectedTool || !selectedModel) {
      message.warning("请先选择标注工具和模型");
      return;
    }
    setIsAutoAnnotating(true);
    try {
      message.info("正在分析图片，请稍候...");
      const proj = getCurrentProject();
      let resp = null;
      if (proj?.id) {
        if (img.remoteSource === "staging" && img.remoteName) {
          resp = await annotateProjectFile(proj.id, {
            source: "staging",
            filename: img.remoteName,
            split: img.split || null,
            tool: selectedTool,
            model: selectedModel,
            return_annotated_image: 0,
          });
        } else if (img.remoteSource === "dataset" && img.remoteName) {
          resp = await annotateProjectFile(proj.id, {
            source: "dataset",
            filename: img.remoteName,
            split: img.split || null,
            tool: selectedTool === "object_detection_obb" ? "object_detection" : selectedTool,
            model: selectedModel,
            return_annotated_image: 0,
          });
        } else if (img.file) {
          const up = await uploadProjectStagingImages(proj.id, [img.file], { overwrite: false });
          const stored = up.data?.images?.[0];
          if (stored?.name) {
            const nextImgs = [...uploadedImages];
            nextImgs[currentImageIndex] = { ...img, remoteSource: "staging", remoteName: stored.name };
            setState({ uploadedImages: nextImgs });
            resp = await annotateProjectFile(proj.id, {
              source: "staging",
              filename: stored.name,
              split: null,
              tool: selectedTool,
              model: selectedModel,
              return_annotated_image: 0,
            });
          }
        }
      }
      if (!resp && img.file) {
        resp = await annotateImage(img.file, selectedTool, selectedModel, []);
      }
      if (!resp) throw new Error("标注请求未发出（缺少项目或图片文件）");
      const processed = normalizeAiAnnotations(resp.data?.annotations || []);
      setState({ imageAnnotations: { ...imageAnnotations, [currentImageIndex]: processed } });
      message.success(`标注完成！检测到 ${processed.length} 个对象`);
    } catch (e) {
      message.error(`标注失败：${e?.response?.data?.detail || e.message}`);
    } finally {
      setIsAutoAnnotating(false);
    }
  };

  const saveCurrentAnnotations = async () => {
    if (!currentImage || currentImageIndex < 0) return;
    if (!currentAnnotations.length) {
      message.warning("当前没有可保存的标注");
      return;
    }
    try {
      await saveBatchAnnotations({ imageName: currentImage.name, tool: selectedTool, annotations: currentAnnotations });
      setState({ savedAnnotations: { ...savedAnnotations, [currentImageIndex]: currentAnnotations } });
      message.success("标注已保存");
    } catch (e) {
      message.error(`保存失败：${e?.response?.data?.detail || e.message}`);
    }
  };

  const clearAnnotations = () => {
    if (currentImageIndex < 0) return;
    if (annotationMode === "manual") setState({ savedAnnotations: { ...savedAnnotations, [currentImageIndex]: [] } });
    else setState({ imageAnnotations: { ...imageAnnotations, [currentImageIndex]: [] } });
    message.success("已清除当前图片标注");
  };

  const exportAnnotations = async () => {
    if (!currentImage || !currentAnnotations.length) {
      message.warning("没有可导出的标注结果");
      return;
    }
    const { width, height } = await getImageDimensions(currentImage.url);
    const data = { image: currentImage.name, annotations: currentAnnotations, tool: selectedTool, width, height };
    const { content, mimeType, extension } = exportAnnotationData(data, exportFormat);
    const exportName = `${String(currentImage.name).split(".")[0]}_annotations.${extension}`;

    if (typeof window.showSaveFilePicker === "function") {
      try {
        const handle = await window.showSaveFilePicker({
          suggestedName: exportName,
          types: [{ description: `${String(exportFormat).toUpperCase()} File`, accept: { [mimeType]: [`.${extension}`] } }],
        });
        const writable = await handle.createWritable();
        await writable.write(content);
        await writable.close();
        message.success(`标注结果已导出为 ${String(exportFormat).toUpperCase()} 格式`);
        return;
      } catch (e) {
        if (e?.name === "AbortError" || String(e?.message || "").includes("cancel")) return;
      }
    }
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = exportName;
    a.click();
    URL.revokeObjectURL(url);
    message.success(`标注结果已导出为 ${String(exportFormat).toUpperCase()} 格式`);
  };

  const batchExportAnnotations = async (indices) => {
    const selected = Array.isArray(indices) ? indices : Object.keys(imageAnnotations).map((k) => Number(k));
    const annotated = selected.filter((i) => (imageAnnotations[i]?.length || savedAnnotations[i]?.length || 0) > 0);
    if (!annotated.length) {
      message.warning("选中的图片中没有标注结果可导出");
      return;
    }
    const zip = new JSZip();
    if (exportFormat === "yolo") {
      const allCategories = new Set();
      annotated.forEach((i) => (savedAnnotations[i] || imageAnnotations[i] || []).forEach((a) => a.label && allCategories.add(a.label)));
      zip.file("classes.txt", Array.from(allCategories).join("\n"));
    }
    for (const i of annotated) {
      const img = uploadedImages[i];
      const anns = savedAnnotations[i] || imageAnnotations[i] || [];
      const { width, height } = await getImageDimensions(img.url);
      const data = { image: img.name, annotations: anns, tool: selectedTool, width, height };
      const { content, extension } = exportAnnotationData(data, exportFormat);
      zip.file(`${String(img.name).split(".")[0]}_annotations.${extension}`, content);
    }
    const blob = await zip.generateAsync({ type: "blob" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `annotations_${exportFormat}.zip`;
    a.click();
    URL.revokeObjectURL(url);
    message.success("批量导出完成");
  };

  const handleBatchClearAnnotations = (indices) => {
    const nextSaved = { ...savedAnnotations };
    const nextAuto = { ...imageAnnotations };
    for (const i of indices || []) {
      nextSaved[i] = [];
      nextAuto[i] = [];
    }
    setState({ savedAnnotations: nextSaved, imageAnnotations: nextAuto });
    message.success("已批量清除标注");
  };

  const handleBatchDelete = (indices) => {
    const sorted = [...(indices || [])].sort((a, b) => b - a);
    sorted.forEach((i) => removeImage(i));
    message.success("已批量删除图片");
  };

  const handleBatchAnnotate = async (indices) => {
    if (!indices?.length) return;
    if (!selectedModel || !selectedTool) {
      message.warning("请先在顶部选择标注类型与模型");
      return;
    }
    const proj = getCurrentProject();
    if (!proj?.id) {
      message.warning("请先在首页创建/选择项目");
      return;
    }
    const nextAuto = { ...imageAnnotations };
    for (const idx of indices) {
      const img = uploadedImages[idx];
      if (!img) continue;
      try {
        if (img.remoteSource === "dataset" && img.remoteName) {
          const resp = await annotateProjectFile(proj.id, {
            source: "dataset",
            filename: img.remoteName,
            split: img.split || null,
            tool: selectedTool === "object_detection_obb" ? "object_detection" : selectedTool,
            model: selectedModel,
            return_annotated_image: 0,
          });
          nextAuto[idx] = normalizeAiAnnotations(resp.data?.annotations || []);
        } else {
          let remoteName = img.remoteName;
          if (!(img.remoteSource === "staging" && remoteName) && img.file) {
            const up = await uploadProjectStagingImages(proj.id, [img.file], { overwrite: false });
            const stored = up.data?.images?.[0];
            remoteName = stored?.name || "";
            const nextImgs = [...uploadedImages];
            nextImgs[idx] = { ...img, remoteSource: "staging", remoteName };
            setState({ uploadedImages: nextImgs });
          }
          if (!remoteName) continue;
          const resp = await annotateProjectFile(proj.id, {
            source: "staging",
            filename: remoteName,
            split: null,
            tool: selectedTool === "object_detection_obb" ? "object_detection" : selectedTool,
            model: selectedModel,
            return_annotated_image: 0,
          });
          nextAuto[idx] = normalizeAiAnnotations(resp.data?.annotations || []);
        }
      } catch {}
    }
    setState({ imageAnnotations: nextAuto });
    message.success("批量标注完成（部分失败会被跳过）");
  };

  const openDatasetImportDialog = () => {
    const proj = getCurrentProject();
    if (!proj?.id) {
      message.warning("请先在首页创建/选择项目");
      return;
    }
    const splits = {};
    uploadedImages.forEach((_, idx) => {
      splits[String(idx)] = splits[String(idx)] || "train";
    });
    setDatasetSplits(splits);
    setDatasetImportOpen(true);
  };

  const datasetImportItems = useMemo(() => {
    return uploadedImages
      .map((img, index) => ({ img, index }))
      .filter(({ img }) => !!img?.file || (img?.remoteSource === "staging" && !!img?.remoteName))
      .map(({ img, index }) => {
        const uploadName = String(img.remoteName || img.name);
        const anns = savedAnnotations[index] || imageAnnotations[index] || [];
        return { index, splitKey: String(index), name: img.name, url: img.url, uploadName, annotationCount: Array.isArray(anns) ? anns.length : 0 };
      });
  }, [uploadedImages, savedAnnotations, imageAnnotations]);

  const datasetSplitStats = useMemo(() => {
    const out = { train: 0, val: 0 };
    datasetImportItems.forEach((it) => {
      const split = datasetSplits[it.splitKey] || "train";
      if (split === "val") out.val += 1;
      else out.train += 1;
    });
    return out;
  }, [datasetImportItems, datasetSplits]);

  const applyAutoSplit = () => {
    const n = datasetImportItems.length;
    if (!n) return;
    const rng = (() => {
      let t = Number(datasetRandomSeed) >>> 0;
      return () => {
        t += 0x6d2b79f5;
        let x = t;
        x = Math.imul(x ^ (x >>> 15), x | 1);
        x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
        return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
      };
    })();
    const idxs = datasetImportItems.map((it) => it.index);
    const shuffled = [...idxs];
    for (let i = shuffled.length - 1; i > 0; i -= 1) {
      const j = Math.floor(rng() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    const nVal = Math.max(0, Math.min(n, Math.round(n * Number(datasetValRatio || 0.2))));
    const next = {};
    idxs.forEach((i) => (next[String(i)] = "train"));
    for (let k = 0; k < nVal; k += 1) next[String(shuffled[k])] = "val";
    setDatasetSplits(next);
  };

  useEffect(() => {
    if (datasetSplitMode === "auto") applyAutoSplit();
  }, [datasetSplitMode, datasetValRatio, datasetRandomSeed, datasetImportOpen]);

  const confirmImportToProject = async () => {
    const proj = getCurrentProject();
    if (!proj?.id) {
      message.warning("请先在首页创建/选择项目");
      return;
    }
    if (!datasetImportItems.length) {
      message.warning("请先上传图片或上传到暂存区后再导入");
      return;
    }
    setDatasetImportLoading(true);
    try {
      const nextImgs = [...uploadedImages];
      for (const item of datasetImportItems) {
        const img = nextImgs[item.index];
        if (!img) continue;
        if (!(img.remoteSource === "staging" && img.remoteName) && img.file) {
          const up = await uploadProjectStagingImages(proj.id, [img.file], { overwrite: false });
          const stored = up.data?.images?.[0];
          if (stored?.name) nextImgs[item.index] = { ...img, remoteSource: "staging", remoteName: stored.name };
        }
      }
      setState({ uploadedImages: nextImgs });

      const staged = datasetImportItems
        .map((it) => ({ ...it, img: nextImgs[it.index] }))
        .filter(({ img }) => img?.remoteSource === "staging" && img?.remoteName);

      const categorySet = new Set();
      const chunkSize = 200;
      let totalImages = 0;
      let totalBboxes = 0;
      for (let i = 0; i < staged.length; i += chunkSize) {
        const chunk = staged.slice(i, i + chunkSize);
        const items = [];
        const annotationsByFilename = {};
        for (const { img, index } of chunk) {
          const filename = img.remoteName;
          const anns = savedAnnotations[index] || imageAnnotations[index] || [];
          const safe = Array.isArray(anns) ? anns : [];
          annotationsByFilename[filename] = safe;
          safe.forEach((a) => {
            const label = String(a?.label || "").trim();
            if (label) categorySet.add(label);
          });
          const split = (datasetSplits[String(index)] || "train") === "val" ? "val" : "train";
          items.push({ filename, split });
        }
        const resp = await importStagingToProjectDataset(proj.id, {
          items,
          annotations_by_filename: annotationsByFilename,
          categories: Array.from(categorySet),
          move: true,
        });
        totalImages += Number(resp.data?.image_count || 0);
        totalBboxes += Number(resp.data?.bbox_count || 0);
      }
      message.success(`导入成功：${totalImages} 张图片，${totalBboxes} 个标注`);
      setDatasetImportOpen(false);
      await loadProjectDataset();
    } catch (e) {
      message.error(`导入到项目失败：${e?.response?.data?.detail || e.message}`);
    } finally {
      setDatasetImportLoading(false);
    }
  };

  const runAugment = async () => {
    const selectedIdx = uploadedImages.map((img, idx) => (img?.selected ? idx : -1)).filter((i) => i >= 0);
    if (!selectedIdx.length) {
      message.warning("请先在图片库中勾选要增广的图片");
      return;
    }
    setIsAugmenting(true);
    try {
      const files = [];
      for (const idx of selectedIdx) {
        const img = uploadedImages[idx];
        if (!img) continue;
        if (img.file) files.push(img.file);
        else if (img.url) {
          try {
            files.push(await dataUrlToFile(img.url, img.name || `aug_${idx}.png`));
          } catch {}
        }
      }
      if (!files.length) {
        message.warning("未找到可增广的图片文件（请选择本地上传的图片，或确保远端图片可正常加载）");
        return;
      }
      const resp = await runAugmentation({
        imageFiles: files,
        mode: augmentationMode,
        instruction: augmentationInstruction,
        apiKey: augmentationApiKey,
        providerPreset: augmentationProviderPreset,
        apiStyle: augmentationApiStyle,
        baseUrl: augmentationBaseUrl,
        imageUrl: augmentationImageUrl,
        textModel: augmentationTextModel,
        imageModel: augmentationImageModel,
        classicOptions,
      });
      const augmented = Array.isArray(resp?.augmented) ? resp.augmented : [];
      const newImages = [];
      for (const item of augmented) {
        if (!item?.image_base64 || !item?.filename) continue;
        const file = base64ToFile(item.image_base64, item.filename);
        newImages.push({ name: item.filename, url: `data:${file.type};base64,${item.image_base64}`, file, selected: false });
      }
      if (newImages.length) {
        setState({ uploadedImages: [...uploadedImages, ...newImages] });
        message.success(`增广完成：新增 ${newImages.length} 张图片`);
      } else {
        message.warning("增广返回为空或全部失败");
      }
    } catch (e) {
      message.error(`增广失败：${e?.response?.data?.detail || e.message}`);
    } finally {
      setIsAugmenting(false);
    }
  };

  const previousImage = () => {
    if (currentImageIndex > 0) selectImage(currentImageIndex - 1);
  };
  const nextImage = () => {
    if (currentImageIndex < uploadedImages.length - 1) selectImage(currentImageIndex + 1);
  };

  useEffect(() => {
    const onKeyDown = (e) => {
      const tag = String(e?.target?.tagName || "").toLowerCase();
      const editing =
        tag === "input" ||
        tag === "textarea" ||
        e?.target?.isContentEditable ||
        String(e?.target?.getAttribute?.("role") || "").toLowerCase() === "textbox";
      if (editing) return;

      if (e.key === "a" || e.key === "ArrowLeft") previousImage();
      if (e.key === "d" || e.key === "ArrowRight") nextImage();
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z" && !e.shiftKey) undoAnnotation();
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z" && e.shiftKey) redoAnnotation();
      if (e.key === "Escape") setSelectedAnnotationIndex(-1);
      if ((e.key === "Delete" || e.key === "Backspace") && selectedAnnotationIndex >= 0) {
        const list = [...currentAnnotations];
        if (selectedAnnotationIndex < list.length) {
          list.splice(selectedAnnotationIndex, 1);
          setSelectedAnnotationIndex((i) => Math.min(i, list.length - 1));
          setAnnotationsForCurrent(list);
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [currentImageIndex, uploadedImages, historyPast, historyFuture, currentAnnotations, selectedAnnotationIndex]);

  const submitImportModel = async () => {
    if (!importModelForm.file) {
      message.warning("请选择模型文件");
      return;
    }
    if (!String(importModelForm.name || "").trim()) {
      message.warning("请填写模型名称");
      return;
    }
    setImportModelUploading(true);
    try {
      const res = await visioFirmAPI.uploadModel(importModelForm.file, importModelForm.name.trim(), importModelForm.task);
      if (res?.model?.id) {
        message.success("模型导入成功");
        setImportModelOpen(false);
        await ensureModelsLoaded();
        setState({ selectedModel: res.model.id });
      }
    } catch (e) {
      message.error(`导入失败：${e?.response?.data?.detail || e.message}`);
    } finally {
      setImportModelUploading(false);
    }
  };

  const setAnnotationsForCurrent = (next) => {
    if (currentImageIndex < 0) return;
    pushHistory(next);
  };

  return (
    <div className="app-page">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12, flexWrap: "wrap" }}>
        <div>
          <Typography.Title level={3} style={{ marginTop: 0, marginBottom: 6 }}>
            图像标注
          </Typography.Title>
          {uploadedImages.length ? (
            <div style={{ color: "rgba(15,23,42,0.65)" }}>
              已标注 <b>{projectStats.annotatedCount}/{projectStats.totalImages}</b> 张 · 共 <b>{projectStats.totalBoxes}</b> 个目标 ·{" "}
              {projectStats.unannotatedCount > 0 ? (
                <span style={{ color: "#1677ff", cursor: "pointer" }} onClick={goToFirstUnannotated}>
                  未标注 {projectStats.unannotatedCount} 张
                </span>
              ) : (
                <span>全部已标注</span>
              )}
            </div>
          ) : null}
        </div>
      </div>

      <Card className="app-card" style={{ marginTop: 12 }}>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
          <input ref={uploadInputRef} type="file" accept="image/*" multiple style={{ display: "none" }} onChange={(e) => handleUploadImages(e.target.files)} />
          <Button type="primary" icon={<PlusOutlined />} onClick={() => uploadInputRef.current?.click()}>
            上传图片
          </Button>

          <input ref={importAnnoRef} type="file" accept=".json,.txt,.xml" multiple style={{ display: "none" }} onChange={(e) => handleImportAnnotations(e.target.files)} />
          <Button icon={<FileTextOutlined />} onClick={() => importAnnoRef.current?.click()}>
            导入标注
          </Button>

          <input ref={batchImportRef} type="file" accept="image/*,.json,.txt,.xml" multiple style={{ display: "none" }} onChange={(e) => importAnnotatedImages(e.target.files)} />
          <Button type="default" icon={<UploadOutlined />} onClick={() => batchImportRef.current?.click()}>
            批量导入（图片+标注）
          </Button>

          <Button type="primary" ghost onClick={openDatasetImportDialog} disabled={!uploadedImages.length || !getCurrentProject()?.id}>
            导入到项目
          </Button>
        </div>

        <div style={{ marginTop: 14, display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
          <span style={{ fontWeight: 700 }}>标注模式：</span>
          <Radio.Group value={annotationMode} onChange={(e) => setState({ annotationMode: e.target.value })}>
            <Radio.Button value="manual">手动标注</Radio.Button>
            <Radio.Button value="auto">自动标注</Radio.Button>
          </Radio.Group>
        </div>

        {annotationMode === "auto" ? (
          <div style={{ marginTop: 14, display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
            <Select
              value={selectedTool}
              onChange={(v) => setState({ selectedTool: v })}
              style={{ width: 200 }}
              options={[
                { label: "目标检测", value: "object_detection" },
                { label: "图像分类", value: "image_classification" },
                { label: "图像分割", value: "image_segmentation" },
              ]}
            />
            <Select
              value={selectedModel || undefined}
              onPopupVisibleChange={(open) => open && ensureModelsLoaded()}
              onChange={(v) => setState({ selectedModel: v })}
              style={{ width: 260 }}
              placeholder="选择模型"
              loading={modelsLoading}
              options={modelOptions}
            />
            <Button onClick={() => { setImportModelForm({ file: null, name: "", task: toolToTask(selectedTool) }); setImportModelOpen(true); }}>
              导入模型
            </Button>
            <Button type="primary" onClick={autoAnnotate} disabled={!currentImage || !selectedTool || !selectedModel} loading={isAutoAnnotating}>
              AI自动标注
            </Button>
          </div>
        ) : (
          <div style={{ marginTop: 14, display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
            <Select
              value={selectedTool}
              onChange={(v) => setState({ selectedTool: v })}
              style={{ width: 200 }}
              options={[
                { label: "矩形框", value: "object_detection" },
                { label: "旋转框(OBB)", value: "object_detection_obb" },
              ]}
            />
            <Button onClick={undoAnnotation} disabled={!historyPast.length}>
              撤销
            </Button>
            <Button onClick={redoAnnotation} disabled={!historyFuture.length}>
              重做
            </Button>
            <Button type="primary" onClick={saveCurrentAnnotations} disabled={!currentAnnotations.length}>
              保存当前标注
            </Button>
          </div>
        )}

        <div style={{ marginTop: 14 }}>
          <Collapse
            items={[
              {
                key: "tips",
                label: "标注工具使用说明",
                children: (
                  <div style={{ color: "rgba(15,23,42,0.75)", lineHeight: 1.8 }}>
                    <b>矩形框</b>
                    <ul>
                      <li>选择「矩形框」后，在图片上左键拖拽绘制；松开后输入标签。</li>
                      <li>右键点击框可删除或编辑标签。</li>
                    </ul>
                    <b>旋转框(OBB)</b>
                    <ul>
                      <li>选择「旋转框(OBB)」后，左键拖拽绘制初始矩形，松开后输入标签。</li>
                      <li>绿色圆点用于旋转（右键菜单支持删除/编辑）。</li>
                    </ul>
                    <b>画布操作</b>
                    <ul>
                      <li>Ctrl + 滚轮：缩放画布；空格 + 左键拖拽：平移画布。</li>
                      <li>右上角按钮：+ / − / 1:1 调节缩放。</li>
                    </ul>
                    <b>快捷键</b>
                    <ul>
                      <li>Ctrl+Z：撤销；Ctrl+Shift+Z：重做（仅当前图片，最多约 20 步）。</li>
                      <li>A / D 或 ← / →：上一张 / 下一张图片。</li>
                    </ul>
                  </div>
                ),
              },
              {
                key: "aug",
                label: "智能体数据增广",
                children: (
                  <div style={{ display: "grid", gap: 10 }}>
                    <div style={{ color: "rgba(15,23,42,0.65)" }}>
                      在上方图片库中勾选要增广的图片。AI 模式会使用你自己填写的 API Key；常规模式不依赖模型配置，也能直接做数据增广。
                    </div>
                    <Radio.Group value={augmentationMode} onChange={(e) => setAugmentationMode(e.target.value)}>
                      <Radio.Button value="ai">AI 模型增广</Radio.Button>
                      <Radio.Button value="classic">常规工具增广</Radio.Button>
                    </Radio.Group>
                    {augmentationMode === "ai" ? (
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                        <div>
                          <div style={{ marginBottom: 6 }}>模型平台</div>
                          <Select
                            value={augmentationProviderPreset || undefined}
                            onChange={setAugmentationProviderPreset}
                            placeholder="选择模型平台（可留空）"
                            options={[
                              { label: "通用 / 自定义", value: "custom" },
                              { label: "DashScope / 通义千问", value: "dashscope" },
                              { label: "OpenAI 兼容", value: "openai" },
                            ]}
                          />
                        </div>
                        <div>
                          <div style={{ marginBottom: 6 }}>API Key</div>
                          <Input.Password value={augmentationApiKey} onChange={(e) => setAugmentationApiKey(e.target.value)} placeholder="请输入你自己的 API Key" />
                        </div>
                        <div>
                          <div style={{ marginBottom: 6 }}>图像模型标识</div>
                          <Input value={augmentationImageModel} onChange={(e) => setAugmentationImageModel(e.target.value)} placeholder="model name / model id / deployment name" />
                        </div>
                        <div>
                          <div style={{ marginBottom: 6 }}>文本模型标识（可选）</div>
                          <Input value={augmentationTextModel} onChange={(e) => setAugmentationTextModel(e.target.value)} placeholder="可留空" />
                        </div>
                        <div>
                          <div style={{ marginBottom: 6 }}>接口类型（可选）</div>
                          <Select
                            value={augmentationApiStyle || undefined}
                            onChange={setAugmentationApiStyle}
                            placeholder="不知道怎么选时留空"
                            options={[
                              { label: "自动/默认", value: "" },
                              { label: "DashScope", value: "dashscope" },
                              { label: "OpenAI Images", value: "openai" },
                            ]}
                          />
                        </div>
                        <div>
                          <div style={{ marginBottom: 6 }}>文本接口 Base URL（可选）</div>
                          <Input value={augmentationBaseUrl} onChange={(e) => setAugmentationBaseUrl(e.target.value)} placeholder="例如：https://your-provider.example/v1" />
                        </div>
                        <div>
                          <div style={{ marginBottom: 6 }}>图像接口 URL（可选）</div>
                          <Input value={augmentationImageUrl} onChange={(e) => setAugmentationImageUrl(e.target.value)} placeholder="可留空；特殊平台才需要" />
                        </div>
                      </div>
                    ) : (
                      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                        {[
                          { value: "brightness_up", label: "亮度增强" },
                          { value: "contrast_up", label: "对比度增强" },
                          { value: "noise_light", label: "轻微噪声" },
                          { value: "blur_light", label: "轻微模糊" },
                          { value: "flip_horizontal", label: "水平翻转" },
                          { value: "rotate_90", label: "旋转90°" },
                        ].map((opt) => (
                          <Tag.CheckableTag
                            key={opt.value}
                            checked={classicOptions.includes(opt.value)}
                            onChange={(checked) =>
                              setClassicOptions((prev) =>
                                checked ? [...prev, opt.value] : prev.filter((v) => v !== opt.value)
                              )
                            }
                          >
                            {opt.label}
                          </Tag.CheckableTag>
                        ))}
                      </div>
                    )}

                    <div>
                      <div style={{ marginBottom: 6 }}>增广需求</div>
                      <Input.TextArea value={augmentationInstruction} onChange={(e) => setAugmentationInstruction(e.target.value)} rows={3} placeholder="例如：增加光照变化、添加轻微高斯噪声、水平翻转、逆时针旋转90度、提高对比度" />
                      <Space wrap style={{ marginTop: 8 }}>
                        <Button size="small" onClick={() => setAugmentationInstruction("增加亮度，亮度倍数约1.2")}>光照增强</Button>
                        <Button size="small" onClick={() => setAugmentationInstruction("添加轻微高斯噪声，噪声标准差0.02")}>添加噪声</Button>
                        <Button size="small" onClick={() => setAugmentationInstruction("水平翻转")}>水平翻转</Button>
                        <Button size="small" onClick={() => setAugmentationInstruction("逆时针旋转90度")}>旋转90°</Button>
                        <Button size="small" onClick={() => setAugmentationInstruction("提高对比度，对比度倍数1.3")}>提高对比度</Button>
                        <Button size="small" onClick={() => setAugmentationInstruction("轻微高斯模糊，核大小3")}>轻微模糊</Button>
                      </Space>
                    </div>

                    <Button type="primary" loading={isAugmenting} onClick={runAugment} disabled={!uploadedImages.length}>
                      {isAugmenting ? "增广中…" : augmentationMode === "ai" ? "执行 AI 增广" : "执行常规增广"}
                    </Button>
                  </div>
                ),
              },
            ]}
          />
        </div>

        <div style={{ marginTop: 14, display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
          <Button onClick={clearAnnotations} disabled={!currentAnnotations.length}>
            清除标注
          </Button>
          <Select
            value={exportFormat}
            onChange={(v) => setState({ exportFormat: v })}
            style={{ width: 160 }}
            options={[
              { label: "JSON", value: "json" },
              { label: "COCO", value: "coco" },
              { label: "Pascal VOC", value: "voc" },
              { label: "YOLO", value: "yolo" },
              { label: "DOTA (OBB)", value: "dota" },
              { label: "CSV", value: "csv" },
              { label: "YAML", value: "yaml" },
            ]}
          />
          <Button type="primary" onClick={exportAnnotations} disabled={!currentAnnotations.length}>
            导出标注结果
          </Button>
        </div>
      </Card>

      <div style={{ marginTop: 12, display: "grid", gap: 12 }}>
        {uploadedImages.length ? (
          <Card className="app-card" size="small">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
              <div>
                <Button type="link" disabled={currentImageIndex <= 0} onClick={previousImage}>
                  上一张
                </Button>
                <span style={{ color: "rgba(15,23,42,0.65)" }}>
                  {currentImageIndex + 1} / {uploadedImages.length}
                </span>
                <Button type="link" disabled={currentImageIndex >= uploadedImages.length - 1} onClick={nextImage}>
                  下一张
                </Button>
              </div>
              <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                {uploadedImages.slice(0, 6).map((img, idx) => (
                  <div key={idx} onClick={() => selectImage(idx)} style={{ width: 44, height: 44, borderRadius: 10, overflow: "hidden", border: idx === currentImageIndex ? "2px solid #1677ff" : "1px solid rgba(15,23,42,0.10)", cursor: "pointer" }}>
                    <img src={img.url} alt={img.name} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                  </div>
                ))}
                {uploadedImages.length > 6 ? <Tag>+{uploadedImages.length - 6}</Tag> : null}
              </div>
            </div>
          </Card>
        ) : null}

        {uploadedImages.length ? (
          <Card className="app-card" size="small" title="图片库">
            <ImageGallery
              images={uploadedImages}
              currentIndex={currentImageIndex}
              annotationStats={annotationStats}
              filterOption={galleryFilter}
              onFilterChange={(v) => setState({ galleryFilter: v })}
              onSelect={selectImage}
              onRemove={removeImage}
              onBatchAnnotate={handleBatchAnnotate}
              onBatchExport={batchExportAnnotations}
              onBatchClearAnnotations={handleBatchClearAnnotations}
              onBatchDelete={handleBatchDelete}
            />
          </Card>
        ) : (
          <Card className="app-card" style={{ textAlign: "center" }}>
            <div style={{ padding: 30, color: "rgba(15,23,42,0.65)" }}>暂无上传的图片</div>
          </Card>
        )}

        <div style={{ display: "grid", gridTemplateColumns: "1fr 420px", gap: 12, alignItems: "start" }}>
          <Card className="app-card" title={currentImage ? currentImage.name : "未选择图片"}>
            {!uploadedImages.length ? (
              <Alert type="info" title="请上传图片开始使用AI标注" showIcon />
            ) : !currentImageUrl ? (
              <Alert type="info" title="请选择一张图片进行标注" showIcon />
            ) : (
              <AnnotationCanvas
                imageUrl={currentImageUrl}
                imageName={currentImage?.name}
                annotations={currentAnnotations}
                selectedTool={selectedTool === "object_detection_obb" ? "object_detection_obb" : selectedTool}
                selectedAnnotationIndex={selectedAnnotationIndex}
                onSelectAnnotationIndex={setSelectedAnnotationIndex}
                onChangeAnnotations={setAnnotationsForCurrent}
              />
            )}
            {currentImage ? (
              <div style={{ marginTop: 10, display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                <Tag color="green">标注数：{currentAnnotations.length}</Tag>
                {annotationMode === "manual" && savedAnnotations[currentImageIndex] ? <Tag color="blue">✓ 已保存</Tag> : null}
              </div>
            ) : null}
          </Card>

          <div>
            {currentAnnotations.length ? (
              <AnnotationList
                annotations={currentAnnotations}
                initialVisible={annotationsVisible}
                selectedIndex={selectedAnnotationIndex}
                onSelect={setSelectedAnnotationIndex}
              />
            ) : (
              <Card className="app-card" size="small">
                <div style={{ color: "rgba(15,23,42,0.65)" }}>当前图片暂无标注，绘制框后可在此查看标注信息</div>
              </Card>
            )}
          </div>
        </div>
      </div>

      <Modal
        open={importModelOpen}
        title="导入自己的模型"
        onCancel={() => setImportModelOpen(false)}
        onOk={submitImportModel}
        okText="导入"
        cancelText="取消"
        confirmLoading={importModelUploading}
        destroyOnHidden
      >
        <div style={{ display: "grid", gap: 12 }}>
          <div>
            <div style={{ marginBottom: 6 }}>模型文件（.pt / .pth / .onnx）</div>
            <input type="file" accept=".pt,.pth,.onnx" onChange={(e) => setImportModelForm((s) => ({ ...s, file: e.target.files?.[0] || null }))} />
          </div>
          <div>
            <div style={{ marginBottom: 6 }}>模型名称</div>
            <Input value={importModelForm.name} onChange={(e) => setImportModelForm((s) => ({ ...s, name: e.target.value }))} maxLength={64} />
          </div>
          <div>
            <div style={{ marginBottom: 6 }}>任务类型</div>
            <Select
              value={importModelForm.task}
              onChange={(v) => setImportModelForm((s) => ({ ...s, task: v }))}
              options={[
                { label: "目标检测", value: "detection" },
                { label: "图像分割", value: "segmentation" },
                { label: "图像分类", value: "classification" },
              ]}
            />
          </div>
        </div>
      </Modal>

      <Modal
        open={datasetImportOpen}
        title="将当前标注导入到项目数据集"
        onCancel={() => setDatasetImportOpen(false)}
        onOk={confirmImportToProject}
        okText="开始导入"
        cancelText="取消"
        width={960}
        confirmLoading={datasetImportLoading}
        destroyOnHidden
      >
        {!getCurrentProject()?.id ? (
          <Alert type="warning" showIcon title="未选择项目" description="请先在首页创建/选择项目，然后再导入数据集。" style={{ marginBottom: 10 }} />
        ) : null}

        <div style={{ display: "grid", gap: 12 }}>
          <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
            <span style={{ fontWeight: 700 }}>划分方式</span>
            <Radio.Group value={datasetSplitMode} onChange={(e) => setDatasetSplitMode(e.target.value)} disabled={!getCurrentProject()?.id}>
              <Radio value="auto">自动划分</Radio>
              <Radio value="manual">手动选择</Radio>
            </Radio.Group>
          </div>

          {datasetSplitMode === "auto" ? (
            <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
              <div style={{ minWidth: 280 }}>
                <div style={{ marginBottom: 6 }}>验证集比例</div>
                <Slider min={0.05} max={0.5} step={0.05} value={datasetValRatio} onChange={setDatasetValRatio} />
              </div>
              <div style={{ minWidth: 220 }}>
                <div style={{ marginBottom: 6 }}>随机种子</div>
                <Input value={datasetRandomSeed} onChange={(e) => setDatasetRandomSeed(e.target.value)} />
              </div>
              <Button onClick={applyAutoSplit} disabled={!datasetImportItems.length}>重新划分</Button>
            </div>
          ) : null}

          <div style={{ color: "rgba(15,23,42,0.65)" }}>
            训练集 {datasetSplitStats.train} 张 · 验证集 {datasetSplitStats.val} 张
          </div>

          <Table
            size="small"
            rowKey={(r) => r.index}
            dataSource={datasetImportItems}
            pagination={false}
            scroll={{ y: 360 }}
            columns={[
              { title: "#", width: 60, render: (_, row) => row.index + 1 },
              {
                title: "图片",
                dataIndex: "name",
                render: (_, row) => (
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <img src={row.url} style={{ width: 44, height: 44, objectFit: "cover", borderRadius: 8, border: "1px solid rgba(15,23,42,0.10)" }} />
                    <div style={{ display: "flex", flexDirection: "column" }}>
                      <b>{row.name}</b>
                      <span style={{ color: "rgba(15,23,42,0.65)", fontSize: 12 }}>标注 {row.annotationCount} 个</span>
                    </div>
                  </div>
                ),
              },
              {
                title: "划分",
                width: 160,
                render: (_, row) => (
                  <Select
                    size="small"
                    value={datasetSplits[row.splitKey] || "train"}
                    onChange={(v) => setDatasetSplits((s) => ({ ...s, [row.splitKey]: v }))}
                    style={{ width: 120 }}
                    disabled={!getCurrentProject()?.id || datasetSplitMode === "auto"}
                    options={[
                      { label: "训练集", value: "train" },
                      { label: "验证集", value: "val" },
                    ]}
                  />
                ),
              },
              { title: "导入文件名", dataIndex: "uploadName", render: (v) => <Tag>{v}</Tag> },
            ]}
          />
        </div>
      </Modal>
    </div>
  );
}
