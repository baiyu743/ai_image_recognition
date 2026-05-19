"use client";

import axios from "axios";
import { getApiUrl, TIMEOUT } from "@/lib/api";

const LONG_TIMEOUT = 10 * 60 * 1000;
const UPLOAD_TIMEOUT = 30 * 60 * 1000;

const api = axios.create({
  baseURL: getApiUrl(),
  timeout: TIMEOUT,
  headers: { "Content-Type": "application/json" },
});

api.interceptors.request.use(
  (config) => {
    config.baseURL = getApiUrl();
    return config;
  },
  (error) => Promise.reject(error)
);

export const annotateImage = async (image, tool, model, categories) => {
  const formData = new FormData();
  formData.append("image", image);
  formData.append("tool", tool);
  formData.append("model", model);
  formData.append("categories", JSON.stringify(categories || []));
  return api.post("/api/auto_annotate", formData, {
    timeout: LONG_TIMEOUT,
    headers: { "Content-Type": "multipart/form-data" },
  });
};

export const annotateProjectFile = async (projectId, payload) => {
  return api.post(`/api/projects/${projectId}/auto_annotate/file`, payload, { timeout: LONG_TIMEOUT });
};

export const uploadProjectStagingImages = async (projectId, images, { overwrite = false } = {}) => {
  const formData = new FormData();
  (images || []).forEach((file) => formData.append("images", file));
  formData.append("overwrite", overwrite ? "1" : "0");

  const base = getApiUrl();
  let preferGateway = true;
  try {
    const u = new URL(base, window.location.origin);
    const host = u.hostname;
    preferGateway = !(host === "localhost" || host === "127.0.0.1");
  } catch {
    preferGateway = true;
  }

  const directPath = `/api/projects/${projectId}/staging/images`;
  const gatewayPath = `/upload-api/projects/${projectId}/staging/images`;
  const first = preferGateway ? gatewayPath : directPath;
  const second = preferGateway ? directPath : gatewayPath;

  const postOnce = (path) =>
    api.post(path, formData, {
      timeout: UPLOAD_TIMEOUT,
      headers: { "Content-Type": "multipart/form-data" },
    });

  try {
    return await postOnce(first);
  } catch (e) {
    const status = e?.response?.status;
    if (status === 404 || status === 405) {
      return await postOnce(second);
    }
    throw e;
  }
};

export const importStagingToProjectDataset = async (projectId, payload) => {
  return api.post(`/api/projects/${projectId}/dataset/from-staging`, payload, { timeout: LONG_TIMEOUT });
};

export const getProjectDatasetState = async (projectId, options = {}) => {
  return api.get(`/api/projects/${projectId}/dataset/state`, { params: options });
};

export const saveBatchAnnotations = async (data) => {
  return api.post("/api/annotations/batch", data);
};

export const exportAnnotationData = (data, format = "json") => {
  let content;
  let mimeType;
  let extension;

  switch (format) {
    case "json":
      content = JSON.stringify(data, null, 2);
      mimeType = "application/json";
      extension = "json";
      break;
    case "coco":
      content = convertToCOCO(data);
      mimeType = "application/json";
      extension = "json";
      break;
    case "voc":
      content = convertToPascalVOC(data);
      mimeType = "application/xml";
      extension = "xml";
      break;
    case "yolo":
      content = convertToYOLO(data);
      mimeType = "text/plain";
      extension = "txt";
      break;
    case "dota":
      content = convertToDOTA(data);
      mimeType = "text/plain";
      extension = "txt";
      break;
    case "csv":
      content = convertToCSV(data);
      mimeType = "text/csv";
      extension = "csv";
      break;
    case "yaml":
    case "yml":
      content = convertToYAML(data);
      mimeType = "application/x-yaml";
      extension = "yaml";
      break;
    default:
      content = JSON.stringify(data, null, 2);
      mimeType = "application/json";
      extension = "json";
  }

  return {
    content,
    url: `data:${mimeType};charset=utf-8,${encodeURIComponent(content)}`,
    mimeType,
    extension,
  };
};

function isBoxAnnotation(ann) {
  return (ann.type === "bbox" || ann.type === "bounding_box" || ann.type === "obb") && ann.bbox;
}

function isOBB(ann) {
  return ann.type === "obb" && ann.bbox && ann.bbox.angle != null && ann.bbox.angle !== 0;
}

function percentBboxToPixel(bbox, imgWidth, imgHeight) {
  if (!imgWidth || !imgHeight) return null;
  return {
    x: (bbox.x / 100) * imgWidth,
    y: (bbox.y / 100) * imgHeight,
    width: (bbox.width / 100) * imgWidth,
    height: (bbox.height / 100) * imgHeight,
  };
}

function obbToFourCorners(bbox, imgWidth, imgHeight) {
  if (!imgWidth || !imgHeight || bbox.width <= 0 || bbox.height <= 0) return null;
  const cx = ((bbox.x + bbox.width / 2) / 100) * imgWidth;
  const cy = ((bbox.y + bbox.height / 2) / 100) * imgHeight;
  const hw = ((bbox.width / 100) * imgWidth) / 2;
  const hh = ((bbox.height / 100) * imgHeight) / 2;
  const a = bbox.angle != null ? bbox.angle : 0;
  const cos = Math.cos(a);
  const sin = Math.sin(a);
  return [
    [-hw, -hh],
    [hw, -hh],
    [hw, hh],
    [-hw, hh],
  ].map(([dx, dy]) => ({
    x: cx + dx * cos - dy * sin,
    y: cy + dx * sin + dy * cos,
  }));
}

function convertToCOCO(data) {
  const imgW = data.width || 0;
  const imgH = data.height || 0;
  const cocoFormat = {
    info: {
      description: "AI Image Recognition Annotation Dataset",
      version: "1.0",
      year: new Date().getFullYear(),
      date_created: new Date().toISOString(),
    },
    images: [
      {
        id: 1,
        file_name: data.image,
        width: imgW,
        height: imgH,
      },
    ],
    annotations: [],
    categories: [],
  };

  const categories = [...new Set((data.annotations || []).map((ann) => ann.label || "unknown"))];
  categories.forEach((category, index) => {
    cocoFormat.categories.push({ id: index + 1, name: category, supercategory: "object" });
  });

  (data.annotations || []).forEach((annotation, index) => {
    if (!isBoxAnnotation(annotation)) return;
    const categoryId = categories.indexOf(annotation.label || "unknown") + 1;
    const bbox = annotation.bbox;
    const pixel = percentBboxToPixel(bbox, imgW, imgH);
    if (!pixel) return;
    const area = pixel.width * pixel.height;

    const out = {
      id: index + 1,
      image_id: 1,
      category_id: categoryId,
      bbox: [pixel.x, pixel.y, pixel.width, pixel.height],
      area,
      iscrowd: 0,
    };

    if (isOBB(annotation)) {
      const corners = obbToFourCorners(bbox, imgW, imgH);
      if (corners) {
        out.segmentation = [
          corners.flatMap((p) => [Number(p.x.toFixed(2)), Number(p.y.toFixed(2))]),
        ];
      }
    }

    cocoFormat.annotations.push(out);
  });

  return JSON.stringify(cocoFormat, null, 2);
}

function escapeXml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function convertToPascalVOC(data) {
  const imgW = data.width || 0;
  const imgH = data.height || 0;
  const filename = data.image || "image.jpg";

  const objects = (data.annotations || [])
    .filter(isBoxAnnotation)
    .map((ann) => {
      const p = percentBboxToPixel(ann.bbox, imgW, imgH);
      if (!p) return "";
      const xmin = Math.max(0, Math.round(p.x));
      const ymin = Math.max(0, Math.round(p.y));
      const xmax = Math.max(0, Math.round(p.x + p.width));
      const ymax = Math.max(0, Math.round(p.y + p.height));
      return `
  <object>
    <name>${escapeXml(ann.label || "unknown")}</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>${xmin}</xmin>
      <ymin>${ymin}</ymin>
      <xmax>${xmax}</xmax>
      <ymax>${ymax}</ymax>
    </bndbox>
  </object>`;
    })
    .filter(Boolean)
    .join("");

  return `<?xml version="1.0" encoding="UTF-8"?>
<annotation>
  <folder>VOC</folder>
  <filename>${escapeXml(filename)}</filename>
  <size>
    <width>${imgW}</width>
    <height>${imgH}</height>
    <depth>3</depth>
  </size>
  <segmented>0</segmented>${objects}
</annotation>`;
}

function convertToYOLO(data) {
  const categories = [...new Set((data.annotations || []).filter(isBoxAnnotation).map((ann) => ann.label || "unknown"))];
  const lines = [];
  for (const ann of data.annotations || []) {
    if (!isBoxAnnotation(ann)) continue;
    const clsId = categories.indexOf(ann.label || "unknown");
    const bbox = ann.bbox;
    const cx = clamp01((bbox.x + bbox.width / 2) / 100);
    const cy = clamp01((bbox.y + bbox.height / 2) / 100);
    const bw = clamp01(bbox.width / 100);
    const bh = clamp01(bbox.height / 100);
    lines.push(`${clsId} ${cx.toFixed(6)} ${cy.toFixed(6)} ${bw.toFixed(6)} ${bh.toFixed(6)}`);
  }
  return lines.join("\n");
}

function convertToDOTA(data) {
  const imgW = data.width || 0;
  const imgH = data.height || 0;
  const lines = [];
  for (const ann of data.annotations || []) {
    if (!isBoxAnnotation(ann)) continue;
    const label = ann.label || "unknown";
    const bbox = ann.bbox;
    if (isOBB(ann)) {
      const corners = obbToFourCorners(bbox, imgW, imgH);
      if (!corners) continue;
      const nums = corners.flatMap((p) => [p.x, p.y]).map((n) => Number(n.toFixed(2)));
      lines.push(`${nums.join(" ")} ${label} 0`);
      continue;
    }
    const pixel = percentBboxToPixel(bbox, imgW, imgH);
    if (!pixel) continue;
    const x1 = pixel.x;
    const y1 = pixel.y;
    const x2 = pixel.x + pixel.width;
    const y2 = pixel.y;
    const x3 = pixel.x + pixel.width;
    const y3 = pixel.y + pixel.height;
    const x4 = pixel.x;
    const y4 = pixel.y + pixel.height;
    const nums = [x1, y1, x2, y2, x3, y3, x4, y4].map((n) => Number(n.toFixed(2)));
    lines.push(`${nums.join(" ")} ${label} 0`);
  }
  return lines.join("\n");
}

function convertToCSV(data) {
  const headers = ["image", "type", "label", "confidence", "x", "y", "width", "height", "angle", "points"];
  const rows = [headers.join(",")];
  for (const ann of data.annotations || []) {
    const bbox = ann.bbox || {};
    const points = Array.isArray(ann.points) ? JSON.stringify(ann.points) : "";
    rows.push(
      [
        csvCell(data.image || ""),
        csvCell(ann.type || ""),
        csvCell(ann.label || ""),
        csvCell(ann.confidence != null ? String(ann.confidence) : ""),
        csvCell(bbox.x != null ? String(bbox.x) : ""),
        csvCell(bbox.y != null ? String(bbox.y) : ""),
        csvCell(bbox.width != null ? String(bbox.width) : ""),
        csvCell(bbox.height != null ? String(bbox.height) : ""),
        csvCell(bbox.angle != null ? String(bbox.angle) : ""),
        csvCell(points),
      ].join(",")
    );
  }
  return rows.join("\n");
}

function csvCell(v) {
  const s = String(v ?? "");
  if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

function convertToYAML(data) {
  const lines = [];
  lines.push(`image: ${yamlScalar(data.image || "")}`);
  if (data.tool) lines.push(`tool: ${yamlScalar(data.tool)}`);
  if (data.width != null) lines.push(`width: ${Number(data.width) || 0}`);
  if (data.height != null) lines.push(`height: ${Number(data.height) || 0}`);
  lines.push("annotations:");
  for (let i = 0; i < (data.annotations || []).length; i += 1) {
    const ann = data.annotations[i] || {};
    lines.push(`  - id: ${i + 1}`);
    lines.push(`    type: ${yamlScalar(ann.type || "unknown")}`);
    lines.push(`    label: ${yamlScalar(ann.label || "unknown")}`);
    lines.push(`    confidence: ${Number.isFinite(Number(ann.confidence)) ? Number(ann.confidence) : 1.0}`);
    if (ann.bbox) {
      const bbox = ann.bbox;
      lines.push("    bbox:");
      lines.push(`      x: ${Number(bbox.x) || 0}`);
      lines.push(`      y: ${Number(bbox.y) || 0}`);
      lines.push(`      width: ${Number(bbox.width) || 0}`);
      lines.push(`      height: ${Number(bbox.height) || 0}`);
      if (bbox.angle != null) lines.push(`      angle: ${Number(bbox.angle) || 0}`);
    }
    if (Array.isArray(ann.points)) {
      lines.push("    points:");
      for (const p of ann.points) {
        if (!Array.isArray(p) || p.length < 2) continue;
        lines.push(`      - [${Number(p[0]) || 0}, ${Number(p[1]) || 0}]`);
      }
    }
  }
  return lines.join("\n");
}

function yamlScalar(v) {
  const s = String(v ?? "");
  if (!s) return '""';
  if (/^[a-zA-Z0-9_./-]+$/.test(s)) return s;
  return JSON.stringify(s);
}

function clamp01(x) {
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

