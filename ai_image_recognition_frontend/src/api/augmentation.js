"use client";

import axios from "axios";
import { getApiUrl, TIMEOUT } from "@/lib/api";

export async function runAugmentation({
  imageFiles,
  mode = "ai",
  instruction = "",
  apiKey = "",
  providerPreset = "",
  apiStyle = "",
  baseUrl = "",
  imageUrl = "",
  textModel = "",
  imageModel = "",
  classicOptions = [],
}) {
  const formData = new FormData();
  formData.append("mode", mode);
  formData.append("instruction", instruction || "");
  if (apiKey) formData.append("api_key", apiKey);
  if (providerPreset) formData.append("provider_preset", providerPreset);
  if (apiStyle) formData.append("api_style", apiStyle);
  if (baseUrl) formData.append("base_url", baseUrl);
  if (imageUrl) formData.append("image_url", imageUrl);
  if (textModel) formData.append("text_model", textModel);
  if (imageModel) formData.append("image_model", imageModel);
  if (Array.isArray(classicOptions) && classicOptions.length > 0) {
    formData.append("classic_options", JSON.stringify(classicOptions));
  }
  (imageFiles || []).forEach((file) => formData.append("images", file));

  const { data } = await axios.post(`${getApiUrl()}/api/augmentation/run`, formData, {
    timeout: Math.max(TIMEOUT, 60000),
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

