"use client";

import { Button, Card, Form, Input, Space, Typography, message } from "antd";
import { FolderOpenOutlined } from "@ant-design/icons";
import axios from "axios";
import { useEffect, useState } from "react";
import { getApiUrl } from "@/lib/api";

export default function SettingsPage() {
  const [loading, setLoading] = useState(false);
  const [form] = Form.useForm();

  const fetchSettings = async () => {
    setLoading(true);
    try {
      const resp = await axios.get(`${getApiUrl()}/api/settings/`);
      const list = Array.isArray(resp.data) ? resp.data : [];
      const next = {};
      for (const item of list) {
        if (!item?.key) continue;
        next[item.key] = item.value;
      }
      form.setFieldsValue(next);
    } catch {
      message.error("获取设置失败");
    } finally {
      setLoading(false);
    }
  };

  const selectOutputFolder = async () => {
    try {
      const resp = await axios.get(`${getApiUrl()}/api/settings/select-path`);
      if (resp.data?.path) {
        form.setFieldValue("training_output_path", resp.data.path);
      }
    } catch {
      message.error("无法打开文件夹选择器");
    }
  };

  const saveSetting = async (key) => {
    try {
      const value = form.getFieldValue(key);
      await axios.post(`${getApiUrl()}/api/settings/`, {
        key,
        value,
        description: key === "training_output_path" ? "模型训练结果输出路径" : "",
      });
      message.success("设置已保存");
    } catch {
      message.error("保存设置失败");
    }
  };

  useEffect(() => {
    fetchSettings();
  }, []);

  return (
    <div className="app-page">
      <Card className="app-card">
        <Typography.Title level={3} style={{ margin: 0 }}>
          系统设置
        </Typography.Title>
        <Typography.Paragraph style={{ marginTop: 8, marginBottom: 0, color: "rgba(15,23,42,0.65)" }}>
          配置系统全局参数
        </Typography.Paragraph>
      </Card>

      <Card className="app-card" style={{ marginTop: 16 }}>
        <Form form={form} layout="vertical" disabled={loading}>
          <Form.Item
            label="模型训练输出路径"
            name="training_output_path"
            extra={<div style={{ fontSize: 12, color: "rgba(15,23,42,0.55)" }}>指定模型训练结果（权重、日志等）的默认保存目录。支持绝对路径或相对路径。</div>}
          >
            <Space.Compact style={{ width: "100%" }}>
              <Input placeholder="例如: runs/train" />
              <Button icon={<FolderOpenOutlined />} onClick={selectOutputFolder} title="选择文件夹" />
              <Button type="primary" onClick={() => saveSetting("training_output_path")}>
                保存
              </Button>
            </Space.Compact>
          </Form.Item>
        </Form>
      </Card>
    </div>
  );
}
