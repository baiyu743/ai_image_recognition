"use client";

import { Alert, Button, Card, Descriptions, Empty, Form, Input, InputNumber, Modal, Progress, Radio, Select, Slider, Space, Switch, Table, Tag, Typography, message } from "antd";
import axios from "axios";
import { useEffect, useMemo, useRef, useState } from "react";
import { getApiUrl, switchToFallback } from "@/lib/api";
import { getCurrentProject } from "@/lib/projectManager";

const TRAINING_TYPE_LABEL = {
  regular: "常规训练",
  incremental: "增量训练",
  freeze_strategy: "冻结策略",
  distillation: "知识蒸馏",
};

const STATUS_LABEL = {
  pending: "等待中",
  running: "运行中",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
};

const STATUS_TAG = {
  pending: "default",
  running: "processing",
  completed: "success",
  failed: "error",
  cancelled: "default",
};

export default function ModelDevelopmentPage() {
  const [form] = Form.useForm();
  const logRef = useRef(null);

  const [datasetYamlPath, setDatasetYamlPath] = useState("");

  const [isTraining, setIsTraining] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [tasksLoading, setTasksLoading] = useState(false);
  const [logsLoading, setLogsLoading] = useState(false);

  const [currentTask, setCurrentTask] = useState(null);
  const [trainingTasks, setTrainingTasks] = useState([]);
  const [trainingLogs, setTrainingLogs] = useState([]);
  const [detailOpen, setDetailOpen] = useState(false);
  const [selectedTask, setSelectedTask] = useState(null);

  const pollingTimerRef = useRef(null);

  const trainingType = Form.useWatch("trainingType", form);

  const stopPolling = () => {
    if (pollingTimerRef.current) {
      clearInterval(pollingTimerRef.current);
      pollingTimerRef.current = null;
    }
  };

  const refreshTasks = async () => {
    setTasksLoading(true);
    try {
      const resp = await axios.get(`${getApiUrl()}/api/training/tasks`);
      setTrainingTasks(Array.isArray(resp.data) ? resp.data : []);
    } catch (e) {
      if (String(e.message || "").includes("Network Error")) {
        const switched = switchToFallback();
        if (switched) {
          message.warning("网络连接问题，已切换到备用服务器，正在重试...");
          setTimeout(() => refreshTasks(), 1000);
          return;
        }
      }
      message.error(`获取任务列表失败：${e?.response?.data?.detail || e.message}`);
    } finally {
      setTasksLoading(false);
    }
  };

  const refreshLogs = async (taskId) => {
    if (!taskId) return;
    setLogsLoading(true);
    try {
      const resp = await axios.get(`${getApiUrl()}/api/training/tasks/${taskId}/logs`);
      setTrainingLogs(resp.data?.logs || []);
      requestAnimationFrame(() => {
        if (logRef.current) {
          logRef.current.scrollTop = logRef.current.scrollHeight;
        }
      });
    } catch (e) {
      if (String(e.message || "").includes("Network Error")) {
        switchToFallback();
      }
    } finally {
      setLogsLoading(false);
    }
  };

  const startPolling = (taskId) => {
    stopPolling();
    pollingTimerRef.current = setInterval(async () => {
      try {
        const resp = await axios.get(`${getApiUrl()}/api/training/tasks/${taskId}`);
        const task = resp.data;
        setCurrentTask(task);
        await refreshLogs(taskId);
        if (["completed", "failed", "cancelled"].includes(task?.status)) {
          stopPolling();
          refreshTasks();
          if (task.status === "completed") message.success("训练完成！");
          if (task.status === "failed") message.error("训练失败！");
          if (task.status === "cancelled") message.info("训练已取消");
        }
      } catch (e) {
        if (String(e.message || "").includes("Network Error")) {
          switchToFallback();
        }
      }
    }, 2000);
  };

  const selectLocalFile = async (field, params) => {
    try {
      const resp = await axios.get(`${getApiUrl()}/api/training/select-local-file`, { params });
      if (resp.data?.path) {
        form.setFieldValue(field, resp.data.path);
        message.success(`已选择文件：${resp.data.path}`);
      }
    } catch (e) {
      message.error("无法打开文件选择对话框，请手动输入路径");
    }
  };

  const startTraining = async () => {
    try {
      const values = await form.validateFields();
      if (!values.data_path) {
        message.warning("未找到项目数据集配置文件，请先在「图像标注」中导入项目生成数据集");
        return;
      }
      setIsTraining(true);
      const endpoint = {
        regular: "/api/training/regular",
        incremental: "/api/training/incremental",
        freeze_strategy: "/api/training/freeze-strategy",
        distillation: "/api/training/distillation",
      }[values.trainingType];
      const resp = await axios.post(`${getApiUrl()}${endpoint}`, values);
      message.success("训练任务已启动");
      if (resp.data?.task_id) startPolling(resp.data.task_id);
    } catch (e) {
      if (e?.errorFields) return;
      message.error(`启动训练失败：${e?.response?.data?.detail || e.message}`);
    } finally {
      setIsTraining(false);
    }
  };

  const cancelTraining = async () => {
    if (!currentTask?.task_id || isCancelling) return;
    Modal.confirm({
      title: "确认取消",
      content: "确定要取消当前训练任务吗？",
      okText: "取消训练",
      cancelText: "返回",
      okButtonProps: { danger: true },
      async onOk() {
        setIsCancelling(true);
        try {
          await axios.post(`${getApiUrl()}/api/training/tasks/${currentTask.task_id}/cancel`);
          setCurrentTask({ ...currentTask, status: "cancelled" });
          startPolling(currentTask.task_id);
          refreshTasks();
          refreshLogs(currentTask.task_id);
        } catch {
          message.error("取消训练失败");
        } finally {
          setIsCancelling(false);
        }
      },
    });
  };

  const cancelTask = async (taskId) => {
    Modal.confirm({
      title: "确认取消",
      content: "确定要取消这个训练任务吗？",
      okText: "取消任务",
      cancelText: "返回",
      okButtonProps: { danger: true },
      async onOk() {
        try {
          await axios.post(`${getApiUrl()}/api/training/tasks/${taskId}/cancel`);
          message.success("任务已取消");
          refreshTasks();
        } catch {
          message.error("取消任务失败");
        }
      },
    });
  };

  const columns = useMemo(
    () => [
      { title: "任务ID", dataIndex: "task_id", key: "task_id", width: 220, ellipsis: true },
      {
        title: "训练类型",
        dataIndex: "training_type",
        key: "training_type",
        width: 120,
        render: (v) => TRAINING_TYPE_LABEL[v] || v,
      },
      {
        title: "状态",
        dataIndex: "status",
        key: "status",
        width: 110,
        render: (v) => <Tag color={STATUS_TAG[v]}>{STATUS_LABEL[v] || v}</Tag>,
      },
      {
        title: "进度",
        dataIndex: "progress",
        key: "progress",
        width: 160,
        render: (v) => (
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <Progress percent={Number(v || 0)} showInfo={false} size="small" style={{ width: 90 }} />
            <span style={{ color: "rgba(15,23,42,0.65)" }}>{Number(v || 0).toFixed(1)}%</span>
          </div>
        ),
      },
      {
        title: "创建时间",
        dataIndex: "created_at",
        key: "created_at",
        width: 180,
        render: (v) => (v ? new Date(v).toLocaleString("zh-CN") : "暂无"),
      },
      { title: "结果路径", dataIndex: "result_path", key: "result_path", ellipsis: true },
      {
        title: "操作",
        key: "actions",
        width: 180,
        render: (_, row) => (
          <Space>
            <Button
              size="small"
              onClick={() => {
                setSelectedTask(row);
                setDetailOpen(true);
              }}
            >
              详情
            </Button>
            {row.status === "running" ? (
              <Button size="small" danger onClick={() => cancelTask(row.task_id)}>
                取消
              </Button>
            ) : null}
          </Space>
        ),
      },
    ],
    []
  );

  useEffect(() => {
    refreshTasks();
    const proj = getCurrentProject();
    if (proj?.id) {
      form.setFieldsValue({ project_id: proj.id, name: proj.name || "" });
      axios
        .get(`${getApiUrl()}/api/projects/${proj.id}`)
        .then((resp) => {
          const datasetYaml = resp.data?.dataset_yaml_path;
          setDatasetYamlPath(datasetYaml || "");
          form.setFieldValue("data_path", datasetYaml || "");
        })
        .catch(() => {});
    }
    return () => stopPolling();
  }, []);

  return (
    <div className="app-page">
      <Card className="app-card" style={{ marginBottom: 16 }}>
        <Typography.Title level={3} style={{ margin: 0 }}>
          模型开发
        </Typography.Title>
        <Typography.Paragraph style={{ marginTop: 8, marginBottom: 0, color: "rgba(15,23,42,0.65)" }}>
          AI模型训练与管理平台，支持常规训练、增量训练、冻结策略、知识蒸馏训练
        </Typography.Paragraph>
      </Card>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <Card
          className="app-card"
          title={
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontWeight: 800 }}>训练配置</span>
              <Button type="primary" onClick={startTraining} loading={isTraining}>
                {isTraining ? "训练中..." : "开始训练"}
              </Button>
            </div>
          }
        >
          <Form
            form={form}
            layout="vertical"
            initialValues={{
              trainingType: "regular",
              task: "detect",
              model_type: "s",
              data_path: "",
              project_id: "",
              epochs: 50,
              imgsz: 640,
              batch: 8,
              patience: 15,
              use_freeze_strategy: true,
              min_epochs_per_stage: 15,
              name: "",
              base_model_path: "",
              new_classes: [],
              old_data_path: "",
              teacher_model_path: "",
              distill_temperature: 2.0,
              distill_cls_weight: 1.0,
              distill_reg_weight: 2.0,
              distill_feat_weight: 5.0,
              enable_consistency: false,
              replay_ratio: 0.2,
            }}
          >
            <Form.Item
              label="训练类型"
              name="trainingType"
              rules={[{ required: true, message: "请选择训练类型" }]}
            >
              <Radio.Group buttonStyle="solid">
                <Radio.Button value="regular">常规训练</Radio.Button>
                <Radio.Button value="incremental">增量训练</Radio.Button>
                <Radio.Button value="freeze_strategy">冻结策略</Radio.Button>
                <Radio.Button value="distillation">知识蒸馏</Radio.Button>
              </Radio.Group>
            </Form.Item>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <Form.Item label="任务类型" name="task" rules={[{ required: true, message: "请选择任务类型" }]}>
                <Select
                  options={[
                    { label: "目标检测", value: "detect" },
                    { label: "图像分割", value: "segment" },
                    { label: "图像分类", value: "classify" },
                  ]}
                />
              </Form.Item>
              <Form.Item label="模型规模" name="model_type" rules={[{ required: true, message: "请选择模型规模" }]}>
                <Select
                  options={[
                    { label: "YOLOv8n (最小)", value: "n" },
                    { label: "YOLOv8s (小型)", value: "s" },
                    { label: "YOLOv8m (中型)", value: "m" },
                    { label: "YOLOv8l (大型)", value: "l" },
                    { label: "YOLOv8x (超大)", value: "x" },
                  ]}
                />
              </Form.Item>
            </div>

            <Form.Item label="数据集来源">
              <div
                style={{
                  padding: "10px 12px",
                  borderRadius: 10,
                  border: "1px solid rgba(15,23,42,0.08)",
                  background: "#fff",
                }}
              >
                {datasetYamlPath ? (
                  <Typography.Text style={{ wordBreak: "break-all" }}>{datasetYamlPath}</Typography.Text>
                ) : (
                  <Typography.Text type="danger">未生成数据集（请先在「图像标注」中导入项目生成）</Typography.Text>
                )}
              </div>
            </Form.Item>
            <Form.Item name="data_path" hidden>
              <Input />
            </Form.Item>

            {trainingType === "incremental" ? (
              <>
                <Form.Item
                  label="基础模型"
                  name="base_model_path"
                  rules={[{ required: true, message: "请选择基础模型文件" }]}
                >
                  <Space.Compact style={{ width: "100%" }}>
                    <Input placeholder="基础模型路径 (.pt)" />
                    <Button onClick={() => selectLocalFile("base_model_path", { file_type: "model" })}>选择本地文件</Button>
                  </Space.Compact>
                </Form.Item>
                <Form.Item
                  label="旧数据配置"
                  name="old_data_path"
                  rules={[{ required: true, message: "请输入旧数据集YAML路径" }]}
                >
                  <Space.Compact style={{ width: "100%" }}>
                    <Input placeholder="旧数据集YAML路径 (必填，用于防止遗忘)" />
                    <Button onClick={() => selectLocalFile("old_data_path")}>选择本地文件</Button>
                  </Space.Compact>
                </Form.Item>
                <Form.Item label="新增类别" name="new_classes">
                  <Select mode="tags" tokenSeparators={[",", " "]} placeholder="+ 添加类别" />
                </Form.Item>
              </>
            ) : null}

            {trainingType === "distillation" ? (
              <>
                <Form.Item
                  label="教师模型"
                  name="teacher_model_path"
                  rules={[{ required: true, message: "请输入教师模型路径" }]}
                >
                  <Space.Compact style={{ width: "100%" }}>
                    <Input placeholder="教师模型权重路径 (.pt)" />
                    <Button onClick={() => selectLocalFile("teacher_model_path", { file_type: "model" })}>选择本地文件</Button>
                  </Space.Compact>
                </Form.Item>

                <Form.Item label="旧数据配置" name="old_data_path">
                  <Space.Compact style={{ width: "100%" }}>
                    <Input placeholder="旧数据集YAML路径 (用于回放)" />
                    <Button onClick={() => selectLocalFile("old_data_path")}>选择本地文件</Button>
                  </Space.Compact>
                </Form.Item>

                <Card size="small" className="app-card" style={{ marginBottom: 12 }}>
                  <Typography.Text style={{ fontWeight: 700 }}>蒸馏参数</Typography.Text>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 10 }}>
                    <Form.Item label="温度 (T)" name="distill_temperature">
                      <InputNumber min={1} step={0.5} style={{ width: "100%" }} />
                    </Form.Item>
                    <Form.Item label="分类权重" name="distill_cls_weight">
                      <InputNumber min={0} step={0.1} style={{ width: "100%" }} />
                    </Form.Item>
                    <Form.Item label="回归权重" name="distill_reg_weight">
                      <InputNumber min={0} step={0.1} style={{ width: "100%" }} />
                    </Form.Item>
                    <Form.Item label="特征权重" name="distill_feat_weight">
                      <InputNumber min={0} step={0.1} style={{ width: "100%" }} />
                    </Form.Item>
                  </div>
                </Card>

                <Card size="small" className="app-card" style={{ marginBottom: 12 }}>
                  <Typography.Text style={{ fontWeight: 700 }}>高级选项</Typography.Text>
                  <div style={{ display: "grid", gap: 10, marginTop: 10 }}>
                    <Form.Item label="一致性训练" name="enable_consistency" valuePropName="checked">
                      <Switch checkedChildren="启用强弱增强一致性" unCheckedChildren="关闭" />
                    </Form.Item>
                    <Form.Item label="旧样本回放" name="replay_ratio">
                      <Slider min={0} max={0.5} step={0.05} />
                    </Form.Item>
                  </div>
                </Card>
              </>
            ) : null}

            {trainingType === "freeze_strategy" ? (
              <>
                <Form.Item label="启用冻结策略" name="use_freeze_strategy" valuePropName="checked">
                  <Switch />
                </Form.Item>
                <Form.Item shouldUpdate noStyle>
                  {({ getFieldValue }) => {
                    const enabled = !!getFieldValue("use_freeze_strategy");
                    return enabled ? (
                      <Form.Item label="最小阶段轮数" name="min_epochs_per_stage">
                        <InputNumber min={5} max={50} step={1} style={{ width: "100%" }} />
                      </Form.Item>
                    ) : null;
                  }}
                </Form.Item>
              </>
            ) : null}

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <Form.Item label="训练轮数" name="epochs" rules={[{ required: true, message: "请输入训练轮数" }]}>
                <InputNumber min={1} max={1000} step={1} style={{ width: "100%" }} />
              </Form.Item>
              <Form.Item label="图像尺寸" name="imgsz" rules={[{ required: true, message: "请选择图像尺寸" }]}>
                <Select options={[{ label: "320", value: 320 }, { label: "640", value: 640 }, { label: "1280", value: 1280 }]} />
              </Form.Item>
              <Form.Item label="批次大小" name="batch" rules={[{ required: true, message: "请输入批次大小" }]}>
                <InputNumber min={1} max={64} step={1} style={{ width: "100%" }} />
              </Form.Item>
              <Form.Item label="早停耐心" name="patience">
                <InputNumber min={5} max={100} step={1} style={{ width: "100%" }} />
              </Form.Item>
            </div>

            <Form.Item label="项目名称" name="name">
              <Input placeholder="输入项目名称（可选）" />
            </Form.Item>
            <Form.Item name="project_id" hidden>
              <Input />
            </Form.Item>
          </Form>
        </Card>

        <div style={{ display: "grid", gap: 16 }}>
          <Card className="app-card" title="训练状态">
            {currentTask ? (
              <>
                <Descriptions column={2} bordered size="small">
                  <Descriptions.Item label="任务ID">{currentTask.task_id}</Descriptions.Item>
                  <Descriptions.Item label="训练类型">
                    {TRAINING_TYPE_LABEL[currentTask.training_type] || currentTask.training_type}
                  </Descriptions.Item>
                  <Descriptions.Item label="状态">
                    <Tag color={STATUS_TAG[currentTask.status]}>{STATUS_LABEL[currentTask.status] || currentTask.status}</Tag>
                  </Descriptions.Item>
                  <Descriptions.Item label="进度">
                    <Progress percent={Number(currentTask.progress || 0)} status={currentTask.status === "failed" ? "exception" : "active"} />
                  </Descriptions.Item>
                  <Descriptions.Item label="当前轮数">
                    {currentTask.current_epoch} / {currentTask.total_epochs}
                  </Descriptions.Item>
                  <Descriptions.Item label="开始时间">
                    {currentTask.started_at ? new Date(currentTask.started_at).toLocaleString("zh-CN") : "暂无"}
                  </Descriptions.Item>
                </Descriptions>

                {(currentTask.status === "running" || isCancelling) && (
                  <div style={{ marginTop: 14 }}>
                    <Button danger onClick={cancelTraining} loading={isCancelling} disabled={isCancelling}>
                      {isCancelling ? "取消中..." : "取消训练"}
                    </Button>
                  </div>
                )}

                {currentTask.error_message ? (
                  <div style={{ marginTop: 14 }}>
                    <Alert type="error" showIcon title="训练错误" description={currentTask.error_message} />
                  </div>
                ) : null}
              </>
            ) : (
              <Empty description="暂无训练任务" />
            )}
          </Card>

          <Card
            className="app-card"
            title={
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span>训练日志</span>
                <Button size="small" onClick={() => refreshLogs(currentTask?.task_id)} loading={logsLoading}>
                  刷新
                </Button>
              </div>
            }
          >
            <div
              ref={logRef}
              style={{
                height: 300,
                overflowY: "auto",
                background: "#f5f5f5",
                padding: 10,
                borderRadius: 10,
                fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
                fontSize: 12,
                lineHeight: 1.5,
              }}
            >
              {trainingLogs.length ? trainingLogs.map((l, i) => <div key={i}>{l}</div>) : <Empty description="暂无日志" image={Empty.PRESENTED_IMAGE_SIMPLE} />}
            </div>
          </Card>
        </div>
      </div>

      <Card
        className="app-card"
        style={{ marginTop: 16 }}
        title={
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontWeight: 800 }}>训练历史</span>
            <Button size="small" onClick={refreshTasks}>
              刷新
            </Button>
          </div>
        }
      >
        <Table rowKey={(r) => r.task_id} columns={columns} dataSource={trainingTasks} loading={tasksLoading} size="small" />
      </Card>

      <Modal
        open={detailOpen}
        onCancel={() => setDetailOpen(false)}
        title="任务详情"
        footer={null}
        width={860}
      >
        {selectedTask ? (
          <>
            <Descriptions column={2} bordered size="small">
              <Descriptions.Item label="任务ID">{selectedTask.task_id}</Descriptions.Item>
              <Descriptions.Item label="训练类型">
                {TRAINING_TYPE_LABEL[selectedTask.training_type] || selectedTask.training_type}
              </Descriptions.Item>
              <Descriptions.Item label="状态">
                <Tag color={STATUS_TAG[selectedTask.status]}>{STATUS_LABEL[selectedTask.status] || selectedTask.status}</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="进度">{Number(selectedTask.progress || 0).toFixed(1)}%</Descriptions.Item>
              <Descriptions.Item label="创建时间">
                {selectedTask.created_at ? new Date(selectedTask.created_at).toLocaleString("zh-CN") : "暂无"}
              </Descriptions.Item>
              <Descriptions.Item label="开始时间">
                {selectedTask.started_at ? new Date(selectedTask.started_at).toLocaleString("zh-CN") : "暂无"}
              </Descriptions.Item>
              <Descriptions.Item label="完成时间">
                {selectedTask.completed_at ? new Date(selectedTask.completed_at).toLocaleString("zh-CN") : "暂无"}
              </Descriptions.Item>
              <Descriptions.Item label="结果路径">{selectedTask.result_path || "暂无"}</Descriptions.Item>
            </Descriptions>

            {selectedTask.metrics && Object.keys(selectedTask.metrics).length ? (
              <div style={{ marginTop: 14 }}>
                <Typography.Title level={5} style={{ marginTop: 0 }}>
                  训练指标
                </Typography.Title>
                <Descriptions bordered size="small" column={3}>
                  {Object.entries(selectedTask.metrics).map(([k, v]) => (
                    <Descriptions.Item key={k} label={k}>
                      {typeof v === "number" ? v.toFixed(4) : String(v)}
                    </Descriptions.Item>
                  ))}
                </Descriptions>
              </div>
            ) : null}
          </>
        ) : null}
      </Modal>
    </div>
  );
}
