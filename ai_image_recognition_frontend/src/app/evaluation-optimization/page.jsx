"use client";

import { Alert, Button, Card, Descriptions, Empty, Input, Slider, Table, Tabs, Tag, Typography, message } from "antd";
import { useEffect, useMemo, useRef, useState } from "react";
import { evaluationAPI } from "@/api/evaluation";
import { getCurrentProject } from "@/lib/projectManager";

function fmtNum(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  return n.toFixed(4);
}

export default function EvaluationOptimizationPage() {
  const [activeTab, setActiveTab] = useState("run");
  const [currentProject, setCurrentProject] = useState(null);

  const [iouThreshold, setIouThreshold] = useState(0.5);
  const [evaluationData, setEvaluationData] = useState(null);
  const [evaluationResult, setEvaluationResult] = useState(null);

  const [loadingEvalData, setLoadingEvalData] = useState(false);
  const [loadingEvaluation, setLoadingEvaluation] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);

  const [projectEvaluations, setProjectEvaluations] = useState([]);
  const [historyDetail, setHistoryDetail] = useState(null);

  const pollingTimerRef = useRef(null);
  const pollingAttemptsRef = useRef(0);
  const maxPollingAttempts = 60;

  const loadingAny = loadingEvalData || loadingEvaluation || loadingHistory;

  const canStartEvaluation = useMemo(() => {
    const preds = evaluationData?.predictions;
    const gts = evaluationData?.ground_truths;
    return !!currentProject?.id && !!evaluationData?.model_id && Array.isArray(preds) && Array.isArray(gts) && preds.length > 0 && gts.length > 0;
  }, [currentProject?.id, evaluationData]);

  const prCurveImage = useMemo(() => {
    const raw = evaluationResult?.pr_curve_image;
    if (!raw) return null;
    return `data:image/png;base64,${raw}`;
  }, [evaluationResult]);

  const statusLabel = useMemo(() => {
    const s = evaluationResult?.status;
    if (s === "pending") return "等待中";
    if (s === "running") return "运行中";
    if (s === "completed") return "已完成";
    if (s === "failed") return "失败";
    return s || "未知";
  }, [evaluationResult?.status]);

  const statusTagColor = useMemo(() => {
    const s = evaluationResult?.status;
    if (s === "completed") return "green";
    if (s === "failed") return "red";
    if (s === "running") return "orange";
    return "default";
  }, [evaluationResult?.status]);

  const classMetricRows = useMemo(() => {
    const classMetrics = evaluationResult?.metrics?.class_metrics || {};
    const rows = Object.entries(classMetrics).map(([className, m]) => ({
      className,
      precision: fmtNum(m?.precision),
      recall: fmtNum(m?.recall),
      f1: fmtNum(m?.f1_score),
      ap: fmtNum(m?.ap),
    }));
    rows.sort((a, b) => String(a.className).localeCompare(String(b.className)));
    return rows;
  }, [evaluationResult]);

  const stopPolling = () => {
    if (pollingTimerRef.current) {
      clearInterval(pollingTimerRef.current);
      pollingTimerRef.current = null;
    }
    pollingAttemptsRef.current = 0;
  };

  const pollResult = async (evaluationId, modelId) => {
    stopPolling();
    pollingTimerRef.current = setInterval(async () => {
      try {
        pollingAttemptsRef.current += 1;
        if (pollingAttemptsRef.current > maxPollingAttempts) {
          stopPolling();
          setLoadingEvaluation(false);
          message.warning("评估轮询超时，请稍后在历史记录中查看");
          return;
        }
        const resp = await evaluationAPI.getEvaluationResult(evaluationId, modelId);
        setEvaluationResult(resp.data);
        if (["completed", "failed"].includes(resp.data?.status)) {
          stopPolling();
          setLoadingEvaluation(false);
          refreshHistory();
        }
      } catch (e) {
        stopPolling();
        setLoadingEvaluation(false);
        message.error("获取评估结果失败");
      }
    }, 2000);
  };

  const loadLatestEvaluationData = async () => {
    const proj = getCurrentProject();
    setCurrentProject(proj);
    if (!proj?.id) {
      message.warning("请先在首页创建/选择项目");
      return;
    }
    setLoadingEvalData(true);
    try {
      const resp = await evaluationAPI.getLatestEvaluationData(proj.id);
      const data = resp.data?.data || null;
      setEvaluationData(data);
      if (!data) message.warning("未找到可用的 evaluation_data.json");
      else message.success("已加载最新训练产物的评估数据");
    } catch (e) {
      message.error(`加载评估数据失败：${e?.response?.data?.detail || e.message}`);
    } finally {
      setLoadingEvalData(false);
    }
  };

  const startEvaluation = async () => {
    const proj = getCurrentProject();
    setCurrentProject(proj);
    if (!canStartEvaluation) {
      const preds = evaluationData?.predictions;
      const gts = evaluationData?.ground_truths;
      if (!Array.isArray(preds) || preds.length === 0) {
        message.warning("评估数据不完整：predictions 为空");
        return;
      }
      if (!Array.isArray(gts) || gts.length === 0) {
        message.warning("评估数据不完整：ground_truths 为空（验证集标签可能缺失或为空）");
        return;
      }
      message.warning("请先加载评估数据");
      return;
    }
    setLoadingEvaluation(true);
    setEvaluationResult(null);
    try {
      const payload = {
        model_id: evaluationData.model_id,
        task_id: evaluationData.task_id || null,
        project_id: proj.id,
        iou_threshold: Number(iouThreshold || 0.5),
        predictions: evaluationData.predictions,
        ground_truths: evaluationData.ground_truths,
      };
      const resp = await evaluationAPI.startEvaluation(evaluationData.model_id, payload);
      const evaluationId = resp.data?.evaluation_id;
      if (!evaluationId) throw new Error("后端未返回 evaluation_id");
      await pollResult(evaluationId, evaluationData.model_id);
    } catch (e) {
      setLoadingEvaluation(false);
      message.error(`启动评估失败：${e?.response?.data?.detail || e.message}`);
    }
  };

  const refreshHistory = async () => {
    const proj = getCurrentProject();
    setCurrentProject(proj);
    if (!proj?.id) return;
    setLoadingHistory(true);
    try {
      const resp = await evaluationAPI.listProjectEvaluations(proj.id, 80);
      setProjectEvaluations(resp.data?.evaluations || []);
    } catch (e) {
      message.error(`刷新历史评估失败：${e?.response?.data?.detail || e.message}`);
    } finally {
      setLoadingHistory(false);
    }
  };

  const loadHistoryDetail = async (evaluationId) => {
    const proj = getCurrentProject();
    setCurrentProject(proj);
    if (!proj?.id) return;
    setLoadingHistory(true);
    try {
      const resp = await evaluationAPI.getProjectEvaluation(proj.id, evaluationId);
      setHistoryDetail(resp.data || null);
    } catch (e) {
      message.error(`加载评估详情失败：${e?.response?.data?.detail || e.message}`);
    } finally {
      setLoadingHistory(false);
    }
  };

  const refreshAll = async () => {
    setCurrentProject(getCurrentProject());
    await refreshHistory();
  };

  useEffect(() => {
    refreshAll();
    return () => stopPolling();
  }, []);

  return (
    <div className="app-page">
      <Card className="app-card" style={{ marginBottom: 14 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}>
          <div>
            <Typography.Title level={3} style={{ margin: 0 }}>
              评估优化
            </Typography.Title>
            <div style={{ marginTop: 6, color: "rgba(15,23,42,0.65)" }}>从训练产物一键评估，并沉淀评估记录用于后续优化迭代</div>
          </div>
          <Button onClick={refreshAll} loading={loadingAny}>
            刷新
          </Button>
        </div>
        <div style={{ marginTop: 10, display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ color: "rgba(15,23,42,0.65)" }}>当前项目</span>
          {currentProject?.id ? <Tag color="green">{currentProject.name} ({currentProject.id})</Tag> : <Tag>未选择</Tag>}
        </div>
        {!currentProject?.id ? (
          <Alert style={{ marginTop: 10 }} type="warning" showIcon title="未选择项目" description="请先在首页创建/选择项目，再进入评估优化。" />
        ) : null}
      </Card>

      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={[
          {
            key: "run",
            label: "一键评估",
            children: (
              <div style={{ display: "grid", gridTemplateColumns: "10fr 14fr", gap: 14 }}>
                <div style={{ display: "grid", gap: 14 }}>
                  <Card
                    className="app-card"
                    title={
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10 }}>
                        <span style={{ fontWeight: 800 }}>评估输入</span>
                        <Button type="primary" ghost onClick={loadLatestEvaluationData} disabled={!currentProject?.id} loading={loadingEvalData}>
                          从最新训练产物加载
                        </Button>
                      </div>
                    }
                  >
                    <div style={{ marginBottom: 10 }}>
                      <div style={{ marginBottom: 6, color: "rgba(15,23,42,0.65)" }}>IOU 阈值</div>
                      <Slider min={0.3} max={0.9} step={0.05} value={iouThreshold} onChange={setIouThreshold} />
                    </div>

                    {evaluationData ? (
                      <>
                        <Descriptions column={1} bordered size="small">
                          <Descriptions.Item label="model_id">{evaluationData.model_id || "未知"}</Descriptions.Item>
                          <Descriptions.Item label="task_id">{evaluationData.task_id || "未知"}</Descriptions.Item>
                          <Descriptions.Item label="predictions">{evaluationData.predictions?.length || 0}</Descriptions.Item>
                          <Descriptions.Item label="ground_truths">{evaluationData.ground_truths?.length || 0}</Descriptions.Item>
                        </Descriptions>
                        {(evaluationData.ground_truths?.length || 0) === 0 ? (
                          <Alert
                            style={{ marginTop: 10 }}
                            type="warning"
                            showIcon
                            title="ground_truths 为 0"
                            description="这会导致评估结果全部为 0。通常原因是：验证集划分到了未标注图片，或 labels/val 下的 txt 为空。请回到“图像标注 → 导入到项目”，确保验证集图片有标注。"
                          />
                        ) : null}
                      </>
                    ) : (
                      <Empty description="未加载评估数据" />
                    )}

                    <div style={{ marginTop: 12, display: "flex", justifyContent: "flex-end" }}>
                      <Button type="primary" onClick={startEvaluation} disabled={!canStartEvaluation} loading={loadingEvaluation}>
                        {loadingEvaluation ? "评估中..." : "开始评估"}
                      </Button>
                    </div>
                  </Card>

                  <Card className="app-card" title={<span style={{ fontWeight: 800 }}>评估日志</span>}>
                    {!evaluationResult ? (
                      <Empty description="暂无评估任务" />
                    ) : (
                      <div style={{ display: "grid", gap: 8 }}>
                        <div><b>evaluation_id</b>：{evaluationResult.evaluation_id}</div>
                        <div><b>status</b>：{evaluationResult.status}</div>
                        {evaluationResult.error_message ? <div style={{ color: "#cf1322" }}><b>error</b>：{evaluationResult.error_message}</div> : null}
                      </div>
                    )}
                  </Card>
                </div>

                <Card
                  className="app-card"
                  title={
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span style={{ fontWeight: 800 }}>评估结果</span>
                      {evaluationResult?.status ? <Tag color={statusTagColor}>{statusLabel}</Tag> : null}
                    </div>
                  }
                >
                  {!evaluationResult ? (
                    <Empty description="暂无评估结果" />
                  ) : (
                    <>
                      {evaluationResult.status === "failed" ? (
                        <Alert style={{ marginBottom: 10 }} type="error" showIcon title="评估失败" description={evaluationResult.error_message || "未知错误"} />
                      ) : null}

                      {evaluationResult.status === "completed" ? (
                        <>
                          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: 10 }}>
                            <Card size="small" className="app-card"><div style={{ color: "rgba(15,23,42,0.65)" }}>mAP@0.5</div><div style={{ fontSize: 18, fontWeight: 800 }}>{fmtNum(evaluationResult.metrics?.mAP50)}</div></Card>
                            <Card size="small" className="app-card"><div style={{ color: "rgba(15,23,42,0.65)" }}>mAP@0.5:0.95</div><div style={{ fontSize: 18, fontWeight: 800 }}>{fmtNum(evaluationResult.metrics?.mAP50_95)}</div></Card>
                            <Card size="small" className="app-card"><div style={{ color: "rgba(15,23,42,0.65)" }}>Precision</div><div style={{ fontSize: 18, fontWeight: 800 }}>{fmtNum(evaluationResult.metrics?.precision)}</div></Card>
                            <Card size="small" className="app-card"><div style={{ color: "rgba(15,23,42,0.65)" }}>Recall</div><div style={{ fontSize: 18, fontWeight: 800 }}>{fmtNum(evaluationResult.metrics?.recall)}</div></Card>
                          </div>

                          <div style={{ marginTop: 14 }}>
                            <Typography.Title level={5} style={{ marginBottom: 8 }}>PR 曲线</Typography.Title>
                            {prCurveImage ? (
                              <div style={{ padding: 10, borderRadius: 12, border: "1px solid rgba(15,23,42,0.06)", background: "#fff" }}>
                                <img src={prCurveImage} alt="PR Curve" style={{ width: "100%", height: "auto" }} />
                              </div>
                            ) : (
                              <Empty description="暂无 PR 曲线" />
                            )}
                          </div>

                          <div style={{ marginTop: 14 }}>
                            <Typography.Title level={5} style={{ marginBottom: 8 }}>分类别指标</Typography.Title>
                            <Table
                              size="small"
                              rowKey={(r) => r.className}
                              dataSource={classMetricRows}
                              pagination={false}
                              scroll={{ y: 240 }}
                              columns={[
                                { title: "类别", dataIndex: "className", key: "className" },
                                { title: "Precision", dataIndex: "precision", key: "precision", width: 120 },
                                { title: "Recall", dataIndex: "recall", key: "recall", width: 120 },
                                { title: "F1", dataIndex: "f1", key: "f1", width: 120 },
                                { title: "AP", dataIndex: "ap", key: "ap", width: 120 },
                              ]}
                            />
                          </div>

                          <div style={{ marginTop: 14 }}>
                            <Typography.Title level={5} style={{ marginBottom: 8 }}>优化建议</Typography.Title>
                            <Input.TextArea rows={10} readOnly value={evaluationResult.llm_analysis || ""} placeholder="暂无分析报告" />
                          </div>
                        </>
                      ) : (
                        <Empty description="评估未完成" />
                      )}
                    </>
                  )}
                </Card>
              </div>
            ),
          },
          {
            key: "history",
            label: "历史评估",
            children: (
              <Card
                className="app-card"
                title={
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontWeight: 800 }}>项目评估记录</span>
                    <Button onClick={refreshHistory} disabled={!currentProject?.id} loading={loadingHistory}>
                      刷新列表
                    </Button>
                  </div>
                }
              >
                {!currentProject?.id ? (
                  <Empty description="未选择项目" />
                ) : (
                  <>
                    <Table
                      rowKey={(r) => r.evaluation_id}
                      dataSource={projectEvaluations}
                      loading={loadingHistory}
                      size="small"
                      scroll={{ y: 340 }}
                      onRow={(row) => ({
                        onClick: () => loadHistoryDetail(row.evaluation_id),
                      })}
                      columns={[
                        { title: "evaluation_id", dataIndex: "evaluation_id", key: "evaluation_id", width: 240, ellipsis: true },
                        { title: "创建时间", dataIndex: "created_at", key: "created_at", width: 180 },
                        { title: "mAP50", key: "mAP50", width: 120, render: (_, row) => fmtNum(row.summary?.mAP50) },
                        { title: "mAP50_95", key: "mAP50_95", width: 120, render: (_, row) => fmtNum(row.summary?.mAP50_95) },
                        {
                          title: "操作",
                          key: "action",
                          width: 110,
                          render: (_, row) => (
                            <Button size="small" onClick={(e) => { e.stopPropagation(); loadHistoryDetail(row.evaluation_id); }}>
                              查看
                            </Button>
                          ),
                        },
                      ]}
                    />

                    <div style={{ marginTop: 12 }}>
                      <Typography.Title level={5} style={{ marginBottom: 8 }}>详情</Typography.Title>
                      {!historyDetail ? (
                        <Empty description="点击上方记录查看详情" />
                      ) : (
                        <>
                          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: 10, marginBottom: 10 }}>
                            <Card size="small" className="app-card"><div style={{ color: "rgba(15,23,42,0.65)" }}>mAP@0.5</div><div style={{ fontSize: 18, fontWeight: 800 }}>{fmtNum(historyDetail.metrics?.mAP50)}</div></Card>
                            <Card size="small" className="app-card"><div style={{ color: "rgba(15,23,42,0.65)" }}>mAP@0.5:0.95</div><div style={{ fontSize: 18, fontWeight: 800 }}>{fmtNum(historyDetail.metrics?.mAP50_95)}</div></Card>
                            <Card size="small" className="app-card"><div style={{ color: "rgba(15,23,42,0.65)" }}>Precision</div><div style={{ fontSize: 18, fontWeight: 800 }}>{fmtNum(historyDetail.metrics?.precision)}</div></Card>
                            <Card size="small" className="app-card"><div style={{ color: "rgba(15,23,42,0.65)" }}>Recall</div><div style={{ fontSize: 18, fontWeight: 800 }}>{fmtNum(historyDetail.metrics?.recall)}</div></Card>
                          </div>
                          <Input.TextArea rows={14} readOnly value={historyDetail.report_md || ""} placeholder="暂无报告" />
                        </>
                      )}
                    </div>
                  </>
                )}
              </Card>
            ),
          },
        ]}
      />
    </div>
  );
}
