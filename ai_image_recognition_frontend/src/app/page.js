"use client";

import { Button, Card, Empty, Input, Modal, Space, Tag, Typography, message } from "antd";
import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import axios from "axios";
import { getApiUrl } from "@/lib/api";
import { clearCurrentProject, getCurrentProject, setCurrentProject } from "@/lib/projectManager";

export default function HomePage() {
  const router = useRouter();
  const projectSectionRef = useRef(null);

  const [projectName, setProjectName] = useState("");
  const [projects, setProjects] = useState([]);
  const [currentProject, setCurrentProjectState] = useState(null);
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [deletingProjectId, setDeletingProjectId] = useState("");
  const [renamingProjectId, setRenamingProjectId] = useState("");

  const syncStoredProject = (list) => {
    const stored = getCurrentProject();
    if (!stored) {
      setCurrentProjectState(null);
      return;
    }
    const matched = (list || []).find((p) => p.project_id === stored.id);
    if (!matched) {
      clearCurrentProject();
      setCurrentProjectState(null);
      return;
    }
    setCurrentProjectState({ id: matched.project_id, name: matched.name });
  };

  const loadProjects = async () => {
    setLoading(true);
    try {
      const resp = await axios.get(`${getApiUrl()}/api/projects`);
      const list = resp.data?.projects || [];
      setProjects(list);
      syncStoredProject(list);
    } catch (e) {
      setProjects([]);
      message.error(`获取项目列表失败：${e?.response?.data?.detail || e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const selectProject = (project, silent = false) => {
    const payload = { id: project.project_id, name: project.name };
    setCurrentProject(payload);
    setCurrentProjectState(payload);
    if (!silent) message.success(`已切换到项目：${project.name}`);
  };

  const requireProjectAndGo = async (path, project = null) => {
    if (project) selectProject(project, true);
    const stored = getCurrentProject();
    if (!stored?.id) {
      message.info("请先创建或选择项目");
      projectSectionRef.current?.scrollIntoView?.({ behavior: "smooth", block: "start" });
      return;
    }
    router.push(path);
  };

  const createProject = async () => {
    const name = projectName.trim();
    if (!name) {
      message.warning("请输入项目名称");
      return;
    }
    setCreating(true);
    try {
      const formData = new FormData();
      formData.append("name", name);
      const resp = await axios.post(`${getApiUrl()}/api/projects`, formData);
      selectProject(resp.data, true);
      setProjectName("");
      await loadProjects();
      message.success("项目已创建并选中");
    } catch (e) {
      message.error(`创建项目失败：${e?.response?.data?.detail || e.message}`);
    } finally {
      setCreating(false);
    }
  };

  const renameProject = async (project) => {
    const oldName = project?.name || "";
    const modal = Modal.confirm({
      title: "重命名项目",
      content: (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div style={{ color: "rgba(15,23,42,0.65)" }}>请输入新的项目名称</div>
          <Input defaultValue={oldName} maxLength={64} id="renameProjectInput" />
        </div>
      ),
      okText: "保存",
      cancelText: "取消",
      async onOk() {
        const input = document.getElementById("renameProjectInput");
        const value = String(input?.value || "").trim();
        if (!value) {
          message.warning("项目名称不能为空");
          return Promise.reject();
        }
        if (value === oldName) {
          message.warning("新名称不能与当前名称相同");
          return Promise.reject();
        }
        setRenamingProjectId(project.project_id);
        try {
          const formData = new FormData();
          formData.append("name", value);
          const resp = await axios.put(`${getApiUrl()}/api/projects/${project.project_id}`, formData);
          if (currentProject?.id === project.project_id) {
            const payload = { id: project.project_id, name: resp.data?.name };
            setCurrentProject(payload);
            setCurrentProjectState(payload);
          }
          await loadProjects();
          message.success(`项目已重命名为：${resp.data?.name}`);
        } catch (e) {
          message.error(`重命名项目失败：${e?.response?.data?.detail || e.message}`);
        } finally {
          setRenamingProjectId("");
        }
      },
    });
    return modal;
  };

  const deleteProject = async (project) => {
    Modal.confirm({
      title: "删除项目",
      okText: "删除",
      cancelText: "取消",
      okButtonProps: { danger: true },
      content: `确定删除项目「${project.name}」吗？该项目下的数据、训练结果和评估结果都会被永久删除。`,
      async onOk() {
        setDeletingProjectId(project.project_id);
        try {
          await axios.delete(`${getApiUrl()}/api/projects/${project.project_id}`);
          if (currentProject?.id === project.project_id) {
            clearCurrentProject();
            setCurrentProjectState(null);
          }
          await loadProjects();
          message.success(`已删除项目：${project.name}`);
        } catch (e) {
          message.error(`删除项目失败：${e?.response?.data?.detail || e.message}`);
        } finally {
          setDeletingProjectId("");
        }
      },
    });
  };

  const formatTime = (v) => {
    if (!v) return "暂无";
    try {
      return new Date(v).toLocaleString("zh-CN");
    } catch {
      return String(v);
    }
  };

  const featureCards = useMemo(
    () => [
      {
        title: "图像标注",
        desc: "围绕项目集中管理图片、标注数据和增广结果。",
        cta: "立即体验",
        route: "/image-annotation",
      },
      {
        title: "模型开发",
        desc: "配置训练策略、监控训练日志并沉淀模型产物。",
        cta: "立即体验",
        route: "/model-development",
      },
      {
        title: "评估优化",
        desc: "生成指标报告并基于 LLM 分析给出优化建议。",
        cta: "立即体验",
        route: "/evaluation-optimization",
      },
    ],
    []
  );

  useEffect(() => {
    setCurrentProjectState(getCurrentProject());
    loadProjects();
  }, []);

  return (
    <div className="app-page" style={{ maxWidth: 1240, margin: "0 auto" }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(0, 1.2fr) minmax(320px, 440px)",
          gap: 24,
          alignItems: "center",
          marginTop: 10,
        }}
      >
        <div>
          <Typography.Text style={{ fontWeight: 800, letterSpacing: "0.08em", color: "#0f766e" }}>
            VISION MODEL ITERATE HUB
          </Typography.Text>
          <Typography.Title style={{ marginTop: 10, marginBottom: 10, fontWeight: 900, lineHeight: 1.05 }}>
            Vision model iterate hub
          </Typography.Title>
          <Typography.Paragraph style={{ fontSize: 16, color: "rgba(15,23,42,0.78)" }}>
            集成 <b style={{ color: "#1677ff" }}>标注</b>、<b style={{ color: "#1677ff" }}>训练</b>、<b style={{ color: "#1677ff" }}>评估</b>{" "}
            与 <b style={{ color: "#1677ff" }}>持续迭代</b> 的一站式平台。
          </Typography.Paragraph>
          <Typography.Paragraph style={{ fontSize: 14, color: "rgba(15,23,42,0.65)" }}>
            推荐先创建或选择项目，再进入标注、训练和评估流程。项目级数据、训练产物和评估结果统一收口管理。
          </Typography.Paragraph>
          <Space wrap>
            <Button type="primary" size="large" onClick={() => requireProjectAndGo("/model-development")}>
              开始模型训练
            </Button>
            <Button size="large" onClick={() => requireProjectAndGo("/image-annotation")}>
              数据标注
            </Button>
          </Space>
        </div>

        <Card className="app-card" style={{ background: "linear-gradient(145deg,#f8fbff 0%,#eef5ff 52%,#f4f8fb 100%)" }}>
          <div style={{ display: "grid", gap: 12 }}>
            <Card size="small" className="app-card">
              <b>知识蒸馏</b>
              <div style={{ color: "rgba(15,23,42,0.65)" }}>Teacher → Student</div>
            </Card>
            <Card size="small" className="app-card">
              <b>LLM 评估</b>
              <div style={{ color: "rgba(15,23,42,0.65)" }}>智能分析优化建议</div>
            </Card>
            <Card size="small" className="app-card">
              <b>增量迭代</b>
              <div style={{ color: "rgba(15,23,42,0.65)" }}>持续演进训练闭环</div>
            </Card>
          </div>
        </Card>
      </div>

      <div ref={projectSectionRef} style={{ marginTop: 22 }}>
        <Card
          className="app-card"
          title={
            <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
              <div style={{ fontWeight: 800 }}>项目管理（推荐：先创建/选择项目，再按流程操作）</div>
              {currentProject?.id ? (
                <div style={{ color: "#1677ff" }}>
                  当前项目：<b>{currentProject.name}</b>
                </div>
              ) : null}
            </div>
          }
        >
          <Space wrap style={{ width: "100%" }}>
            <Input
              placeholder="输入项目名称，例如：缺陷检测_2026Q1"
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              onPressEnter={createProject}
              style={{ width: 420, maxWidth: "100%" }}
            />
            <Button type="primary" loading={creating} onClick={createProject}>
              创建项目
            </Button>
            <Button loading={loading} onClick={loadProjects}>
              刷新列表
            </Button>
            {currentProject?.id ? (
              <Button
                danger
                type="default"
                onClick={() => {
                  clearCurrentProject();
                  setCurrentProjectState(null);
                  message.success("已退出当前项目");
                }}
              >
                退出当前项目
              </Button>
            ) : null}
          </Space>

          <div style={{ marginTop: 14 }}>
            {projects.length === 0 && !loading ? (
              <Empty description="暂无项目，请先创建一个。" />
            ) : (
              <div style={{ display: "grid", gap: 10 }}>
                {projects.map((project) => (
                  <div
                    key={project.project_id}
                    style={{
                      background: "#fff",
                      borderRadius: 12,
                      padding: 14,
                      border: "1px solid rgba(15,23,42,0.06)",
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      gap: 12,
                      flexWrap: "wrap",
                    }}
                  >
                    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                      <span style={{ fontWeight: 800 }}>{project.name}</span>
                      <Space wrap size={8}>
                        <span style={{ color: "rgba(15,23,42,0.65)" }}>ID: {project.project_id}</span>
                        <span style={{ color: "rgba(15,23,42,0.65)" }}>
                          更新时间：{formatTime(project.updated_at || project.created_at)}
                        </span>
                        {project.dataset_yaml_path ? <Tag color="green">已生成数据集</Tag> : null}
                      </Space>
                    </div>

                    <Space wrap>
                      <Button type="primary" size="small" onClick={() => selectProject(project)}>
                        选择
                      </Button>
                      <Button size="small" onClick={() => requireProjectAndGo("/image-annotation", project)}>
                        进入标注
                      </Button>
                      <Button
                        size="small"
                        loading={renamingProjectId === project.project_id}
                        onClick={() => renameProject(project)}
                      >
                        重命名
                      </Button>
                      <Button
                        size="small"
                        danger
                        loading={deletingProjectId === project.project_id}
                        onClick={() => deleteProject(project)}
                      >
                        删除
                      </Button>
                    </Space>
                  </div>
                ))}
              </div>
            )}
          </div>
        </Card>
      </div>

      <div style={{ marginTop: 18 }}>
        <Typography.Title level={3} style={{ marginBottom: 12 }}>
          核心功能
        </Typography.Title>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 14 }}>
          {featureCards.map((f) => (
            <Card key={f.route} className="app-card" hoverable>
              <Typography.Title level={4} style={{ marginTop: 0 }}>
                {f.title}
              </Typography.Title>
              <Typography.Paragraph style={{ color: "rgba(15,23,42,0.65)" }}>{f.desc}</Typography.Paragraph>
              <Button type="link" onClick={() => requireProjectAndGo(f.route)}>
                {f.cta}
              </Button>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
