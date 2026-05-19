"use client";

import { Layout, Menu, Typography } from "antd";
import {
  HomeOutlined,
  PictureOutlined,
  DeploymentUnitOutlined,
  LineChartOutlined,
  SettingOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
} from "@ant-design/icons";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useMemo, useState } from "react";

const { Sider, Content, Header } = Layout;

const ROUTES = [
  { key: "/", label: "首页", icon: <HomeOutlined />, href: "/" },
  { key: "/image-annotation", label: "图像标注", icon: <PictureOutlined />, href: "/image-annotation" },
  { key: "/model-development", label: "模型开发", icon: <DeploymentUnitOutlined />, href: "/model-development" },
  { key: "/evaluation-optimization", label: "评估优化", icon: <LineChartOutlined />, href: "/evaluation-optimization" },
  { key: "/settings", label: "系统设置", icon: <SettingOutlined />, href: "/settings" },
];

function normalizePath(input) {
  const raw = String(input || "");
  if (!raw) return "/";
  const noQuery = raw.split("?")[0].split("#")[0];
  if (noQuery.length > 1 && noQuery.endsWith("/")) return noQuery.slice(0, -1);
  return noQuery;
}

export default function AppShell({ children }) {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);

  const selectedKey = useMemo(() => {
    const current = normalizePath(pathname);
    const hit = ROUTES.find((r) => r.key === current);
    return hit ? hit.key : "/";
  }, [pathname]);

  const menuItems = useMemo(
    () =>
      ROUTES.map((r) => ({
        key: r.key,
        icon: r.icon,
        label: <Link href={r.href}>{r.label}</Link>,
      })),
    []
  );

  return (
    <Layout style={{ minHeight: "100vh" }}>
      <Sider
        collapsible
        collapsed={collapsed}
        trigger={null}
        width={220}
        collapsedWidth={72}
        style={{
          position: "sticky",
          top: 0,
          height: "100vh",
          overflow: "auto",
          background: "linear-gradient(180deg, rgba(236,245,255,1) 0%, rgba(245,248,255,1) 100%)",
          borderRight: "1px solid rgba(15, 23, 42, 0.06)",
        }}
      >
        <div
          style={{
            padding: collapsed ? "18px 0" : "18px 16px",
            display: "flex",
            alignItems: "center",
            justifyContent: collapsed ? "center" : "flex-start",
            gap: 10,
          }}
        >
          <img
            src="/logo.svg"
            alt="Vision model iterate hub"
            width={collapsed ? 28 : 32}
            height={collapsed ? 28 : 32}
            style={{
              display: "block",
              borderRadius: 10,
              boxShadow: "0 10px 20px rgba(15,23,42,0.10)",
            }}
          />
          {collapsed ? null : (
            <Typography.Title level={5} style={{ margin: 0, lineHeight: 1.2 }}>
              Vision model iterate hub
            </Typography.Title>
          )}
        </div>
        <Menu
          mode="inline"
          inlineCollapsed={collapsed}
          inlineIndent={18}
          selectedKeys={[selectedKey]}
          items={menuItems}
          style={{ background: "transparent", borderInlineEnd: "none" }}
        />
        <div style={{ position: "absolute", bottom: 18, left: 0, right: 0, display: "flex", justifyContent: "center" }}>
          <div
            onClick={() => setCollapsed((v) => !v)}
            role="button"
            tabIndex={0}
            style={{
              width: 34,
              height: 34,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              borderRadius: 999,
              background: "#1677ff",
              color: "#fff",
              boxShadow: "0 8px 20px rgba(22, 119, 255, 0.25)",
              cursor: "pointer",
              userSelect: "none",
            }}
          >
            {collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
          </div>
        </div>
      </Sider>

      <Layout>
        <Header
          style={{
            background: "transparent",
            padding: "0 20px",
            display: "flex",
            alignItems: "center",
            gap: 10,
          }}
        />
        <Content style={{ padding: "0 20px 20px" }}>{children}</Content>
      </Layout>
    </Layout>
  );
}
