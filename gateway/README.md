# 🚪 Java API Gateway

> Spring Cloud Gateway + Sentinel + Nacos + Spring Security — 标准微服务网关

## 架构

```
客户端 (Browser / 微信 / CLI)
          │
    ┌─────▼────────────────────────────────────┐
    │  Gateway (:8080)                          │
    │  Spring Cloud Gateway                     │
    │  ├── Spring Security (JWT 认证)            │
    │  ├── Sentinel (限流 / 熔断 / 降级)         │
    │  ├── LoadBalancer (负载均衡)               │
    │  └── RequestLoggingFilter (请求日志)       │
    └─────┬────────────────────────────────────┘
          │ (路由分发)
    ┌─────┼──────────────┐──────────────┐
    │     │              │              │
    ▼     ▼              ▼              ▼
 RAG API  A2A Agent   ReAct Agent   Frontend
 :8000    :5001       :5002         :3000
```

## 技术选型

| 能力 | 组件 | 说明 |
|------|------|------|
| 网关路由 | **Spring Cloud Gateway** | 基于 WebFlux 的响应式网关 |
| 限流/熔断/降级 | **Sentinel** | 阿里开源，支持 Dashboard 动态规则下发 |
| 服务注册/发现 | **Nacos Discovery** | 可选，默认关闭走本地配置 |
| 配置中心 | **Nacos Config** | 可选，默认关闭走本地 application.yml |
| 负载均衡 | **Spring Cloud LoadBalancer** | 替代已废弃的 Ribbon |
| 认证 | **Spring Security + JWT** | 标准 Reactive Security，非手写 filter |
| 监控 | **Spring Boot Actuator** | /health, /metrics, /gateway/routes |

## 快速开始

### 最简启动 (无需 Nacos/Sentinel Dashboard)

```bash
cd gateway
mvn clean package -DskipTests
java -jar target/gateway-1.0.0.jar
# Gateway 启动在 :8080, Sentinel/Nacos 未连接时自动降级
```

### 完整启动 (带 Sentinel Dashboard)

```bash
# 1. 启动 Sentinel Dashboard
java -jar sentinel-dashboard.jar --server.port=8858
# 访问 http://localhost:8858 (admin/sentinel)

# 2. 启动 Gateway (连接 Sentinel)
cd gateway
SENTINEL_ENABLED=true mvn spring-boot:run
```

### 完整启动 (带 Nacos)

```bash
# 1. 启动 Nacos (单机模式)
sh nacos/bin/startup.sh -m standalone
# 访问 http://localhost:8848/nacos (nacos/nacos)

# 2. 启动 Gateway (连接 Nacos + Sentinel)
NACOS_ENABLED=true SENTINEL_ENABLED=true java -jar target/gateway-1.0.0.jar
```

## 目录结构

```
gateway/
├── pom.xml
└── src/main/
    ├── java/com/agentlearning/gateway/
    │   ├── GatewayApplication.java
    │   ├── config/
    │   │   ├── GatewayConfig.java           # CORS
    │   │   ├── SecurityConfig.java          # Spring Security 配置
    │   │   ├── SentinelConfig.java          # Sentinel 限流返回格式
    │   │   └── AuthProperties.java          # JWT 配置属性
    │   ├── security/
    │   │   ├── JwtAuthenticationManager.java     # JWT 验证 (Security 标准接口)
    │   │   └── JwtServerAuthenticationConverter.java  # Token 提取
    │   ├── filter/
    │   │   └── RequestLoggingFilter.java    # 请求日志
    │   └── controller/
    │       ├── AuthController.java          # /auth/token, /auth/status
    │       ├── FallbackController.java      # 降级端点
    │       └── HealthAggregationController.java  # /health/all
    └── resources/
        ├── application.yml                  # 主配置
        └── bootstrap.yml                    # Nacos 配置中心引导
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `NACOS_ENABLED` | `false` | 是否启用 Nacos 注册/配置 |
| `NACOS_ADDR` | `localhost:8848` | Nacos 地址 |
| `SENTINEL_ENABLED` | `true` | 是否启用 Sentinel |
| `SENTINEL_DASHBOARD` | `localhost:8858` | Sentinel Dashboard 地址 |
| `AUTH_ENABLED` | `false` | 是否启用 JWT 认证 |
| `AUTH_SECRET` | (内置) | JWT 签名密钥 (≥32字符) |
| `RAG_URI` | `http://localhost:8000` | RAG 后端地址 |
| `A2A_URI` | `http://localhost:5001` | A2A 后端地址 |
| `REACT_URI` | `http://localhost:5002` | ReAct 后端地址 |
