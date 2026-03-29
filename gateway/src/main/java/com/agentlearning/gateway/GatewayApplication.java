package com.agentlearning.gateway;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Agent Learning API Gateway
 * 
 * 统一入口，负责:
 *   1. 路由分发 → Python 后端 (RAG :8000, A2A :5001, ReAct :5002)
 *   2. JWT 认证 (可选，默认关闭)
 *   3. 请求限流 (基于 IP)
 *   4. 熔断降级 (后端不可用时返回友好错误)
 *   5. CORS 统一处理
 *   6. 请求日志 + 调用链追踪
 */
@SpringBootApplication
public class GatewayApplication {

    public static void main(String[] args) {
        SpringApplication.run(GatewayApplication.class, args);
    }
}
