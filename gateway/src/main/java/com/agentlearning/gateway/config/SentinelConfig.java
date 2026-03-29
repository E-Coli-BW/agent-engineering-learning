package com.agentlearning.gateway.config;

import com.alibaba.csp.sentinel.adapter.gateway.sc.callback.BlockRequestHandler;
import com.alibaba.csp.sentinel.adapter.gateway.sc.callback.GatewayCallbackManager;
import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.server.ServerResponse;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Mono;

import java.util.Map;

/**
 * Sentinel Gateway 配置
 * 
 * 自定义限流/熔断的返回格式，替代 Sentinel 默认的纯文本 "Blocked by Sentinel"。
 * 
 * Sentinel 在 Gateway 模式下的工作方式:
 *   1. 自动识别 Spring Cloud Gateway 的路由 ID (rag-api, a2a-agent, react-agent)
 *   2. 基于路由维度进行限流/熔断
 *   3. 限流规则可通过 Sentinel Dashboard 动态下发
 *   4. 规则持久化到 Nacos (配置在 application.yml)
 * 
 * Sentinel Dashboard 使用:
 *   - 下载: https://github.com/alibaba/Sentinel/releases
 *   - 启动: java -jar sentinel-dashboard.jar --server.port=8858
 *   - 访问: http://localhost:8858 (admin/sentinel)
 *   - Gateway 启动后会自动注册
 */
@Slf4j
@Configuration
public class SentinelConfig {

    @PostConstruct
    public void init() {
        // 自定义限流返回 JSON (替代默认的纯文本)
        GatewayCallbackManager.setBlockHandler(new JsonBlockRequestHandler());
        log.info("Sentinel Gateway block handler initialized");
    }

    /**
     * 限流/熔断时返回结构化 JSON
     */
    static class JsonBlockRequestHandler implements BlockRequestHandler {

        @Override
        public Mono<ServerResponse> handleRequest(ServerWebExchange exchange, Throwable t) {
            String path = exchange.getRequest().getURI().getPath();
            log.warn("Sentinel blocked: {} — {}", path, t.getClass().getSimpleName());

            Map<String, Object> body = Map.of(
                    "error", "rate_limited",
                    "message", "请求过于频繁，请稍后重试",
                    "path", path,
                    "sentinel_rule", t.getClass().getSimpleName()
            );

            return ServerResponse.status(HttpStatus.TOO_MANY_REQUESTS)
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(body);
        }
    }
}
