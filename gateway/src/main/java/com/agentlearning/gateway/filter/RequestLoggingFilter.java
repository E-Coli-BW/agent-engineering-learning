package com.agentlearning.gateway.filter;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import lombok.extern.slf4j.Slf4j;
import org.springframework.cloud.gateway.filter.GatewayFilterChain;
import org.springframework.cloud.gateway.filter.GlobalFilter;
import org.springframework.core.Ordered;
import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.UUID;

/**
 * 请求日志 + 追踪 ID + Prometheus 指标
 *
 * 功能:
 *   1. 为每个请求注入 X-Request-Id (全链路追踪标识)
 *   2. 记录结构化日志: method, path, status, latency, ip, requestId
 *   3. 上报 Micrometer 指标 → Prometheus 采集:
 *      - gateway_requests_total (Counter, 按 route/status 分组)
 *      - gateway_request_duration_seconds (Timer, 按 route 分组)
 *
 * Python 后端可以从 X-Request-Id header 中读取追踪 ID，
 * 写入自己的日志，实现跨服务关联。
 */
@Slf4j
@Component
public class RequestLoggingFilter implements GlobalFilter, Ordered {

    private final MeterRegistry meterRegistry;

    public RequestLoggingFilter(MeterRegistry meterRegistry) {
        this.meterRegistry = meterRegistry;
    }

    @Override
    public int getOrder() {
        return -90;
    }

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        long startTime = System.nanoTime();

        // 生成或继承请求追踪 ID
        String requestId = exchange.getRequest().getHeaders().getFirst("X-Request-Id");
        if (requestId == null || requestId.isBlank()) {
            requestId = UUID.randomUUID().toString().substring(0, 8);
        }

        String method = exchange.getRequest().getMethod().name();
        String path = exchange.getRequest().getURI().getPath();
        String clientIp = exchange.getRequest().getRemoteAddress() != null
                ? exchange.getRequest().getRemoteAddress().getAddress().getHostAddress()
                : "unknown";

        // 注入 X-Request-Id 到下游请求
        final String traceId = requestId;
        ServerHttpRequest mutatedRequest = exchange.getRequest().mutate()
                .header("X-Request-Id", traceId)
                .build();

        return chain.filter(exchange.mutate().request(mutatedRequest).build())
                .then(Mono.fromRunnable(() -> {
                    long elapsed = (System.nanoTime() - startTime) / 1_000_000; // ms
                    int status = exchange.getResponse().getStatusCode() != null
                            ? exchange.getResponse().getStatusCode().value()
                            : 0;

                    // 推导路由名
                    String route = resolveRoute(path);

                    // 结构化日志
                    log.info("[{}] {} {} → {} ({}ms) [ip={}, route={}]",
                            traceId, method, path, status, elapsed, clientIp, route);

                    // Prometheus 指标: 请求计数
                    Counter.builder("gateway_requests_total")
                            .tag("method", method)
                            .tag("route", route)
                            .tag("status", String.valueOf(status))
                            .register(meterRegistry)
                            .increment();

                    // Prometheus 指标: 请求耗时
                    Timer.builder("gateway_request_duration_seconds")
                            .tag("method", method)
                            .tag("route", route)
                            .register(meterRegistry)
                            .record(Duration.ofMillis(elapsed));

                    // 添加追踪 ID 到响应头
                    exchange.getResponse().getHeaders().add("X-Request-Id", traceId);
                }));
    }

    /**
     * 从 path 推导路由名称
     */
    private String resolveRoute(String path) {
        if (path.startsWith("/api/rag")) return "rag-api";
        if (path.startsWith("/api/a2a")) return "a2a-agent";
        if (path.startsWith("/api/react")) return "react-agent";
        if (path.startsWith("/actuator")) return "actuator";
        if (path.startsWith("/health")) return "health";
        if (path.startsWith("/auth")) return "auth";
        if (path.startsWith("/fallback")) return "fallback";
        return "frontend";
    }
}
