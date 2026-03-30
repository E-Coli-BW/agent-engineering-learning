package com.agentlearning.gateway;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.web.reactive.server.WebTestClient;

/**
 * Gateway 集成测试
 *
 * 测试: 路由配置、Fallback、健康检查、认证端点
 * 注意: 不需要后端服务运行，测试的是 Gateway 自身行为
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class GatewayApplicationTests {

    @Autowired
    private WebTestClient webTestClient;

    // ---- Actuator 健康检查 ----

    @Test
    void actuatorHealthReturns200() {
        webTestClient.get().uri("/actuator/health")
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.status").exists();
    }

    @Test
    void actuatorPrometheusReturns200() {
        webTestClient.get().uri("/actuator/prometheus")
                .exchange()
                .expectStatus().isOk()
                .expectBody(String.class)
                .value(body -> {
                    assert body.contains("jvm_memory");
                });
    }

    // ---- 认证端点 ----

    @Test
    void authStatusReturns200() {
        webTestClient.get().uri("/auth/status")
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.authEnabled").isEqualTo(false);
    }

    @Test
    void authTokenWithValidCredentials() {
        webTestClient.post().uri("/auth/token")
                .bodyValue(java.util.Map.of("username", "admin", "password", "admin"))
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.token").exists()
                .jsonPath("$.type").isEqualTo("Bearer");
    }

    @Test
    void authTokenWithInvalidCredentials() {
        webTestClient.post().uri("/auth/token")
                .bodyValue(java.util.Map.of("username", "wrong", "password", "wrong"))
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.error").isEqualTo("invalid_credentials");
    }

    // ---- Fallback 端点 ----

    @Test
    void ragFallbackReturnsJson() {
        webTestClient.get().uri("/fallback/rag")
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.error").isEqualTo("service_unavailable")
                .jsonPath("$.service").isEqualTo("RAG API Server");
    }

    @Test
    void a2aFallbackReturnsJson() {
        webTestClient.get().uri("/fallback/a2a")
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.service").isEqualTo("A2A Expert Agent");
    }

    // ---- 聚合健康检查 (后端可能未运行, 验证结构) ----

    @Test
    void healthAllReturnsStructuredResponse() {
        webTestClient.get().uri("/health/all")
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.gateway").isEqualTo("ok")
                .jsonPath("$.rag").exists()
                .jsonPath("$.a2a").exists()
                .jsonPath("$.react").exists()
                .jsonPath("$.overall").exists()
                .jsonPath("$.timestamp").exists();
    }

    @Test
    void healthAllIncludesLatency() {
        webTestClient.get().uri("/health/all")
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.rag.latency_ms").exists()
                .jsonPath("$.a2a.latency_ms").exists()
                .jsonPath("$.react.latency_ms").exists();
    }

    // ---- 响应头 (Gateway filter 只在路由转发时触发, 此处验证本地端点不报错) ----

    @Test
    void localEndpointDoesNotRequireGatewayHeaders() {
        // 非路由请求不经过 Gateway filter chain, 不会有 X-Gateway 头
        // 这是正确行为: 只有被路由到下游的请求才有 filter
        webTestClient.get().uri("/auth/status")
                .exchange()
                .expectStatus().isOk();
    }

    @Test
    void healthAllDoesNotCrash() {
        // 即使后端未运行, /health/all 也不应该 500
        webTestClient.get().uri("/health/all")
                .exchange()
                .expectStatus().isOk();
    }
}
