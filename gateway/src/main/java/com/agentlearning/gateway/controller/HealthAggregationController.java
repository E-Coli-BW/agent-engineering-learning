package com.agentlearning.gateway.controller;

import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.time.LocalDateTime;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 聚合健康检查
 * 
 * GET /health/all — 一次性检查所有下游服务状态
 * 
 * 前端 Dashboard 可以调这一个端点就拿到全部状态，
 * 而不用分别请求 3 个后端的 /health。
 */
@Slf4j
@RestController
public class HealthAggregationController {

    private final WebClient webClient = WebClient.create();

    @GetMapping(value = "/health/all", produces = MediaType.APPLICATION_JSON_VALUE)
    public Mono<Map<String, Object>> aggregatedHealth() {
        Mono<Map<String, Object>> ragHealth = checkService("RAG API", "http://localhost:8000/health");
        Mono<Map<String, Object>> a2aHealth = checkService("A2A Expert", "http://localhost:5001/health");
        Mono<Map<String, Object>> reactHealth = checkService("ReAct Agent", "http://localhost:5002/health");

        return Mono.zip(ragHealth, a2aHealth, reactHealth)
                .map(tuple -> {
                    Map<String, Object> result = new LinkedHashMap<>();
                    result.put("gateway", "ok");
                    result.put("rag", tuple.getT1());
                    result.put("a2a", tuple.getT2());
                    result.put("react", tuple.getT3());
                    result.put("timestamp", LocalDateTime.now().toString());

                    // 总体状态
                    boolean allOk = "ok".equals(tuple.getT1().get("status"))
                            && "ok".equals(tuple.getT2().get("status"))
                            && "ok".equals(tuple.getT3().get("status"));
                    result.put("overall", allOk ? "ok" : "degraded");

                    return result;
                });
    }

    private Mono<Map<String, Object>> checkService(String name, String url) {
        return webClient.get()
                .uri(url)
                .retrieve()
                .bodyToMono(Map.class)
                .map(body -> {
                    Map<String, Object> result = new LinkedHashMap<>();
                    result.put("status", "ok");
                    result.put("name", name);
                    result.put("detail", body);
                    return result;
                })
                .timeout(Duration.ofSeconds(5))
                .onErrorResume(e -> {
                    Map<String, Object> result = new LinkedHashMap<>();
                    result.put("status", "down");
                    result.put("name", name);
                    result.put("error", e.getMessage());
                    return Mono.just(result);
                });
    }
}
