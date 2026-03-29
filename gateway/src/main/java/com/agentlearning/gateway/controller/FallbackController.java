package com.agentlearning.gateway.controller;

import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Mono;

import java.time.LocalDateTime;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 熔断降级 Fallback 控制器
 * 
 * 当后端服务不可用时，CircuitBreaker 会将请求转发到这里，
 * 返回友好的错误信息而不是 502/503。
 */
@RestController
@RequestMapping("/fallback")
public class FallbackController {

    @GetMapping(value = "/rag", produces = MediaType.APPLICATION_JSON_VALUE)
    public Mono<Map<String, Object>> ragFallback() {
        return buildFallback("RAG API Server", 8000,
                "RAG 知识库服务暂不可用，请检查: python project/api_server.py");
    }

    @GetMapping(value = "/a2a", produces = MediaType.APPLICATION_JSON_VALUE)
    public Mono<Map<String, Object>> a2aFallback() {
        return buildFallback("A2A Expert Agent", 5001,
                "A2A Agent 暂不可用，请检查: python project/a2a_agent.py --serve");
    }

    @GetMapping(value = "/react", produces = MediaType.APPLICATION_JSON_VALUE)
    public Mono<Map<String, Object>> reactFallback() {
        return buildFallback("ReAct Agent", 5002,
                "ReAct Agent 暂不可用，请检查: python project/react_agent.py --serve");
    }

    @GetMapping(value = "/frontend", produces = MediaType.APPLICATION_JSON_VALUE)
    public Mono<Map<String, Object>> frontendFallback() {
        return buildFallback("Frontend Dev Server", 3000,
                "前端服务暂不可用，请检查: cd frontend && npm run dev");
    }

    private Mono<Map<String, Object>> buildFallback(String service, int port, String hint) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("error", "service_unavailable");
        body.put("service", service);
        body.put("port", port);
        body.put("message", service + " 暂不可用 (熔断降级)");
        body.put("hint", hint);
        body.put("timestamp", LocalDateTime.now().toString());
        return Mono.just(body);
    }
}
