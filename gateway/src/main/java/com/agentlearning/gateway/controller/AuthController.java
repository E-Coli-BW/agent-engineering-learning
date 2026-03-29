package com.agentlearning.gateway.controller;

import com.agentlearning.gateway.config.AuthProperties;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.security.Keys;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

import javax.crypto.SecretKey;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 简易认证端点
 * 
 * POST /auth/token — 签发 JWT
 * GET  /auth/status — 检查认证是否启用
 * 
 * 注意: 这是演示用的简易认证，生产环境应接入 OAuth2 / OIDC。
 * 当 gateway.auth.enabled=false 时，token 端点仍可用但无实际意义。
 */
@RestController
@RequestMapping("/auth")
@RequiredArgsConstructor
public class AuthController {

    private final AuthProperties authProperties;

    /**
     * 签发 JWT Token
     * 
     * 请求体: { "username": "admin", "password": "admin" }
     * 响应:   { "token": "eyJ...", "expiresIn": 86400 }
     * 
     * 演示用: 用户名密码均为 admin/admin
     */
    @PostMapping(value = "/token", produces = MediaType.APPLICATION_JSON_VALUE)
    public Mono<Map<String, Object>> issueToken(@RequestBody Map<String, String> body) {
        String username = body.getOrDefault("username", "");
        String password = body.getOrDefault("password", "");

        // 简易验证 (生产环境应查数据库)
        if (!"admin".equals(username) || !"admin".equals(password)) {
            Map<String, Object> error = new LinkedHashMap<>();
            error.put("error", "invalid_credentials");
            error.put("message", "用户名或密码错误");
            return Mono.just(error);
        }

        SecretKey key = Keys.hmacShaKeyFor(
                authProperties.getSecret().getBytes(StandardCharsets.UTF_8));

        Instant now = Instant.now();
        Instant expiry = now.plus(authProperties.getExpirationHours(), ChronoUnit.HOURS);

        String token = Jwts.builder()
                .subject(username)
                .claim("name", username)
                .claim("role", "admin")
                .issuedAt(Date.from(now))
                .expiration(Date.from(expiry))
                .signWith(key)
                .compact();

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("token", token);
        result.put("type", "Bearer");
        result.put("expiresIn", authProperties.getExpirationHours() * 3600);
        return Mono.just(result);
    }

    /**
     * 认证状态
     */
    @GetMapping(value = "/status", produces = MediaType.APPLICATION_JSON_VALUE)
    public Mono<Map<String, Object>> status() {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("authEnabled", authProperties.isEnabled());
        result.put("publicPaths", authProperties.getPublicPaths());
        return Mono.just(result);
    }
}
