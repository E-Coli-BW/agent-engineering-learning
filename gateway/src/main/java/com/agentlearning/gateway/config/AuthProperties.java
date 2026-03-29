package com.agentlearning.gateway.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

import java.util.List;

/**
 * JWT 认证配置
 * 
 * 通过 application.yml 中 gateway.auth.* 配置
 * 默认 enabled=false，开启后所有非 public-paths 的请求都需要 JWT
 */
@Data
@Configuration
@ConfigurationProperties(prefix = "gateway.auth")
public class AuthProperties {

    /** 是否启用 JWT 认证 */
    private boolean enabled = false;

    /** JWT 签名密钥 (至少 32 字符) */
    private String secret = "change-me-in-production-at-least-32-chars!!";

    /** Token 过期时间 (小时) */
    private int expirationHours = 24;

    /** 不需要认证的路径列表 */
    private List<String> publicPaths = List.of(
            "/api/rag/health",
            "/api/a2a/health",
            "/api/react/health",
            "/auth/**",
            "/fallback/**",
            "/actuator/**"
    );
}
