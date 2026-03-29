package com.agentlearning.gateway.security;

import com.agentlearning.gateway.config.AuthProperties;
import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.security.Keys;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.authentication.ReactiveAuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import javax.crypto.SecretKey;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * JWT 认证管理器 (Spring Security 标准接口)
 * 
 * 职责: 验证 JWT token 的签名和有效期，提取用户信息。
 * 由 SecurityConfig 注入到 AuthenticationWebFilter 中。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class JwtAuthenticationManager implements ReactiveAuthenticationManager {

    private final AuthProperties authProperties;

    @Override
    public Mono<Authentication> authenticate(Authentication authentication) {
        String token = authentication.getCredentials().toString();

        try {
            SecretKey key = Keys.hmacShaKeyFor(
                    authProperties.getSecret().getBytes(StandardCharsets.UTF_8));

            Claims claims = Jwts.parser()
                    .verifyWith(key)
                    .build()
                    .parseSignedClaims(token)
                    .getPayload();

            String userId = claims.getSubject();
            String role = claims.get("role", String.class);

            var authorities = List.of(new SimpleGrantedAuthority("ROLE_" +
                    (role != null ? role.toUpperCase() : "USER")));

            var auth = new UsernamePasswordAuthenticationToken(userId, token, authorities);
            return Mono.just(auth);

        } catch (Exception e) {
            log.debug("JWT verification failed: {}", e.getMessage());
            return Mono.empty();
        }
    }
}
