#!/usr/bin/env bash
# ============================================================
# 一键启动 Agent Learning 全栈服务
# ============================================================
#
# 启动:
#   ./start.sh          — 启动全部 (RAG API + A2A + ReAct + 前端)
#   ./start.sh front    — 只启动前端
#   ./start.sh back     — 只启动后端 (三个服务)
#   ./start.sh rag      — 只启动 RAG API Server
#   ./start.sh a2a      — 只启动 A2A Expert Agent
#   ./start.sh react    — 只启动 ReAct Agent
#
# 停止:
#   ./start.sh stop     — 停止所有服务
#
# 状态:
#   ./start.sh status   — 查看各服务状态
#
# ============================================================

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"
LOG_DIR="$ROOT_DIR/data/logs"
mkdir -p "$LOG_DIR"

# ---- 颜色 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ---- PID 文件 ----
PID_RAG="$LOG_DIR/rag_api.pid"
PID_A2A="$LOG_DIR/a2a_agent.pid"
PID_REACT="$LOG_DIR/react_agent.pid"
PID_FRONT="$LOG_DIR/frontend.pid"
PID_GATEWAY="$LOG_DIR/gateway.pid"
PID_ORCHESTRATOR="$LOG_DIR/orchestrator.pid"
GATEWAY_DIR="$ROOT_DIR/gateway"

# ============================================================
# 工具函数
# ============================================================

log_info()  { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }
log_step()  { echo -e "${CYAN}[→]${NC} $1"; }

check_port() {
    local port=$1
    if lsof -ti:$port > /dev/null 2>&1; then
        return 0  # port in use
    fi
    return 1  # port free
}

wait_for_port() {
    local port=$1
    local name=$2
    local max_wait=${3:-30}
    local i=0
    while [ $i -lt $max_wait ]; do
        if check_port $port; then
            log_info "$name 已启动 (port $port)"
            return 0
        fi
        sleep 1
        i=$((i + 1))
    done
    log_error "$name 启动超时 (port $port)"
    return 1
}

kill_by_pid_file() {
    local pidfile=$1
    local name=$2
    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
            sleep 1
            kill -9 "$pid" 2>/dev/null || true
            log_info "已停止 $name (PID $pid)"
        fi
        rm -f "$pidfile"
    fi
}

kill_by_port() {
    local port=$1
    local name=$2
    local pids=$(lsof -ti:$port 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "$pids" | xargs kill -9 2>/dev/null || true
        log_info "已停止 $name (port $port)"
    fi
}

# ============================================================
# 启动函数
# ============================================================

start_rag_api() {
    if check_port 8000; then
        log_warn "RAG API Server 已在运行 (port 8000)"
        return 0
    fi
    log_step "启动 RAG API Server (port 8000)..."
    cd "$ROOT_DIR"
    PYTHONPATH="$ROOT_DIR" python project/api_server.py > "$LOG_DIR/rag_api.log" 2>&1 &
    echo $! > "$PID_RAG"
    if ! wait_for_port 8000 "RAG API Server" 15; then
        log_error "  日志: tail $LOG_DIR/rag_api.log"
        tail -3 "$LOG_DIR/rag_api.log" 2>/dev/null | sed 's/^/  /'
        return 1
    fi
}

start_a2a_agent() {
    if check_port 5001; then
        log_warn "A2A Expert Agent 已在运行 (port 5001)"
        return 0
    fi
    log_step "启动 A2A Expert Agent (port 5001)..."
    cd "$ROOT_DIR"
    PYTHONPATH="$ROOT_DIR" python project/a2a_agent.py --serve > "$LOG_DIR/a2a_agent.log" 2>&1 &
    echo $! > "$PID_A2A"
    if ! wait_for_port 5001 "A2A Expert Agent" 15; then
        log_error "  日志: tail $LOG_DIR/a2a_agent.log"
        tail -3 "$LOG_DIR/a2a_agent.log" 2>/dev/null | sed 's/^/  /'
        return 1
    fi
}

start_react_agent() {
    if check_port 5002; then
        log_warn "ReAct Agent 已在运行 (port 5002)"
        return 0
    fi
    log_step "启动 ReAct Agent (port 5002)..."
    cd "$ROOT_DIR"
    PYTHONPATH="$ROOT_DIR" python project/react_agent.py --serve > "$LOG_DIR/react_agent.log" 2>&1 &
    echo $! > "$PID_REACT"
    if ! wait_for_port 5002 "ReAct Agent" 15; then
        log_error "  日志: tail $LOG_DIR/react_agent.log"
        tail -3 "$LOG_DIR/react_agent.log" 2>/dev/null | sed 's/^/  /'
        return 1
    fi
}

start_frontend() {
    if check_port 3000; then
        log_warn "前端 Dev Server 已在运行 (port 3000)"
        return 0
    fi
    log_step "启动前端 Dev Server (port 3000)..."
    cd "$FRONTEND_DIR"
    npm run dev > "$LOG_DIR/frontend.log" 2>&1 &
    echo $! > "$PID_FRONT"
    wait_for_port 3000 "前端 Dev Server" 15
}

start_orchestrator() {
    if check_port 5003; then
        log_warn "Orchestrator 已在运行 (port 5003)"
        return 0
    fi
    log_step "启动 Multi-Agent Orchestrator (port 5003)..."
    cd "$ROOT_DIR"
    PYTHONPATH="$ROOT_DIR" python project/orchestrator/server.py > "$LOG_DIR/orchestrator.log" 2>&1 &
    echo $! > "$PID_ORCHESTRATOR"
    if ! wait_for_port 5003 "Orchestrator" 15; then
        log_error "  日志: tail $LOG_DIR/orchestrator.log"
        tail -3 "$LOG_DIR/orchestrator.log" 2>/dev/null | sed 's/^/  /'
        return 1
    fi
}

start_gateway() {
    if check_port 8080; then
        log_warn "API Gateway 已在运行 (port 8080)"
        return 0
    fi
    # 检查是否已编译
    local jar="$GATEWAY_DIR/target/gateway-1.0.0.jar"
    if [ ! -f "$jar" ]; then
        log_step "编译 Gateway (首次启动)..."
        cd "$GATEWAY_DIR"
        mvn clean package -DskipTests -q > "$LOG_DIR/gateway_build.log" 2>&1
        if [ $? -ne 0 ]; then
            log_error "Gateway 编译失败，查看: $LOG_DIR/gateway_build.log"
            return 1
        fi
    fi
    log_step "启动 API Gateway (port 8080)..."
    cd "$GATEWAY_DIR"
    java -jar "$jar" > "$LOG_DIR/gateway.log" 2>&1 &
    echo $! > "$PID_GATEWAY"
    if ! wait_for_port 8080 "API Gateway" 20; then
        log_error "  日志: tail $LOG_DIR/gateway.log"
        tail -3 "$LOG_DIR/gateway.log" 2>/dev/null | sed 's/^/  /'
        return 1
    fi
}

# ============================================================
# 停止函数
# ============================================================

stop_all() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════${NC}"
    echo -e "${CYAN}  停止所有服务${NC}"
    echo -e "${CYAN}═══════════════════════════════════════${NC}"
    echo ""

    kill_by_pid_file "$PID_FRONT" "前端 Dev Server"
    kill_by_port 3000 "前端 (port 3000)"

    kill_by_pid_file "$PID_RAG" "RAG API Server"
    kill_by_port 8000 "RAG API (port 8000)"

    kill_by_pid_file "$PID_A2A" "A2A Expert Agent"
    kill_by_port 5001 "A2A Agent (port 5001)"

    kill_by_pid_file "$PID_REACT" "ReAct Agent"
    kill_by_port 5002 "ReAct Agent (port 5002)"

    kill_by_pid_file "$PID_GATEWAY" "API Gateway"
    kill_by_port 8080 "Gateway (port 8080)"

    kill_by_pid_file "$PID_ORCHESTRATOR" "Orchestrator"
    kill_by_port 5003 "Orchestrator (port 5003)"

    echo ""
    log_info "所有服务已停止"
}

# ============================================================
# 状态函数
# ============================================================

show_status() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════${NC}"
    echo -e "${CYAN}  服务状态${NC}"
    echo -e "${CYAN}═══════════════════════════════════════${NC}"
    echo ""

    local services=("API Gateway:8080" "RAG API Server:8000" "A2A Expert Agent:5001" "ReAct Agent:5002" "Orchestrator:5003" "Frontend Dev:3000")

    for svc in "${services[@]}"; do
        local name="${svc%%:*}"
        local port="${svc##*:}"
        if check_port $port; then
            echo -e "  ${GREEN}●${NC}  $name  ${GREEN}运行中${NC}  (port $port)"
        else
            echo -e "  ${RED}●${NC}  $name  ${RED}未运行${NC}  (port $port)"
        fi
    done

    echo ""

    # 简单健康检查
    if check_port 8000; then
        local health=$(curl -s http://localhost:8000/health 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Ollama={d.get(\"ollama\",\"?\")}, VectorDB={d.get(\"vector_store\",\"?\")}, Vectors={d.get(\"vector_count\",\"?\")}')" 2>/dev/null || echo "无法获取")
        echo -e "  ${BLUE}ℹ${NC}  RAG 详情: $health"
    fi

    echo ""
    echo -e "  📂 日志目录: $LOG_DIR/"
    echo ""
}

# ============================================================
# 主入口
# ============================================================

print_banner() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  🚀 Agent Learning — 全栈启动${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  项目根目录: ${BLUE}$ROOT_DIR${NC}"
    echo -e "  前端目录:   ${BLUE}$FRONTEND_DIR${NC}"
    echo ""
}

print_summary() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  ✅ 启动完成${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  🖥️  前端:       ${GREEN}http://localhost:3000${NC}"
    echo -e "  �  Gateway:    ${GREEN}http://localhost:8080${NC}  (统一入口)"
    echo -e "  �📚  RAG API:    ${GREEN}http://localhost:8000/docs${NC}"
    echo -e "  🤖  A2A Agent:  ${GREEN}http://localhost:5001/.well-known/agent.json${NC}"
    echo -e "  ⚡  ReAct Agent: ${GREEN}http://localhost:5002/.well-known/agent.json${NC}"
    echo -e "  🎯  Orchestrator: ${GREEN}http://localhost:5003/.well-known/agent.json${NC}  (Multi-Agent)"
    echo ""
    echo -e "  📂 日志: $LOG_DIR/"
    echo -e "  🛑 停止: ${YELLOW}./start.sh stop${NC}"
    echo -e "  📊 状态: ${YELLOW}./start.sh status${NC}"
    echo ""
}

case "${1:-all}" in
    all)
        print_banner
        start_rag_api || log_warn "RAG API 启动失败，跳过 (可能缺少 Python 依赖，运行: pip install -r requirements.txt)"
        start_a2a_agent || log_warn "A2A Agent 启动失败，跳过"
        start_react_agent || log_warn "ReAct Agent 启动失败，跳过"
        start_orchestrator || log_warn "Orchestrator 启动失败，跳过"
        start_frontend
        start_gateway || log_warn "Gateway 启动失败，跳过 (需要 Java 21 + Maven)"
        print_summary
        ;;
    back|backend)
        print_banner
        start_rag_api || log_warn "RAG API 启动失败"
        start_a2a_agent || log_warn "A2A Agent 启动失败"
        start_react_agent || log_warn "ReAct Agent 启动失败"
        echo ""
        log_info "后端启动流程完成"
        echo ""
        ;;
    front|frontend)
        print_banner
        start_frontend
        echo ""
        log_info "前端已启动: http://localhost:3000"
        echo ""
        ;;
    rag)
        print_banner
        start_rag_api
        ;;
    a2a)
        print_banner
        start_a2a_agent
        ;;
    react)
        print_banner
        start_react_agent
        ;;
    gateway|gw)
        print_banner
        start_gateway
        ;;
    stop)
        stop_all
        ;;
    status|st)
        show_status
        ;;
    *)
        echo ""
        echo "用法: $0 [命令]"
        echo ""
        echo "命令:"
        echo "  all       启动全部服务 (默认)"
        echo "  back      只启动后端 (RAG + A2A + ReAct)"
        echo "  front     只启动前端"
        echo "  rag       只启动 RAG API Server (:8000)"
        echo "  a2a       只启动 A2A Expert Agent (:5001)"
        echo "  react     只启动 ReAct Agent (:5002)"
        echo "  gateway   只启动 API Gateway (:8080)"
        echo "  stop      停止所有服务"
        echo "  status    查看服务状态"
        echo ""
        ;;
esac
