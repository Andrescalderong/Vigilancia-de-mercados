#!/bin/bash
# ===========================================
# Market Intelligence Platform
# Deployment Script
# ===========================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "\n${BLUE}=========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Docker
    if command -v docker &> /dev/null; then
        print_success "Docker installed: $(docker --version)"
    else
        print_error "Docker not found. Please install Docker."
        exit 1
    fi
    
    # Docker Compose
    if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
        print_success "Docker Compose available"
    else
        print_error "Docker Compose not found."
        exit 1
    fi
    
    # .env file
    if [ -f .env ]; then
        print_success ".env file exists"
    else
        print_warning ".env file not found. Copying from .env.example..."
        cp .env.example .env
        print_warning "Please edit .env with your API keys before continuing."
        exit 1
    fi
}

# Validate environment
validate_env() {
    print_header "Validating Environment Variables"
    
    source .env
    
    # Check LLM keys
    if [ -n "$ANTHROPIC_API_KEY" ] || [ -n "$OPENAI_API_KEY" ]; then
        print_success "LLM API key configured"
    else
        print_error "No LLM API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY"
        exit 1
    fi
    
    # Check Google API (optional but recommended)
    if [ -n "$GOOGLE_API_KEY" ] && [ -n "$GOOGLE_SEARCH_ENGINE_ID" ]; then
        print_success "Google Search API configured"
    else
        print_warning "Google Search API not configured (optional)"
    fi
    
    # Check Finnhub (optional)
    if [ -n "$FINNHUB_API_KEY" ]; then
        print_success "Finnhub API configured"
    else
        print_warning "Finnhub API not configured (optional)"
    fi
}

# Build images
build_images() {
    print_header "Building Docker Images"
    
    docker compose build --no-cache
    
    print_success "Images built successfully"
}

# Deploy local development
deploy_local() {
    print_header "Deploying Local Development Environment"
    
    docker compose up -d backend redis
    
    echo ""
    print_success "Local deployment complete!"
    echo ""
    echo "Services:"
    echo "  - Backend API: http://localhost:8000"
    echo "  - API Docs: http://localhost:8000/docs"
    echo "  - Health: http://localhost:8000/health"
    echo ""
    echo "View logs: docker compose logs -f backend"
}

# Deploy with frontend
deploy_full() {
    print_header "Deploying Full Stack"
    
    docker compose up -d
    
    echo ""
    print_success "Full deployment complete!"
    echo ""
    echo "Services:"
    echo "  - Frontend: http://localhost:3000"
    echo "  - Backend API: http://localhost:8000"
    echo "  - API Docs: http://localhost:8000/docs"
    echo ""
}

# Deploy production
deploy_production() {
    print_header "Deploying Production Environment"
    
    # Check for SSL certificates
    if [ ! -f ./nginx/ssl/fullchain.pem ]; then
        print_warning "SSL certificates not found in ./nginx/ssl/"
        print_warning "Please add your SSL certificates or use Let's Encrypt"
        echo ""
        read -p "Continue without SSL? (y/N) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    docker compose --profile production up -d
    
    echo ""
    print_success "Production deployment complete!"
    echo ""
}

# Deploy with monitoring
deploy_monitoring() {
    print_header "Deploying with Monitoring Stack"
    
    docker compose --profile monitoring up -d
    
    echo ""
    print_success "Monitoring stack deployed!"
    echo ""
    echo "Services:"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Grafana: http://localhost:3001 (admin/admin)"
    echo ""
}

# Stop all services
stop_services() {
    print_header "Stopping All Services"
    
    docker compose --profile production --profile monitoring down
    
    print_success "All services stopped"
}

# Clean up
cleanup() {
    print_header "Cleaning Up"
    
    docker compose down -v --remove-orphans
    docker system prune -f
    
    print_success "Cleanup complete"
}

# AWS deployment
deploy_aws() {
    print_header "Deploying to AWS ECS"
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI not found. Please install it first."
        exit 1
    fi
    
    # Login to ECR
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
    
    # Build and push images
    docker compose -f docker-compose.cloud.yml build
    docker compose -f docker-compose.cloud.yml push
    
    # Deploy to ECS
    docker compose -f docker-compose.cloud.yml up
    
    print_success "AWS deployment initiated"
}

# GCP deployment
deploy_gcp() {
    print_header "Deploying to Google Cloud Run"
    
    # Check gcloud CLI
    if ! command -v gcloud &> /dev/null; then
        print_error "gcloud CLI not found. Please install it first."
        exit 1
    fi
    
    # Deploy backend
    gcloud run deploy mi-backend \
        --source ./backend \
        --platform managed \
        --region $GCP_REGION \
        --allow-unauthenticated \
        --set-env-vars="$(cat .env | grep -v '^#' | xargs | tr ' ' ',')"
    
    print_success "GCP deployment complete"
}

# Show status
show_status() {
    print_header "Service Status"
    
    docker compose ps
    
    echo ""
    echo "Health checks:"
    curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || echo "Backend not responding"
}

# Show help
show_help() {
    echo "Market Intelligence Platform - Deployment Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  check       Check prerequisites and environment"
    echo "  build       Build Docker images"
    echo "  local       Deploy local development (backend + redis)"
    echo "  full        Deploy full stack (backend + frontend + redis)"
    echo "  production  Deploy production with nginx"
    echo "  monitoring  Deploy with Prometheus + Grafana"
    echo "  aws         Deploy to AWS ECS"
    echo "  gcp         Deploy to Google Cloud Run"
    echo "  stop        Stop all services"
    echo "  cleanup     Stop and clean up all resources"
    echo "  status      Show service status"
    echo "  help        Show this help message"
    echo ""
}

# Main
case "${1:-help}" in
    check)
        check_prerequisites
        validate_env
        ;;
    build)
        check_prerequisites
        build_images
        ;;
    local)
        check_prerequisites
        validate_env
        deploy_local
        ;;
    full)
        check_prerequisites
        validate_env
        build_images
        deploy_full
        ;;
    production)
        check_prerequisites
        validate_env
        build_images
        deploy_production
        ;;
    monitoring)
        check_prerequisites
        deploy_monitoring
        ;;
    aws)
        deploy_aws
        ;;
    gcp)
        deploy_gcp
        ;;
    stop)
        stop_services
        ;;
    cleanup)
        cleanup
        ;;
    status)
        show_status
        ;;
    help|*)
        show_help
        ;;
esac
