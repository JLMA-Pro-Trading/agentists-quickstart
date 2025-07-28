"""
Enhanced Universal Trading Syndicate - Main FastAPI Application
Educational Trading System with Production Architecture
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import uvicorn

from config.system_config import CONFIG, TradingMode, SecurityLevel
from connectors.binance_connector import BinancePaperConnector
from neural_networks.daa_specialists import DAAManager, TradingSignal
from paper_trading.trading_engine import PaperTradingEngine
from vault.quantum_vault import QuantumSecurityVault
from security.progressive_trust import ProgressiveTrustSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TradingSyndicate")

# Global instances
binance_connector: Optional[BinancePaperConnector] = None
daa_manager: Optional[DAAManager] = None
trading_engine: Optional[PaperTradingEngine] = None
vault: Optional[QuantumSecurityVault] = None
trust_system: Optional[ProgressiveTrustSystem] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 Starting Enhanced Universal Trading Syndicate")
    logger.info(f"Mode: {CONFIG.trading_mode.value}")
    logger.info(f"Security Level: {CONFIG.security_level.value}")
    
    # Verify educational constraints
    if not CONFIG.enforce_educational_constraints():
        raise RuntimeError("Educational constraints validation failed")
    
    global binance_connector, daa_manager, trading_engine, vault, trust_system
    
    # Initialize core components
    binance_connector = BinancePaperConnector()
    await binance_connector.initialize()
    
    daa_manager = DAAManager()
    trading_engine = PaperTradingEngine()
    vault = QuantumSecurityVault()
    trust_system = ProgressiveTrustSystem()
    
    # Start background services
    asyncio.create_task(trading_engine.start_engine())
    
    logger.info("✅ Enhanced Universal Trading Syndicate Started")
    print("🎯 EDUCATIONAL TRADING SYSTEM READY")
    print("   - Paper Trading Only")
    print("   - Binance Testnet Integration")
    print("   - 12 Specialized DAA Neural Networks")
    print("   - Production-Ready Architecture")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Enhanced Universal Trading Syndicate")
    if trading_engine:
        await trading_engine.stop_engine()
    if binance_connector:
        await binance_connector.close()

# FastAPI application
app = FastAPI(
    title="Enhanced Universal Trading Syndicate",
    description="Educational Trading System with Advanced DAA Neural Networks",
    version="2.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Pydantic models
class PortfolioCreateRequest(BaseModel):
    user_id: str
    initial_balance: float = 10000.0

class OrderRequest(BaseModel):
    portfolio_id: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    order_type: str = "market"
    price: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str
    trading_mode: str
    educational_constraints: bool

# Dependency injection
async def get_trading_engine() -> PaperTradingEngine:
    if not trading_engine:
        raise HTTPException(status_code=503, detail="Trading engine not initialized")
    return trading_engine

async def get_daa_manager() -> DAAManager:
    if not daa_manager:
        raise HTTPException(status_code=503, detail="DAA manager not initialized")
    return daa_manager

async def get_binance_connector() -> BinancePaperConnector:
    if not binance_connector:
        raise HTTPEStatus=503, detail="Binance connector not initialized")
    return binance_connector

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check"""
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="2.0.0",
        trading_mode=CONFIG.trading_mode.value,
        educational_constraints=True
    )

# System information
@app.get("/system/info")
async def system_info():
    """Get system information and constraints"""
    return {
        "system": "Enhanced Universal Trading Syndicate",
        "version": "2.0.0",
        "mode": CONFIG.trading_mode.value,
        "security_level": CONFIG.security_level.value,
        "educational_only": True,
        "live_trading_disabled": True,
        "binance_testnet_only": True,
        "supported_exchanges": ["Binance Testnet"],
        "daa_count": len(daa_manager.daas) if daa_manager else 0,
        "constraints": {
            "paper_trading_only": True,
            "max_position_size": CONFIG.max_position_size,
            "require_explicit_consent": CONFIG.require_explicit_consent
        }
    }

# Portfolio management
@app.post("/portfolios")
async def create_portfolio(
    request: PortfolioCreateRequest,
    engine: PaperTradingEngine = Depends(get_trading_engine)
):
    """Create a new paper trading portfolio"""
    try:
        portfolio_id = await engine.create_portfolio(
            request.user_id, 
            request.initial_balance
        )
        return {
            "portfolio_id": portfolio_id,
            "user_id": request.user_id,
            "initial_balance": request.initial_balance,
            "status": "created",
            "educational_mode": True
        }
    except Exception as e:
        logger.error(f"Portfolio creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolios/{portfolio_id}")
async def get_portfolio(
    portfolio_id: str,
    engine: PaperTradingEngine = Depends(get_trading_engine)
):
    """Get portfolio status and performance"""
    status = engine.get_portfolio_status(portfolio_id)
    if not status:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return status

@app.get("/portfolios")
async def list_portfolios(
    engine: PaperTradingEngine = Depends(get_trading_engine)
):
    """List all portfolios"""
    portfolios = []
    for portfolio_id, portfolio in engine.portfolios.items():
        portfolios.append({
            "portfolio_id": portfolio_id,
            "user_id": portfolio.user_id,
            "status": portfolio.status.value,
            "equity": portfolio.equity,
            "performance": portfolio.performance_metrics
        })
    return {"portfolios": portfolios, "count": len(portfolios)}

# Trading operations
@app.post("/orders")
async def place_order(
    request: OrderRequest,
    engine: PaperTradingEngine = Depends(get_trading_engine)
):
    """Place a paper trading order"""
    from connectors.universal_broker import OrderRequest as BrokerOrderRequest
    
    broker_request = BrokerOrderRequest(
        symbol=request.symbol,
        side=request.side,
        amount=request.quantity,
        order_type=request.order_type,
        price=request.price,
        paper_trading=True  # Force paper trading
    )
    
    try:
        order_id = await engine.place_order(request.portfolio_id, broker_request)
        if not order_id:
            raise HTTPException(status_code=400, detail="Order rejected")
        
        return {
            "order_id": order_id,
            "portfolio_id": request.portfolio_id,
            "symbol": request.symbol,
            "side": request.side,
            "quantity": request.quantity,
            "status": "placed",
            "paper_trading": True
        }
    except Exception as e:
        logger.error(f"Order placement error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Market data
@app.get("/market/{symbol}")
async def get_market_data(
    symbol: str,
    connector: BinancePaperConnector = Depends(get_binance_connector)
):
    """Get real-time market data"""
    market_data = await connector.get_market_data(symbol)
    if not market_data:
        raise HTTPException(status_code=404, detail="Symbol not found")
    
    return {
        "symbol": market_data.symbol,
        "price": market_data.price,
        "bid": market_data.bid_price,
        "ask": market_data.ask_price,
        "volume": market_data.volume,
        "high_24h": market_data.high,
        "low_24h": market_data.low,
        "change_24h": market_data.change_24h,
        "change_percent_24h": market_data.change_percent_24h,
        "timestamp": market_data.timestamp
    }

@app.get("/market/{symbol}/orderbook")
async def get_order_book(
    symbol: str,
    limit: int = 100,
    connector: BinancePaperConnector = Depends(get_binance_connector)
):
    """Get order book data"""
    order_book = await connector.get_order_book(symbol, limit)
    if not order_book:
        raise HTTPException(status_code=404, detail="Order book not available")
    
    return {
        "symbol": order_book.symbol,
        "bids": order_book.bids[:limit],
        "asks": order_book.asks[:limit],
        "timestamp": order_book.timestamp
    }

@app.get("/market/symbols")
async def get_supported_symbols(
    connector: BinancePaperConnector = Depends(get_binance_connector)
):
    """Get supported trading symbols"""
    return {
        "symbols": connector.get_supported_symbols(),
        "count": len(connector.get_supported_symbols()),
        "exchange": "Binance Testnet",
        "educational_only": True
    }

# DAA Neural Networks
@app.get("/daa/signals/{symbol}")
async def get_daa_signals(
    symbol: str,
    manager: DAAManager = Depends(get_daa_manager),
    connector: BinancePaperConnector = Depends(get_binance_connector)
):
    """Get trading signals from all DAA neural networks"""
    # Get current market data
    market_data = await connector.get_market_data(symbol)
    if not market_data:
        raise HTTPException(status_code=404, detail="Market data not available")
    
    # Convert to dictionary format for DAAs
    market_dict = {
        "price": market_data.price,
        "volume": market_data.volume,
        "bid_price": market_data.bid_price,
        "ask_price": market_data.ask_price,
        "high": market_data.high,
        "low": market_data.low,
        "change_24h": market_data.change_24h
    }
    
    # Get signals from all DAAs
    signals = await manager.get_all_signals(market_dict)
    
    return {
        "symbol": symbol,
        "timestamp": time.time(),
        "signals": [
            {
                "daa_id": signal.daa_id,
                "action": signal.action,
                "confidence": signal.confidence,
                "size_ratio": signal.size_ratio,
                "reasoning": signal.reasoning
            }
            for signal in signals
        ],
        "total_signals": len(signals)
    }

@app.get("/daa/performance")
async def get_daa_performance(
    manager: DAAManager = Depends(get_daa_manager)
):
    """Get performance report for all DAA neural networks"""
    return manager.get_performance_report()

@app.get("/daa/list")
async def list_daas(
    manager: DAAManager = Depends(get_daa_manager)
):
    """List all available DAA neural networks"""
    daas = []
    for daa_id, daa in manager.daas.items():
        daas.append({
            "daa_id": daa_id,
            "strategy_type": daa.strategy_type.value,
            "primary_asset": daa.primary_asset,
            "risk_tolerance": daa.risk_tolerance,
            "total_trades": daa.total_trades,
            "win_rate": (daa.winning_trades / daa.total_trades * 100) if daa.total_trades > 0 else 0
        })
    
    return {
        "daas": daas,
        "total_count": len(daas),
        "educational_mode": True
    }

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket):
    """WebSocket endpoint for real-time market data and signals"""
    await websocket.accept()
    
    try:
        while True:
            # Send real-time updates
            data = {
                "timestamp": time.time(),
                "type": "heartbeat",
                "message": "Educational trading system active"
            }
            
            await websocket.send_json(data)
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    return HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    # Ensure educational constraints
    print("🎓 EDUCATIONAL TRADING SYSTEM")
    print("   PAPER TRADING ONLY - NO LIVE TRADING")
    print("   BINANCE TESTNET ENDPOINTS ONLY")
    print("   FULL MONITORING AND TRANSPARENCY")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        access_log=True
    )