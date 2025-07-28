"""
Universal Broker Integration Framework
Enhanced CCXT + Gateway + Custom Connectors
Educational Trading System
"""

import asyncio
import ccxt
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import time
import json

from ..vault.quantum_vault import QuantumSecurityVault, APICredential
from ..config.system_config import CONFIG, TradingMode

class ConnectorType(Enum):
    CEX = "centralized_exchange"  # CCXT-based
    DEX = "decentralized_exchange"  # Gateway Protocol
    PREDICTION = "prediction_market"  # Polymarket
    SPORTS = "sports_betting"  # Sports APIs
    CUSTOM = "custom_platform"  # Future integrations

class ConnectionStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    PAPER_ONLY = "paper_only"  # Educational mode

@dataclass
class MarketData:
    """Standardized market data structure"""
    symbol: str
    timestamp: float
    bid: float
    ask: float
    last: float
    volume: float
    high: float
    low: float
    change: float
    change_percent: float

@dataclass
class OrderRequest:
    """Standardized order request"""
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    price: Optional[float] = None  # None for market orders
    order_type: str = "limit"  # 'market', 'limit', 'stop'
    paper_trading: bool = True  # Default to paper trading

@dataclass
class OrderResponse:
    """Standardized order response"""
    order_id: str
    symbol: str
    side: str
    amount: float
    price: float
    status: str
    timestamp: float
    fees: float = 0.0
    paper_trading: bool = True

class BaseConnector(ABC):
    """Abstract base class for all connectors"""
    
    def __init__(self, platform: str, connector_type: ConnectorType):
        self.platform = platform
        self.connector_type = connector_type
        self.status = ConnectionStatus.DISCONNECTED
        self.vault = QuantumSecurityVault()
        self.logger = logging.getLogger(f'Connector.{platform}')
        
        # Educational constraints
        self.paper_trading_only = True
        self.educational_mode = CONFIG.trading_mode == TradingMode.OBSERVER
        
    @abstractmethod
    async def connect(self, credentials: APICredential) -> bool:
        """Connect to the platform"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the platform"""
        pass
    
    @abstractmethod
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get real-time market data"""
        pass
    
    @abstractmethod
    async def place_order(self, order: OrderRequest) -> Optional[OrderResponse]:
        """Place trading order (paper only in educational mode)"""
        pass
    
    @abstractmethod
    async def get_balance(self) -> Dict[str, float]:
        """Get account balance"""
        pass
    
    def enforce_educational_constraints(self, order: OrderRequest) -> OrderRequest:
        """Ensure all orders are paper trading in educational mode"""
        if self.educational_mode or self.paper_trading_only:
            order.paper_trading = True
        
        # Additional safety checks
        if order.amount > CONFIG.max_position_size:
            raise ValueError(f"Order size {order.amount} exceeds educational limit {CONFIG.max_position_size}")
        
        return order

class CCXTConnector(BaseConnector):
    """Enhanced CCXT connector for centralized exchanges"""
    
    def __init__(self, exchange_name: str):
        super().__init__(exchange_name, ConnectorType.CEX)
        self.exchange_class = getattr(ccxt, exchange_name, None)
        self.exchange = None
        
        if not self.exchange_class:
            raise ValueError(f"Exchange {exchange_name} not supported by CCXT")
    
    async def connect(self, credentials: APICredential) -> bool:
        """Connect to CCXT exchange"""
        try:
            self.status = ConnectionStatus.CONNECTING
            
            # Initialize exchange with credentials
            config = {
                'apiKey': credentials.api_key,
                'secret': credentials.secret_key,
                'sandbox': credentials.sandbox,  # Always use sandbox in educational mode
                'enableRateLimit': True,
            }
            
            if credentials.passphrase:
                config['password'] = credentials.passphrase
            
            # Force sandbox mode for educational trading
            if self.educational_mode:
                config['sandbox'] = True
            
            self.exchange = self.exchange_class(config)
            
            # Test connection
            await self.exchange.load_markets()
            
            self.status = ConnectionStatus.CONNECTED
            self.logger.info(f"Connected to {self.platform} (sandbox: {config['sandbox']})")
            
            return True
            
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.logger.error(f"Failed to connect to {self.platform}: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from exchange"""
        try:
            if self.exchange:
                await self.exchange.close()
            self.status = ConnectionStatus.DISCONNECTED
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting from {self.platform}: {e}")
            return False
    
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get real-time market data from CCXT exchange"""
        if not self.exchange or self.status != ConnectionStatus.CONNECTED:
            return None
        
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            
            return MarketData(
                symbol=symbol,
                timestamp=ticker['timestamp'],
                bid=ticker['bid'],
                ask=ticker['ask'],
                last=ticker['last'],
                volume=ticker['baseVolume'],
                high=ticker['high'],
                low=ticker['low'],
                change=ticker['change'],
                change_percent=ticker['percentage']
            )
        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol}: {e}")
            return None
    
    async def place_order(self, order: OrderRequest) -> Optional[OrderResponse]:
        """Place order (educational constraints enforced)"""
        # Enforce educational constraints
        order = self.enforce_educational_constraints(order)
        
        if not order.paper_trading and self.educational_mode:
            print("⚠️  Live trading disabled in educational mode")
            return None
        
        try:
            if order.paper_trading:
                # Simulate paper trading order
                return self._simulate_paper_order(order)
            
            # This would be live trading (disabled in educational mode)
            if CONFIG.live_trading_enabled and CONFIG.require_explicit_consent:
                print("🚨 Live trading requires explicit user consent")
                return None
            
            # Live order execution would go here
            # Currently disabled for educational system
            print("🚫 Live trading not implemented in educational version")
            return None
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    def _simulate_paper_order(self, order: OrderRequest) -> OrderResponse:
        """Simulate paper trading order"""
        order_id = f"paper_{int(time.time())}_{order.symbol}"
        
        # Simulate order execution with current market price
        current_price = order.price if order.price else 100.0  # Mock price
        
        return OrderResponse(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            amount=order.amount,
            price=current_price,
            status="filled",
            timestamp=time.time(),
            fees=current_price * order.amount * 0.001,  # 0.1% fee simulation
            paper_trading=True
        )
    
    async def get_balance(self) -> Dict[str, float]:
        """Get account balance"""
        if not self.exchange or self.status != ConnectionStatus.CONNECTED:
            return {}
        
        try:
            if self.paper_trading_only:
                # Return simulated paper trading balance
                return {
                    "USD": 10000.0,  # $10,000 paper trading balance
                    "BTC": 0.0,
                    "ETH": 0.0
                }
            
            balance = await self.exchange.fetch_balance()
            return balance['total']
            
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return {}

class GatewayConnector(BaseConnector):
    """Gateway Protocol connector for DEX platforms"""
    
    def __init__(self, dex_name: str):
        super().__init__(dex_name, ConnectorType.DEX)
        self.gateway_url = "http://localhost:15888"  # Default Gateway URL
    
    async def connect(self, credentials: APICredential) -> bool:
        """Connect to DEX via Gateway Protocol"""
        try:
            self.status = ConnectionStatus.CONNECTING
            
            # In educational mode, simulate connection
            if self.educational_mode:
                self.status = ConnectionStatus.PAPER_ONLY
                self.logger.info(f"Connected to {self.platform} in paper mode")
                return True
            
            # Real Gateway connection would go here
            self.status = ConnectionStatus.CONNECTED
            return True
            
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.logger.error(f"Failed to connect to {self.platform}: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from DEX"""
        self.status = ConnectionStatus.DISCONNECTED
        return True
    
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get DEX market data"""
        if self.status == ConnectionStatus.PAPER_ONLY:
            # Return simulated DEX data
            return MarketData(
                symbol=symbol,
                timestamp=time.time(),
                bid=99.5,
                ask=100.5,
                last=100.0,
                volume=1000000,
                high=105.0,
                low=95.0,
                change=5.0,
                change_percent=5.0
            )
        
        return None
    
    async def place_order(self, order: OrderRequest) -> Optional[OrderResponse]:
        """Place DEX order (paper only in educational mode)"""
        order = self.enforce_educational_constraints(order)
        
        if order.paper_trading:
            return self._simulate_dex_order(order)
        
        print("🚫 Live DEX trading not implemented in educational version")
        return None
    
    def _simulate_dex_order(self, order: OrderRequest) -> OrderResponse:
        """Simulate DEX paper trading order"""
        order_id = f"dex_paper_{int(time.time())}_{order.symbol}"
        
        return OrderResponse(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            amount=order.amount,
            price=order.price or 100.0,
            status="filled",
            timestamp=time.time(),
            fees=0.003 * order.amount * (order.price or 100.0),  # 0.3% DEX fee
            paper_trading=True
        )
    
    async def get_balance(self) -> Dict[str, float]:
        """Get DEX balance (simulated in educational mode)"""
        return {
            "ETH": 10.0,  # 10 ETH paper balance
            "USDC": 10000.0,  # $10,000 USDC paper balance
            "DAI": 5000.0
        }

class UniversalBrokerManager:
    """
    Universal Broker Manager
    Coordinates all connector types with educational constraints
    """
    
    def __init__(self):
        self.connectors: Dict[str, BaseConnector] = {}
        self.vault = QuantumSecurityVault()
        self.logger = logging.getLogger('UniversalBroker')
        
        # Educational mode enforcement
        self.educational_mode = CONFIG.trading_mode == TradingMode.OBSERVER
        
        # Supported platforms
        self.supported_platforms = {
            'cex': ['binance', 'coinbase', 'kraken', 'okx'],
            'dex': ['uniswap', 'pancakeswap', 'balancer'],
            'prediction': ['polymarket'],
            'sports': ['sportsbet_api']
        }
    
    async def add_platform(self, platform: str, platform_type: str = 'cex') -> bool:
        """Add new trading platform"""
        try:
            # Create appropriate connector
            if platform_type == 'cex':
                connector = CCXTConnector(platform)
            elif platform_type == 'dex':
                connector = GatewayConnector(platform)
            else:
                print(f"❌ Platform type {platform_type} not yet implemented")
                return False
            
            # Get credentials from vault
            credentials = self.vault.get_credentials(platform)
            if not credentials:
                print(f"⚠️  No credentials found for {platform}")
                print(f"   Please add credentials to vault first")
                return False
            
            # Connect to platform
            success = await connector.connect(credentials)
            if success:
                self.connectors[platform] = connector
                print(f"✅ Connected to {platform}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error adding platform {platform}: {e}")
            return False
    
    async def get_market_data(self, platform: str, symbol: str) -> Optional[MarketData]:
        """Get market data from specific platform"""
        if platform not in self.connectors:
            return None
        
        return await self.connectors[platform].get_market_data(symbol)
    
    async def place_order(self, platform: str, order: OrderRequest) -> Optional[OrderResponse]:
        """Place order on specific platform"""
        if platform not in self.connectors:
            print(f"❌ Platform {platform} not connected")
            return None
        
        return await self.connectors[platform].place_order(order)
    
    def get_connected_platforms(self) -> List[str]:
        """Get list of connected platforms"""
        return [
            platform for platform, connector in self.connectors.items()
            if connector.status in [ConnectionStatus.CONNECTED, ConnectionStatus.PAPER_ONLY]
        ]
    
    def get_platform_status(self) -> Dict[str, str]:
        """Get status of all platforms"""
        return {
            platform: connector.status.value
            for platform, connector in self.connectors.items()
        }
    
    async def shutdown(self):
        """Shutdown all connections"""
        for connector in self.connectors.values():
            await connector.disconnect()
        
        self.connectors.clear()
        print("🔌 All platform connections closed")

# Example usage
if __name__ == "__main__":
    async def main():
        broker = UniversalBrokerManager()
        
        # Example: Add educational Binance testnet
        success = await broker.add_platform('binance', 'cex')
        
        if success:
            # Get market data
            data = await broker.get_market_data('binance', 'BTC/USDT')
            print(f"Market Data: {data}")
            
            # Place paper trading order
            order = OrderRequest(
                symbol='BTC/USDT',
                side='buy',
                amount=0.001,
                price=50000.0,
                paper_trading=True
            )
            
            response = await broker.place_order('binance', order)
            print(f"Order Response: {response}")
        
        await broker.shutdown()
    
    asyncio.run(main())