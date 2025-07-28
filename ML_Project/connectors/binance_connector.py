"""
Binance Paper Trading Connector
Educational-only implementation with strict paper trading constraints
"""

import asyncio
import json
import time
import hmac
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import websockets
import aiohttp
import logging

from ..config.system_config import CONFIG, TradingMode

class BinanceOrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"
    LIMIT_MAKER = "LIMIT_MAKER"

class BinanceOrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class BinanceMarketData:
    """Binance market data structure"""
    symbol: str
    price: float
    bid_price: float
    ask_price: float
    volume: float
    high: float
    low: float
    change_24h: float
    change_percent_24h: float
    timestamp: float

@dataclass
class BinanceOrderBook:
    """Binance order book data"""
    symbol: str
    bids: List[List[float]]  # [price, quantity]
    asks: List[List[float]]  # [price, quantity]
    timestamp: float

class BinancePaperConnector:
    """
    Binance Paper Trading Connector
    EDUCATIONAL USE ONLY - No live trading capabilities
    """
    
    def __init__(self, paper_mode=True):
        self.logger = logging.getLogger('BinancePaper')
        
        # CRITICAL: Force paper trading mode
        self.paper_mode = True  # Always True for educational use
        self.live_trading_enabled = False  # Never allow live trading
        
        # Binance testnet endpoints (paper trading)
        self.base_url = "https://testnet.binance.vision"  # Testnet only
        self.ws_url = "wss://testnet-dstream.binance.vision/ws"
        
        # Educational constraints
        if CONFIG.trading_mode == TradingMode.OBSERVER:
            self.api_calls_enabled = False
        else:
            self.api_calls_enabled = True
        
        # Connection state
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_connections: Dict[str, Any] = {}
        self.market_data_cache: Dict[str, BinanceMarketData] = {}
        
        # Rate limiting (educational limits)
        self.request_weight = 0
        self.request_limit = 1200  # Per minute
        self.last_request_time = 0
        
        # Supported symbols for educational trading
        self.supported_symbols = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT",
            "SOLUSDT", "DOTUSDT", "DOGEUSDT", "AVAXUSDT", "MATICUSDT",
            "LINKUSDT", "LTCUSDT", "UNIUSDT", "ATOMUSDT", "VETUSDT"
        ]
        
        self.logger.info("Binance Paper Trading Connector initialized")
        print("🏦 Binance Paper Trading Connector")
        print("   Mode: EDUCATIONAL PAPER TRADING ONLY")
        print("   Endpoint: Binance Testnet")
        print(f"   Supported Symbols: {len(self.supported_symbols)}")
    
    async def initialize(self) -> bool:
        """Initialize connector with educational constraints"""
        if not self.paper_mode:
            raise ValueError("Live trading not allowed in educational mode")
        
        # Create HTTP session
        self.session = aiohttp.ClientSession()
        
        # Test connection to testnet
        try:
            async with self.session.get(f"{self.base_url}/api/v3/ping") as response:
                if response.status == 200:
                    self.logger.info("Connected to Binance testnet")
                    print("✅ Connected to Binance Testnet")
                    return True
                else:
                    self.logger.error(f"Failed to connect: {response.status}")
                    return False
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            print(f"❌ Connection failed: {e}")
            return False
    
    async def get_market_data(self, symbol: str) -> Optional[BinanceMarketData]:
        """Get real-time market data for educational analysis"""
        if symbol not in self.supported_symbols:
            self.logger.warning(f"Symbol {symbol} not supported for educational trading")
            return None
        
        if not self.session:
            await self.initialize()
        
        try:
            # Get 24hr ticker statistics
            async with self.session.get(
                f"{self.base_url}/api/v3/ticker/24hr",
                params={"symbol": symbol}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    market_data = BinanceMarketData(
                        symbol=symbol,
                        price=float(data['lastPrice']),
                        bid_price=float(data['bidPrice']),
                        ask_price=float(data['askPrice']),
                        volume=float(data['volume']),
                        high=float(data['highPrice']),
                        low=float(data['lowPrice']),
                        change_24h=float(data['priceChange']),
                        change_percent_24h=float(data['priceChangePercent']),
                        timestamp=time.time()
                    )
                    
                    self.market_data_cache[symbol] = market_data
                    return market_data
                else:
                    self.logger.error(f"Failed to get market data: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Market data error: {e}")
            return None
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> Optional[BinanceOrderBook]:
        """Get order book data for educational analysis"""
        if symbol not in self.supported_symbols:
            return None
        
        if not self.session:
            await self.initialize()
        
        try:
            async with self.session.get(
                f"{self.base_url}/api/v3/depth",
                params={"symbol": symbol, "limit": limit}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Convert to float arrays
                    bids = [[float(price), float(qty)] for price, qty in data['bids']]
                    asks = [[float(price), float(qty)] for price, qty in data['asks']]
                    
                    return BinanceOrderBook(
                        symbol=symbol,
                        bids=bids,
                        asks=asks,
                        timestamp=time.time()
                    )
                else:
                    self.logger.error(f"Failed to get order book: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Order book error: {e}")
            return None
    
    async def place_paper_order(self, symbol: str, side: str, quantity: float,
                               order_type: str = "MARKET", price: Optional[float] = None) -> Dict[str, Any]:
        """
        Simulate order placement for educational purposes
        NO REAL ORDERS ARE PLACED
        """
        if not self.paper_mode:
            raise ValueError("Only paper trading allowed")
        
        if symbol not in self.supported_symbols:
            return {"error": f"Symbol {symbol} not supported"}
        
        # Generate paper order ID
        order_id = f"PAPER_ORDER_{int(time.time())}_{hash(f'{symbol}{side}{quantity}') % 10000}"
        
        # Get current market data for realistic simulation
        market_data = await self.get_market_data(symbol)
        if not market_data:
            return {"error": "Unable to get market data"}
        
        # Simulate order execution
        execution_price = price if price and order_type == "LIMIT" else market_data.price
        
        paper_order = {
            "orderId": order_id,
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "price": execution_price,
            "status": "FILLED",  # Paper orders always fill for educational purposes
            "executedQty": quantity,
            "executedPrice": execution_price,
            "commission": quantity * execution_price * 0.001,  # 0.1% fee simulation
            "commissionAsset": "USDT",
            "time": int(time.time() * 1000),
            "paper_trading": True,
            "educational_only": True
        }
        
        self.logger.info(f"Paper order simulated: {order_id}")
        print(f"📝 Paper Order Simulated")
        print(f"   Order ID: {order_id}")
        print(f"   Symbol: {symbol}")
        print(f"   Side: {side}")
        print(f"   Quantity: {quantity}")
        print(f"   Price: ${execution_price}")
        print(f"   Status: FILLED (PAPER TRADING)")
        
        return paper_order
    
    async def get_kline_data(self, symbol: str, interval: str = "1m", 
                            limit: int = 100) -> List[List[float]]:
        """Get historical kline/candlestick data for educational analysis"""
        if symbol not in self.supported_symbols:
            return []
        
        if not self.session:
            await self.initialize()
        
        try:
            async with self.session.get(
                f"{self.base_url}/api/v3/klines",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "limit": limit
                }
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Convert to float arrays: [timestamp, open, high, low, close, volume]
                    klines = []
                    for kline in data:
                        klines.append([
                            float(kline[0]) / 1000,  # timestamp in seconds
                            float(kline[1]),  # open
                            float(kline[2]),  # high
                            float(kline[3]),  # low
                            float(kline[4]),  # close
                            float(kline[5])   # volume
                        ])
                    
                    return klines
                else:
                    self.logger.error(f"Failed to get kline data: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Kline data error: {e}")
            return []
    
    async def start_websocket_stream(self, symbol: str, callback) -> bool:
        """Start WebSocket stream for real-time data (educational monitoring)"""
        if not self.paper_mode:
            return False
        
        symbol_lower = symbol.lower()
        stream_url = f"{self.ws_url}/{symbol_lower}@ticker"
        
        try:
            async def websocket_handler():
                async with websockets.connect(stream_url) as websocket:
                    self.ws_connections[symbol] = websocket
                    self.logger.info(f"WebSocket connected for {symbol}")
                    
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            
                            # Convert to our market data format
                            market_data = BinanceMarketData(
                                symbol=data['s'],
                                price=float(data['c']),
                                bid_price=float(data.get('b', data['c'])),
                                ask_price=float(data.get('a', data['c'])),
                                volume=float(data['v']),
                                high=float(data['h']),
                                low=float(data['l']),
                                change_24h=float(data['P']),
                                change_percent_24h=float(data['P']),
                                timestamp=float(data['E']) / 1000
                            )
                            
                            self.market_data_cache[symbol] = market_data
                            await callback(market_data)
                            
                        except Exception as e:
                            self.logger.error(f"WebSocket message error: {e}")
            
            # Start WebSocket in background
            asyncio.create_task(websocket_handler())
            return True
            
        except Exception as e:
            self.logger.error(f"WebSocket connection error: {e}")
            return False
    
    async def close(self):
        """Close all connections"""
        if self.session:
            await self.session.close()
        
        for ws in self.ws_connections.values():
            if not ws.closed:
                await ws.close()
        
        self.logger.info("Binance connector closed")
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols for educational trading"""
        return self.supported_symbols.copy()
    
    def is_paper_trading_only(self) -> bool:
        """Confirm this is paper trading only"""
        return True  # Always True for educational version

# Example usage for educational purposes
if __name__ == "__main__":
    async def main():
        connector = BinancePaperConnector()
        
        # Initialize
        if await connector.initialize():
            print("✅ Binance Paper Trading Connector Ready")
            
            # Get market data
            market_data = await connector.get_market_data("BTCUSDT")
            if market_data:
                print(f"BTC Price: ${market_data.price}")
            
            # Simulate paper order
            order = await connector.place_paper_order(
                symbol="BTCUSDT",
                side="BUY",
                quantity=0.001,
                order_type="MARKET"
            )
            print(f"Paper Order: {order}")
            
            # Close connections
            await connector.close()
    
    asyncio.run(main())