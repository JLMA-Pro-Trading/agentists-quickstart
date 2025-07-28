"""
Paper Trading Engine
Real Market Data with Virtual Portfolio Management
Educational Trading System with Risk Assessment
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging
from collections import defaultdict, deque
import random

from ..config.system_config import CONFIG, TradingMode
from ..connectors.universal_broker import UniversalBrokerManager, MarketData, OrderRequest, OrderResponse
from ..vault.quantum_vault import QuantumSecurityVault
from ..security.progressive_trust import ProgressiveTrustSystem

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class PortfolioStatus(Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    LIQUIDATING = "liquidating"
    CLOSED = "closed"

@dataclass
class PaperPosition:
    """Paper trading position"""
    symbol: str
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    current_price: float
    entry_time: float
    unrealized_pnl: float
    realized_pnl: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class PaperOrder:
    """Paper trading order"""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: str  # 'market', 'limit', 'stop'
    price: Optional[float]
    status: OrderStatus
    created_at: float
    filled_at: Optional[float] = None
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    fees: float = 0.0
    
@dataclass
class VirtualPortfolio:
    """Virtual portfolio for paper trading"""
    portfolio_id: str
    user_id: str
    initial_balance: float
    current_balance: float
    equity: float
    margin_used: float
    margin_available: float
    positions: Dict[str, PaperPosition]
    orders: Dict[str, PaperOrder]
    trade_history: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    status: PortfolioStatus
    created_at: float
    last_updated: float

@dataclass
class RiskMetrics:
    """Risk assessment metrics"""
    portfolio_id: str
    timestamp: float
    total_exposure: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    var_95: float  # Value at Risk 95%
    beta: float
    correlation_metrics: Dict[str, float]
    concentration_risk: float
    leverage_ratio: float

class MarketDataSimulator:
    """Simulate realistic market data for educational purposes"""
    
    def __init__(self):
        self.base_prices = {
            "BTC/USDT": 50000.0,
            "ETH/USDT": 3000.0,
            "AAPL": 150.0,
            "GOOGL": 2500.0,
            "SPY": 400.0
        }
        self.price_history = {symbol: deque(maxlen=1000) for symbol in self.base_prices}
        self.volatility = {symbol: 0.02 for symbol in self.base_prices}
        
    def generate_market_data(self, symbol: str) -> MarketData:
        """Generate realistic market data"""
        if symbol not in self.base_prices:
            # Default for unknown symbols
            self.base_prices[symbol] = 100.0
            self.price_history[symbol] = deque(maxlen=1000)
            self.volatility[symbol] = 0.03
        
        # Generate price movement using geometric Brownian motion
        current_price = self.base_prices[symbol]
        vol = self.volatility[symbol]
        
        # Random walk with drift
        drift = random.uniform(-0.001, 0.001)  # Small drift
        shock = random.gauss(0, vol)
        
        new_price = current_price * (1 + drift + shock)
        new_price = max(new_price, current_price * 0.95)  # Floor at 5% drop
        new_price = min(new_price, current_price * 1.05)  # Ceiling at 5% gain
        
        self.base_prices[symbol] = new_price
        self.price_history[symbol].append(new_price)
        
        # Calculate derived data
        bid = new_price * 0.999
        ask = new_price * 1.001
        volume = random.uniform(100000, 1000000)
        
        # Calculate 24h change
        if len(self.price_history[symbol]) > 1:
            old_price = self.price_history[symbol][0] if len(self.price_history[symbol]) == 1000 else current_price
            change = new_price - old_price
            change_percent = (change / old_price) * 100
        else:
            change = 0
            change_percent = 0
        
        return MarketData(
            symbol=symbol,
            timestamp=time.time(),
            bid=bid,
            ask=ask,
            last=new_price,
            volume=volume,
            high=new_price * 1.02,
            low=new_price * 0.98,
            change=change,
            change_percent=change_percent
        )

class RiskManager:
    """Risk management for paper trading"""
    
    def __init__(self):
        self.max_position_size = 0.1  # 10% of portfolio per position
        self.max_leverage = 2.0
        self.max_drawdown = 0.20  # 20% maximum drawdown
        self.max_concentration = 0.30  # 30% max in single asset
        
    def assess_order_risk(self, portfolio: VirtualPortfolio, 
                         order: PaperOrder, 
                         current_price: float) -> Tuple[bool, List[str]]:
        """Assess risk of placing an order"""
        risk_issues = []
        
        # Calculate order value
        order_value = order.quantity * current_price
        
        # Check position size limit
        position_size_ratio = order_value / portfolio.equity
        if position_size_ratio > self.max_position_size:
            risk_issues.append(f"Position size {position_size_ratio:.1%} exceeds limit {self.max_position_size:.1%}")
        
        # Check available balance
        if order.side == 'buy' and order_value > portfolio.current_balance:
            risk_issues.append("Insufficient balance for order")
        
        # Check concentration risk
        existing_exposure = sum(
            pos.quantity * pos.current_price 
            for pos in portfolio.positions.values()
            if pos.symbol == order.symbol
        )
        
        total_exposure = existing_exposure + order_value
        concentration_ratio = total_exposure / portfolio.equity
        
        if concentration_ratio > self.max_concentration:
            risk_issues.append(f"Concentration risk {concentration_ratio:.1%} exceeds limit {self.max_concentration:.1%}")
        
        # Check leverage
        total_exposure_all = sum(
            pos.quantity * pos.current_price 
            for pos in portfolio.positions.values()
        ) + order_value
        
        leverage_ratio = total_exposure_all / portfolio.equity
        if leverage_ratio > self.max_leverage:
            risk_issues.append(f"Leverage {leverage_ratio:.1f}x exceeds limit {self.max_leverage:.1f}x")
        
        can_place_order = len(risk_issues) == 0
        return can_place_order, risk_issues
    
    def calculate_risk_metrics(self, portfolio: VirtualPortfolio) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        if not portfolio.trade_history:
            return RiskMetrics(
                portfolio_id=portfolio.portfolio_id,
                timestamp=time.time(),
                total_exposure=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                var_95=0.0,
                beta=0.0,
                correlation_metrics={},
                concentration_risk=0.0,
                leverage_ratio=1.0
            )
        
        # Calculate total exposure
        total_exposure = sum(
            pos.quantity * pos.current_price 
            for pos in portfolio.positions.values()
        )
        
        # Calculate drawdown from trade history
        equity_curve = [trade.get('portfolio_equity', portfolio.initial_balance) 
                       for trade in portfolio.trade_history]
        peak = portfolio.initial_balance
        max_drawdown = 0.0
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate concentration risk
        if portfolio.equity > 0:
            concentration_risk = max(
                (pos.quantity * pos.current_price / portfolio.equity)
                for pos in portfolio.positions.values()
            ) if portfolio.positions else 0.0
        else:
            concentration_risk = 0.0
        
        # Calculate leverage
        leverage_ratio = total_exposure / portfolio.equity if portfolio.equity > 0 else 1.0
        
        return RiskMetrics(
            portfolio_id=portfolio.portfolio_id,
            timestamp=time.time(),
            total_exposure=total_exposure,
            max_drawdown=max_drawdown,
            sharpe_ratio=portfolio.performance_metrics.get('sharpe_ratio', 0.0),
            sortino_ratio=portfolio.performance_metrics.get('sortino_ratio', 0.0),
            var_95=total_exposure * 0.05,  # Simplified VaR calculation
            beta=1.0,  # Simplified beta
            correlation_metrics={},
            concentration_risk=concentration_risk,
            leverage_ratio=leverage_ratio
        )

class PaperTradingEngine:
    """
    Main Paper Trading Engine
    Simulates real trading with virtual portfolios and real market data
    """
    
    def __init__(self):
        self.logger = logging.getLogger('PaperTrading')
        
        # Core components
        self.market_simulator = MarketDataSimulator()
        self.risk_manager = RiskManager() 
        self.trust_system = ProgressiveTrustSystem()
        self.vault = QuantumSecurityVault()
        
        # Portfolio management
        self.portfolios: Dict[str, VirtualPortfolio] = {}
        self.active_orders: Dict[str, PaperOrder] = {}
        self.market_data_cache: Dict[str, MarketData] = {}
        
        # Engine state
        self.is_running = False
        self.update_interval = 1.0  # 1 second market data updates
        self.order_fill_probability = 0.95  # 95% of orders get filled
        
        # Educational mode
        self.educational_mode = CONFIG.trading_mode == TradingMode.OBSERVER
        
    async def create_portfolio(self, user_id: str, 
                              initial_balance: float = 10000.0) -> str:
        """Create new virtual portfolio for paper trading"""
        portfolio_id = f"PAPER_{user_id}_{int(time.time())}"
        
        # Check user trust level
        user_status = self.trust_system.get_user_status(user_id)
        if user_status.get("error"):
            # Register user if not found
            self.trust_system.register_user(user_id)
            user_status = self.trust_system.get_user_status(user_id)
        
        # Adjust initial balance based on trust level
        trust_level = user_status.get("current_level", "OBSERVER")
        if trust_level == "OBSERVER":
            initial_balance = min(initial_balance, 10000.0)  # $10k limit for observers
        elif trust_level.startswith("PAPER"):
            initial_balance = min(initial_balance, 50000.0)  # $50k limit for paper traders
        
        portfolio = VirtualPortfolio(
            portfolio_id=portfolio_id,
            user_id=user_id,
            initial_balance=initial_balance,
            current_balance=initial_balance,
            equity=initial_balance,
            margin_used=0.0,
            margin_available=initial_balance,
            positions={},
            orders={},
            trade_history=[],
            performance_metrics={
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 1.0
            },
            status=PortfolioStatus.ACTIVE,
            created_at=time.time(),
            last_updated=time.time()
        )
        
        self.portfolios[portfolio_id] = portfolio
        
        self.logger.info(f"Created paper trading portfolio: {portfolio_id}")
        print(f"📊 Paper Trading Portfolio Created")
        print(f"   Portfolio ID: {portfolio_id}")
        print(f"   User: {user_id}")
        print(f"   Initial Balance: ${initial_balance:,.2f}")
        print(f"   Trust Level: {trust_level}")
        
        return portfolio_id
    
    async def place_order(self, portfolio_id: str, 
                         order_request: OrderRequest) -> Optional[str]:
        """Place paper trading order"""
        if portfolio_id not in self.portfolios:
            self.logger.error(f"Portfolio not found: {portfolio_id}")
            return None
        
        portfolio = self.portfolios[portfolio_id]
        
        if portfolio.status != PortfolioStatus.ACTIVE:
            self.logger.warning(f"Portfolio {portfolio_id} not active")
            return None
        
        # Force paper trading in educational mode
        if self.educational_mode or not order_request.paper_trading:
            order_request.paper_trading = True
        
        # Get current market data
        market_data = await self.get_market_data(order_request.symbol)
        if not market_data:
            self.logger.error(f"No market data for {order_request.symbol}")
            return None
        
        # Create paper order
        order_id = f"ORDER_{portfolio_id}_{int(time.time())}_{random.randint(100, 999)}"
        
        order = PaperOrder(
            order_id=order_id,
            symbol=order_request.symbol,
            side=order_request.side,
            quantity=order_request.amount,
            order_type=order_request.order_type,
            price=order_request.price,
            status=OrderStatus.PENDING,
            created_at=time.time()
        )
        
        # Risk assessment
        current_price = market_data.last
        can_place, risk_issues = self.risk_manager.assess_order_risk(
            portfolio, order, current_price
        )
        
        if not can_place:
            order.status = OrderStatus.REJECTED
            self.logger.warning(f"Order rejected: {', '.join(risk_issues)}")
            print(f"❌ Order Rejected: {order_id}")
            for issue in risk_issues:
                print(f"   • {issue}")
            return None
        
        # Add to active orders
        self.active_orders[order_id] = order
        portfolio.orders[order_id] = order
        
        self.logger.info(f"Paper order placed: {order_id}")
        print(f"📝 Paper Order Placed: {order_id}")
        print(f"   Symbol: {order.symbol}")
        print(f"   Side: {order.side.upper()}")
        print(f"   Quantity: {order.quantity}")
        print(f"   Type: {order.order_type}")
        print(f"   Price: ${order.price or 'MARKET'}")
        
        # Try to fill immediately for market orders
        if order.order_type == 'market':
            await self._try_fill_order(order_id)
        
        return order_id
    
    async def _try_fill_order(self, order_id: str) -> bool:
        """Try to fill an order"""
        if order_id not in self.active_orders:
            return False
        
        order = self.active_orders[order_id]
        portfolio = self.portfolios[order.order_id.split('_')[1] + '_' + order.order_id.split('_')[2]]
        
        # Get current market data
        market_data = await self.get_market_data(order.symbol)
        if not market_data:
            return False
        
        # Determine fill conditions
        should_fill = False
        fill_price = None
        
        if order.order_type == 'market':
            should_fill = random.random() < self.order_fill_probability
            fill_price = market_data.ask if order.side == 'buy' else market_data.bid
            
        elif order.order_type == 'limit':
            if order.side == 'buy' and order.price >= market_data.ask:
                should_fill = True
                fill_price = order.price
            elif order.side == 'sell' and order.price <= market_data.bid:
                should_fill = True
                fill_price = order.price
        
        if not should_fill:
            return False
        
        # Fill the order
        order.status = OrderStatus.FILLED
        order.filled_at = time.time()
        order.filled_quantity = order.quantity
        order.filled_price = fill_price
        order.fees = order.quantity * fill_price * 0.001  # 0.1% fee
        
        # Update portfolio
        await self._update_portfolio_from_fill(portfolio, order)
        
        # Remove from active orders
        del self.active_orders[order_id]
        
        self.logger.info(f"Order filled: {order_id} at ${fill_price}")
        print(f"✅ Order Filled: {order_id}")
        print(f"   Price: ${fill_price:.2f}")
        print(f"   Fees: ${order.fees:.2f}")
        
        return True
    
    async def _update_portfolio_from_fill(self, portfolio: VirtualPortfolio, 
                                        order: PaperOrder):
        """Update portfolio after order fill"""
        fill_value = order.filled_quantity * order.filled_price
        
        if order.side == 'buy':
            # Buying - decrease cash, increase position
            portfolio.current_balance -= (fill_value + order.fees)
            
            if order.symbol in portfolio.positions:
                # Add to existing position
                pos = portfolio.positions[order.symbol]
                total_quantity = pos.quantity + order.filled_quantity
                weighted_price = ((pos.quantity * pos.entry_price) + 
                                (order.filled_quantity * order.filled_price)) / total_quantity
                pos.quantity = total_quantity
                pos.entry_price = weighted_price
            else:
                # Create new position
                portfolio.positions[order.symbol] = PaperPosition(
                    symbol=order.symbol,
                    side='long',
                    quantity=order.filled_quantity,
                    entry_price=order.filled_price,
                    current_price=order.filled_price,
                    entry_time=time.time(),
                    unrealized_pnl=0.0,
                    realized_pnl=0.0
                )
        
        else:  # sell
            # Selling - increase cash, decrease position
            portfolio.current_balance += (fill_value - order.fees)
            
            if order.symbol in portfolio.positions:
                pos = portfolio.positions[order.symbol]
                
                if pos.quantity >= order.filled_quantity:
                    # Calculate realized P&L
                    realized_pnl = ((order.filled_price - pos.entry_price) * 
                                  order.filled_quantity)
                    pos.realized_pnl += realized_pnl
                    pos.quantity -= order.filled_quantity
                    
                    # Remove position if fully closed
                    if pos.quantity == 0:
                        del portfolio.positions[order.symbol]
                
                else:
                    self.logger.warning(f"Insufficient position to sell: {order.symbol}")
        
        # Record trade
        trade_record = {
            'timestamp': time.time(),
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side,
            'quantity': order.filled_quantity,
            'price': order.filled_price,
            'fees': order.fees,
            'portfolio_balance': portfolio.current_balance,
            'portfolio_equity': await self._calculate_portfolio_equity(portfolio)
        }
        
        portfolio.trade_history.append(trade_record)
        portfolio.last_updated = time.time()
        
        # Update performance metrics
        await self._update_performance_metrics(portfolio)
        
        # Record trade with trust system
        trade_result = {
            'successful': realized_pnl > 0 if 'realized_pnl' in locals() else True,
            'pnl': realized_pnl if 'realized_pnl' in locals() else 0.0
        }
        self.trust_system.record_trade(portfolio.user_id, trade_result)
    
    async def _calculate_portfolio_equity(self, portfolio: VirtualPortfolio) -> float:
        """Calculate total portfolio equity"""
        equity = portfolio.current_balance
        
        for position in portfolio.positions.values():
            # Get current market price
            market_data = await self.get_market_data(position.symbol)
            if market_data:
                position.current_price = market_data.last
                position_value = position.quantity * position.current_price
                equity += position_value
                
                # Update unrealized P&L
                position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
        
        portfolio.equity = equity
        return equity
    
    async def _update_performance_metrics(self, portfolio: VirtualPortfolio):
        """Update portfolio performance metrics"""
        if not portfolio.trade_history:
            return
        
        # Calculate total return
        total_return = ((portfolio.equity - portfolio.initial_balance) / 
                       portfolio.initial_balance) * 100
        
        # Calculate win rate
        profitable_trades = sum(1 for trade in portfolio.trade_history 
                               if trade.get('realized_pnl', 0) > 0)
        total_trades = len([t for t in portfolio.trade_history if 'realized_pnl' in t])
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate profit factor
        total_profits = sum(max(trade.get('realized_pnl', 0), 0) 
                           for trade in portfolio.trade_history)
        total_losses = sum(abs(min(trade.get('realized_pnl', 0), 0)) 
                          for trade in portfolio.trade_history)
        profit_factor = (total_profits / total_losses) if total_losses > 0 else 1.0
        
        # Update metrics
        portfolio.performance_metrics.update({
            'total_return': total_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'total_pnl': total_profits - total_losses
        })
    
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get market data for symbol"""
        # Check cache first
        if symbol in self.market_data_cache:
            cached_data = self.market_data_cache[symbol]
            if time.time() - cached_data.timestamp < 5.0:  # 5 second cache
                return cached_data
        
        # Generate new market data
        market_data = self.market_simulator.generate_market_data(symbol)
        self.market_data_cache[symbol] = market_data
        
        return market_data
    
    async def start_engine(self):
        """Start paper trading engine"""
        self.is_running = True
        self.logger.info("Paper trading engine started")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._market_data_updater()),
            asyncio.create_task(self._order_processor()),
            asyncio.create_task(self._portfolio_updater())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.info("Paper trading engine stopped")
    
    async def stop_engine(self):
        """Stop paper trading engine"""
        self.is_running = False
    
    async def _market_data_updater(self):
        """Update market data periodically"""
        while self.is_running:
            try:
                # Update cached market data
                for symbol in list(self.market_data_cache.keys()):
                    self.market_data_cache[symbol] = self.market_simulator.generate_market_data(symbol)
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Market data update error: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _order_processor(self):
        """Process pending orders"""
        while self.is_running:
            try:
                # Process all pending orders
                pending_orders = [
                    order_id for order_id, order in self.active_orders.items()
                    if order.status == OrderStatus.PENDING
                ]
                
                for order_id in pending_orders:
                    await self._try_fill_order(order_id)
                
                await asyncio.sleep(0.5)  # Check orders every 500ms
                
            except Exception as e:
                self.logger.error(f"Order processing error: {e}")
                await asyncio.sleep(1.0)
    
    async def _portfolio_updater(self):
        """Update portfolio valuations"""
        while self.is_running:
            try:
                for portfolio in self.portfolios.values():
                    if portfolio.status == PortfolioStatus.ACTIVE:
                        await self._calculate_portfolio_equity(portfolio)
                        await self._update_performance_metrics(portfolio)
                
                await asyncio.sleep(5.0)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Portfolio update error: {e}")
                await asyncio.sleep(5.0)
    
    def get_portfolio_status(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive portfolio status"""
        if portfolio_id not in self.portfolios:
            return None
        
        portfolio = self.portfolios[portfolio_id]
        risk_metrics = self.risk_manager.calculate_risk_metrics(portfolio)
        
        return {
            'portfolio_id': portfolio_id,
            'user_id': portfolio.user_id,
            'status': portfolio.status.value,
            'balance': portfolio.current_balance,
            'equity': portfolio.equity,
            'initial_balance': portfolio.initial_balance,
            'total_return': portfolio.performance_metrics.get('total_return', 0.0),
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'side': pos.side
                }
                for symbol, pos in portfolio.positions.items()
            },
            'active_orders': len([o for o in portfolio.orders.values() 
                                if o.status == OrderStatus.PENDING]),
            'performance_metrics': portfolio.performance_metrics,
            'risk_metrics': asdict(risk_metrics),
            'last_updated': portfolio.last_updated
        }

# Example usage
if __name__ == "__main__":
    async def main():
        engine = PaperTradingEngine()
        
        # Create portfolio
        portfolio_id = await engine.create_portfolio("educational_user_001", 10000.0)
        
        # Place some orders
        order1 = OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            amount=0.1,
            order_type="market",
            paper_trading=True
        )
        
        order_id = await engine.place_order(portfolio_id, order1)
        
        if order_id:
            # Wait for processing
            await asyncio.sleep(2)
            
            # Check portfolio status
            status = engine.get_portfolio_status(portfolio_id)
            print(f"Portfolio Status: {status}")
    
    asyncio.run(main())