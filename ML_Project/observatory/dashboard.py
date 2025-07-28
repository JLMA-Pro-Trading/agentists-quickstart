"""
Observatory Dashboard - Real-Time Transparency Engine
Educational Trading System Monitoring and Analytics
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import logging
from collections import defaultdict, deque
import threading

from ..config.system_config import CONFIG, TradingMode
from ..connectors.universal_broker import UniversalBrokerManager, MarketData, OrderResponse
from ..vault.quantum_vault import QuantumSecurityVault

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class SyndicateType(Enum):
    OPERATIONS = "operations"
    INTELLIGENCE = "intelligence"
    EVOLUTION = "evolution"
    AUDIT = "audit"

@dataclass
class TradingDecision:
    """Represents a trading decision with full transparency"""
    timestamp: float
    syndicate: SyndicateType
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0-100
    reasoning: str
    market_context: Dict[str, Any]
    alternatives_considered: List[str]
    risk_assessment: Dict[str, float]
    expected_outcome: Dict[str, float]
    paper_trading: bool = True

@dataclass
class PerformanceMetric:
    """Performance tracking for DAAs and strategies"""
    entity_id: str  # DAA or strategy ID
    metric_name: str
    value: float
    timestamp: float
    benchmark_comparison: Optional[float] = None

@dataclass
class SystemAlert:
    """System alerts and notifications"""
    timestamp: float
    level: AlertLevel
    component: str
    message: str
    details: Dict[str, Any]
    acknowledged: bool = False

@dataclass
class ResourceUtilization:
    """System resource monitoring"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    network_latency: float
    active_connections: int
    processing_queue_size: int

class RealTimeMonitor:
    """Real-time system monitoring with circular buffers"""
    
    def __init__(self, max_buffer_size: int = 1000):
        self.max_buffer_size = max_buffer_size
        
        # Circular buffers for real-time data
        self.trading_decisions = deque(maxlen=max_buffer_size)
        self.performance_metrics = deque(maxlen=max_buffer_size)
        self.system_alerts = deque(maxlen=max_buffer_size)
        self.resource_utilization = deque(maxlen=max_buffer_size)
        self.market_data = deque(maxlen=max_buffer_size)
        
        # Real-time statistics
        self.stats = {
            'total_decisions': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_alerts': 0,
            'system_uptime': time.time(),
            'last_update': time.time()
        }
        
        # Event subscribers
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
    def add_trading_decision(self, decision: TradingDecision):
        """Add new trading decision to monitor"""
        self.trading_decisions.append(decision)
        self.stats['total_decisions'] += 1
        self.stats['last_update'] = time.time()
        
        # Notify subscribers
        self._notify_subscribers('trading_decision', decision)
    
    def add_performance_metric(self, metric: PerformanceMetric):
        """Add performance metric"""
        self.performance_metrics.append(metric)
        self.stats['last_update'] = time.time()
        
        self._notify_subscribers('performance_metric', metric)
    
    def add_system_alert(self, alert: SystemAlert):
        """Add system alert"""
        self.system_alerts.append(alert)
        self.stats['total_alerts'] += 1
        self.stats['last_update'] = time.time()
        
        # Log critical alerts
        if alert.level == AlertLevel.CRITICAL:
            logging.critical(f"CRITICAL ALERT: {alert.message}")
        
        self._notify_subscribers('system_alert', alert)
    
    def add_resource_data(self, resource: ResourceUtilization):
        """Add resource utilization data"""
        self.resource_utilization.append(resource)
        self.stats['last_update'] = time.time()
        
        self._notify_subscribers('resource_data', resource)
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to real-time events"""
        self.subscribers[event_type].append(callback)
    
    def _notify_subscribers(self, event_type: str, data: Any):
        """Notify all subscribers of an event"""
        for callback in self.subscribers[event_type]:
            try:
                callback(data)
            except Exception as e:
                logging.error(f"Error in subscriber callback: {e}")
    
    def get_recent_decisions(self, limit: int = 50) -> List[TradingDecision]:
        """Get recent trading decisions"""
        return list(self.trading_decisions)[-limit:]
    
    def get_performance_summary(self, time_window: int = 3600) -> Dict[str, Any]:
        """Get performance summary for time window (seconds)"""
        cutoff_time = time.time() - time_window
        
        recent_metrics = [
            m for m in self.performance_metrics 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        # Aggregate metrics by entity
        entity_performance = defaultdict(list)
        for metric in recent_metrics:
            entity_performance[metric.entity_id].append(metric)
        
        summary = {}
        for entity_id, metrics in entity_performance.items():
            summary[entity_id] = {
                'total_metrics': len(metrics),
                'latest_update': max(m.timestamp for m in metrics),
                'metrics': {m.metric_name: m.value for m in metrics}
            }
        
        return summary

class ObservatoryDashboard:
    """
    Main Observatory Dashboard
    Real-time transparency engine for educational trading system
    """
    
    def __init__(self):
        self.monitor = RealTimeMonitor()
        self.broker = UniversalBrokerManager()
        self.vault = QuantumSecurityVault()
        self.logger = logging.getLogger('Observatory')
        
        # Dashboard state
        self.is_running = False
        self.update_interval = 1.0  # 1 second updates
        self.last_health_check = time.time()
        
        # Educational mode status
        self.educational_mode = CONFIG.trading_mode == TradingMode.OBSERVER
        
        # Competition leaderboard data
        self.leaderboard = {}
        self.tournament_data = {}
        
        # Setup event handlers
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Setup real-time event handlers"""
        # Subscribe to monitor events
        self.monitor.subscribe('system_alert', self._handle_alert)
        self.monitor.subscribe('trading_decision', self._handle_decision)
        self.monitor.subscribe('performance_metric', self._handle_performance)
    
    def _handle_alert(self, alert: SystemAlert):
        """Handle system alerts"""
        if alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
            self.logger.error(f"ALERT: {alert.message}")
            
            # In educational mode, provide learning context
            if self.educational_mode:
                self._provide_educational_context(alert)
    
    def _handle_decision(self, decision: TradingDecision):
        """Handle trading decisions"""
        # Log decision with full transparency
        self.logger.info(f"Trading Decision: {decision.action} {decision.symbol}")
        self.logger.info(f"Reasoning: {decision.reasoning}")
        self.logger.info(f"Confidence: {decision.confidence}%")
        
        if self.educational_mode:
            self._explain_decision_educational(decision)
    
    def _handle_performance(self, metric: PerformanceMetric):
        """Handle performance metrics"""
        # Update leaderboard if it's a competition metric
        if metric.metric_name.startswith('competition_'):
            self._update_leaderboard(metric)
    
    def _provide_educational_context(self, alert: SystemAlert):
        """Provide educational context for alerts"""
        educational_context = {
            AlertLevel.WARNING: "⚠️  Learning Opportunity: This warning helps understand market risks",
            AlertLevel.ERROR: "📚 Educational Note: This error demonstrates real trading challenges",
            AlertLevel.CRITICAL: "🎓 Critical Learning: This shows why risk management is essential"
        }
        
        context = educational_context.get(alert.level, "")
        if context:
            print(f"{context}")
            print(f"Alert: {alert.message}")
    
    def _explain_decision_educational(self, decision: TradingDecision):
        """Explain trading decision for educational purposes"""
        print(f"\n📊 EDUCATIONAL TRADING DECISION ANALYSIS")
        print(f"Symbol: {decision.symbol}")
        print(f"Action: {decision.action.upper()}")
        print(f"Confidence: {decision.confidence}%")
        print(f"Reasoning: {decision.reasoning}")
        print(f"Alternatives Considered: {', '.join(decision.alternatives_considered)}")
        print(f"Risk Assessment: {decision.risk_assessment}")
        print(f"Expected Outcome: {decision.expected_outcome}")
        print(f"Mode: {'PAPER TRADING' if decision.paper_trading else 'LIVE TRADING'}")
        print("─" * 60)
    
    def _update_leaderboard(self, metric: PerformanceMetric):
        """Update competition leaderboard"""
        if metric.entity_id not in self.leaderboard:
            self.leaderboard[metric.entity_id] = {}
        
        self.leaderboard[metric.entity_id][metric.metric_name] = {
            'value': metric.value,
            'timestamp': metric.timestamp,
            'benchmark': metric.benchmark_comparison
        }
    
    async def start_monitoring(self):
        """Start real-time monitoring"""
        self.is_running = True
        self.logger.info("🔭 Observatory Dashboard started")
        
        # Start monitoring tasks
        monitor_tasks = [
            asyncio.create_task(self._monitor_system_health()),
            asyncio.create_task(self._monitor_trading_activity()),
            asyncio.create_task(self._monitor_performance()),
            asyncio.create_task(self._update_dashboard())
        ]
        
        try:
            await asyncio.gather(*monitor_tasks)
        except asyncio.CancelledError:
            self.logger.info("Monitoring tasks cancelled")
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        self.is_running = False
        self.logger.info("🔭 Observatory Dashboard stopped")
    
    async def _monitor_system_health(self):
        """Monitor system health metrics"""
        while self.is_running:
            try:
                # Simulate system resource monitoring
                resource_data = ResourceUtilization(
                    timestamp=time.time(),
                    cpu_usage=50.0,  # Simulated
                    memory_usage=1024.0,  # MB
                    network_latency=10.0,  # ms
                    active_connections=len(self.broker.get_connected_platforms()),
                    processing_queue_size=0
                )
                
                self.monitor.add_resource_data(resource_data)
                
                # Health check alerts
                if resource_data.cpu_usage > 80:
                    alert = SystemAlert(
                        timestamp=time.time(),
                        level=AlertLevel.WARNING,
                        component="system",
                        message=f"High CPU usage: {resource_data.cpu_usage}%",
                        details={"cpu_usage": resource_data.cpu_usage}
                    )
                    self.monitor.add_system_alert(alert)
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _monitor_trading_activity(self):
        """Monitor trading activity across all platforms"""
        while self.is_running:
            try:
                # Check for new trading decisions
                # This would integrate with the DAA competition system
                
                # Simulate educational trading decision
                if self.educational_mode:
                    decision = TradingDecision(
                        timestamp=time.time(),
                        syndicate=SyndicateType.OPERATIONS,
                        symbol="BTC/USDT",
                        action="hold",
                        confidence=75.0,
                        reasoning="Market volatility suggests waiting for clearer trend",
                        market_context={"price": 50000, "volume": 1000000},
                        alternatives_considered=["buy", "sell"],
                        risk_assessment={"max_loss": 2.0, "probability": 0.1},
                        expected_outcome={"profit": 5.0, "timeline": "24h"},
                        paper_trading=True
                    )
                    
                    self.monitor.add_trading_decision(decision)
                
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Trading activity monitoring error: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _monitor_performance(self):
        """Monitor DAA and strategy performance"""
        while self.is_running:
            try:
                # Simulate performance metrics for educational system
                entities = ["DAA_001", "DAA_002", "Strategy_Alpha", "Strategy_Beta"]
                
                for entity in entities:
                    # Generate educational performance metrics
                    metrics = [
                        PerformanceMetric(
                            entity_id=entity,
                            metric_name="sharpe_ratio",
                            value=1.2 + (hash(entity) % 100) / 100,
                            timestamp=time.time()
                        ),
                        PerformanceMetric(
                            entity_id=entity,
                            metric_name="win_rate",
                            value=60.0 + (hash(entity) % 20),
                            timestamp=time.time()
                        )
                    ]
                    
                    for metric in metrics:
                        self.monitor.add_performance_metric(metric)
                
                await asyncio.sleep(30.0)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _update_dashboard(self):
        """Update dashboard display"""
        while self.is_running:
            try:
                # Generate dashboard summary
                dashboard_data = self.get_dashboard_data()
                
                # In educational mode, provide periodic summaries
                if self.educational_mode and int(time.time()) % 60 == 0:
                    self._print_educational_summary(dashboard_data)
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Dashboard update error: {e}")
                await asyncio.sleep(self.update_interval)
    
    def _print_educational_summary(self, data: Dict[str, Any]):
        """Print educational summary"""
        print(f"\n🔭 OBSERVATORY EDUCATIONAL SUMMARY")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Mode: {CONFIG.trading_mode.value.upper()}")
        print(f"Connected Platforms: {len(data.get('connected_platforms', []))}")
        print(f"Recent Decisions: {len(data.get('recent_decisions', []))}")
        print(f"Active Alerts: {len(data.get('active_alerts', []))}")
        print(f"System Uptime: {data.get('uptime_hours', 0):.1f} hours")
        print("─" * 50)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        uptime_seconds = time.time() - self.monitor.stats['system_uptime']
        
        return {
            'timestamp': time.time(),
            'educational_mode': self.educational_mode,
            'trading_mode': CONFIG.trading_mode.value,
            'uptime_hours': uptime_seconds / 3600,
            'connected_platforms': self.broker.get_connected_platforms(),
            'platform_status': self.broker.get_platform_status(),
            'recent_decisions': self.monitor.get_recent_decisions(10),
            'performance_summary': self.monitor.get_performance_summary(),
            'active_alerts': [
                alert for alert in self.monitor.system_alerts 
                if not alert.acknowledged
            ],
            'system_stats': self.monitor.stats,
            'leaderboard': self.leaderboard,
            'resource_utilization': list(self.monitor.resource_utilization)[-10:],
            'vault_status': self.vault.get_vault_status()
        }
    
    def get_syndicate_activity(self, syndicate: SyndicateType) -> Dict[str, Any]:
        """Get activity for specific syndicate"""
        syndicate_decisions = [
            decision for decision in self.monitor.trading_decisions
            if decision.syndicate == syndicate
        ]
        
        return {
            'syndicate': syndicate.value,
            'total_decisions': len(syndicate_decisions),
            'recent_decisions': syndicate_decisions[-10:],
            'performance_metrics': [
                metric for metric in self.monitor.performance_metrics
                if metric.entity_id.startswith(syndicate.value)
            ]
        }
    
    def get_competition_leaderboard(self) -> Dict[str, Any]:
        """Get competition leaderboard data"""
        # Sort entities by performance
        sorted_entities = sorted(
            self.leaderboard.items(),
            key=lambda x: x[1].get('sharpe_ratio', {}).get('value', 0),
            reverse=True
        )
        
        return {
            'leaderboard': sorted_entities,
            'last_update': max(
                [
                    metric.get('timestamp', 0) 
                    for entity_data in self.leaderboard.values()
                    for metric in entity_data.values()
                ] or [0]
            ),
            'total_entities': len(self.leaderboard)
        }

# Example usage
if __name__ == "__main__":
    async def main():
        dashboard = ObservatoryDashboard()
        
        try:
            # Start monitoring in educational mode
            print("🔭 Starting Observatory Dashboard in Educational Mode")
            await dashboard.start_monitoring()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down Observatory Dashboard")
            await dashboard.stop_monitoring()
    
    asyncio.run(main())