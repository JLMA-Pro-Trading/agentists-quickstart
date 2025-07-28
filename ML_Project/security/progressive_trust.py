"""
Progressive Trust System
Observer → Paper → Micro-Live Trading Progression
Educational Trading System with Safety Mechanisms
"""

import time
import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging
import hashlib

from ..config.system_config import CONFIG, TradingMode, SecurityLevel
from ..vault.quantum_vault import QuantumSecurityVault

class TrustLevel(Enum):
    OBSERVER = 0      # Days 1-7: No trading, observation only
    PAPER_BEGINNER = 1  # Days 8-14: Basic paper trading
    PAPER_ADVANCED = 2  # Days 15-30: Advanced paper trading
    MICRO_LIVE_1 = 3   # Days 31+: $100-500 live trading
    MICRO_LIVE_2 = 4   # After 60 days: $500-1000 live trading
    RESTRICTED = -1    # Restricted due to poor performance/violations

class PerformanceThreshold(Enum):
    SHARPE_RATIO = 1.0
    WIN_RATE = 55.0  # 55% minimum
    MAX_DRAWDOWN = 10.0  # 10% maximum
    PROFIT_FACTOR = 1.2
    CONSECUTIVE_LOSSES = 5  # Maximum consecutive losses

@dataclass
class TrustCheckpoint:
    """Trust level progression checkpoint"""
    level: TrustLevel
    min_days: int
    max_capital: float
    performance_requirements: Dict[str, float]
    educational_requirements: List[str]
    safety_checks: List[str]

@dataclass
class UserProgress:
    """User progress tracking"""
    user_id: str
    start_date: float
    current_level: TrustLevel
    days_at_current_level: int
    total_trades: int
    successful_trades: int
    total_pnl: float
    max_drawdown: float
    consecutive_losses: int
    violations: List[Dict[str, Any]]
    performance_history: List[Dict[str, float]]
    last_assessment: float

@dataclass
class SafetyViolation:
    """Safety rule violation tracking"""
    timestamp: float
    violation_type: str
    severity: str  # 'minor', 'major', 'critical'
    description: str
    automated_response: str
    user_notified: bool = False

class ProgressiveTrustSystem:
    """
    Progressive Trust System for Educational Trading
    
    Manages user progression through trust levels with comprehensive
    safety checks and educational requirements.
    """
    
    def __init__(self, data_path: str = "./ML_Project/security/trust_data"):
        self.data_path = data_path
        self.vault = QuantumSecurityVault()
        self.logger = logging.getLogger('ProgressiveTrust')
        
        # Trust level definitions
        self.trust_checkpoints = self._define_trust_checkpoints()
        
        # User progress tracking
        self.user_progress: Dict[str, UserProgress] = {}
        self.safety_violations: List[SafetyViolation] = []
        
        # Safety settings
        self.emergency_stop_enabled = True
        self.require_explicit_consent = True
        self.educational_mode_only = True  # Default to educational mode
        
        # Initialize system
        self._ensure_data_directory()
        self._load_user_progress()
        
    def _define_trust_checkpoints(self) -> Dict[TrustLevel, TrustCheckpoint]:
        """Define trust level progression checkpoints"""
        return {
            TrustLevel.OBSERVER: TrustCheckpoint(
                level=TrustLevel.OBSERVER,
                min_days=0,
                max_capital=0.0,
                performance_requirements={},
                educational_requirements=[
                    "Complete trading basics tutorial",
                    "Understand risk management principles",
                    "Learn about market volatility"
                ],
                safety_checks=[
                    "System observation completed",
                    "Risk assessment understanding verified"
                ]
            ),
            TrustLevel.PAPER_BEGINNER: TrustCheckpoint(
                level=TrustLevel.PAPER_BEGINNER,
                min_days=7,
                max_capital=10000.0,  # Paper trading balance
                performance_requirements={
                    "min_trades": 10,
                    "observation_hours": 40
                },
                educational_requirements=[
                    "Complete paper trading tutorial",
                    "Demonstrate order placement",
                    "Understand position sizing"
                ],
                safety_checks=[
                    "Observer period completed successfully",
                    "Basic trading knowledge verified"
                ]
            ),
            TrustLevel.PAPER_ADVANCED: TrustCheckpoint(
                level=TrustLevel.PAPER_ADVANCED,
                min_days=14,
                max_capital=50000.0,  # Advanced paper balance
                performance_requirements={
                    "min_trades": 50,
                    "win_rate": 45.0,
                    "max_drawdown": 15.0,
                    "sharpe_ratio": 0.5
                },
                educational_requirements=[
                    "Advanced risk management course",
                    "Portfolio diversification understanding",
                    "Market analysis techniques"
                ],
                safety_checks=[
                    "Consistent paper trading performance",
                    "Risk management demonstrated"
                ]
            ),
            TrustLevel.MICRO_LIVE_1: TrustCheckpoint(
                level=TrustLevel.MICRO_LIVE_1,
                min_days=30,
                max_capital=500.0,  # $500 maximum live trading
                performance_requirements={
                    "min_trades": 100,
                    "win_rate": 55.0,
                    "max_drawdown": 10.0,
                    "sharpe_ratio": 1.0,
                    "profit_factor": 1.2
                },
                educational_requirements=[
                    "Live trading risks assessment",
                    "Capital preservation strategies",
                    "Emergency procedures training"
                ],
                safety_checks=[
                    "Excellent paper trading record",
                    "Risk management mastery",
                    "Explicit consent for live trading"
                ]
            ),
            TrustLevel.MICRO_LIVE_2: TrustCheckpoint(
                level=TrustLevel.MICRO_LIVE_2,
                min_days=60,
                max_capital=1000.0,  # $1000 maximum
                performance_requirements={
                    "min_trades": 200,
                    "win_rate": 60.0,
                    "max_drawdown": 8.0,
                    "sharpe_ratio": 1.5,
                    "profit_factor": 1.5,
                    "consecutive_profitable_months": 2
                },
                educational_requirements=[
                    "Advanced portfolio management",
                    "Multi-platform trading strategies",
                    "Professional risk assessment"
                ],
                safety_checks=[
                    "Proven micro-live performance",
                    "Zero major violations",
                    "Continued educational compliance"
                ]
            )
        }
    
    def _ensure_data_directory(self):
        """Create trust data directory"""
        os.makedirs(self.data_path, exist_ok=True)
        
        # Set secure permissions
        try:
            os.chmod(self.data_path, 0o700)
        except OSError:
            pass
    
    def _load_user_progress(self):
        """Load user progress from storage"""
        progress_file = os.path.join(self.data_path, "user_progress.json")
        
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    data = json.load(f)
                
                for user_id, progress_data in data.items():
                    self.user_progress[user_id] = UserProgress(**progress_data)
                    
            except Exception as e:
                self.logger.error(f"Error loading user progress: {e}")
    
    def _save_user_progress(self):
        """Save user progress to storage"""
        progress_file = os.path.join(self.data_path, "user_progress.json")
        
        try:
            data = {
                user_id: asdict(progress) 
                for user_id, progress in self.user_progress.items()
            }
            
            with open(progress_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving user progress: {e}")
    
    def register_user(self, user_id: str) -> bool:
        """Register new user in progressive trust system"""
        if user_id in self.user_progress:
            self.logger.warning(f"User {user_id} already registered")
            return False
        
        # Create initial user progress
        self.user_progress[user_id] = UserProgress(
            user_id=user_id,
            start_date=time.time(),
            current_level=TrustLevel.OBSERVER,
            days_at_current_level=0,
            total_trades=0,
            successful_trades=0,
            total_pnl=0.0,
            max_drawdown=0.0,
            consecutive_losses=0,
            violations=[],
            performance_history=[],
            last_assessment=time.time()
        )
        
        self._save_user_progress()
        
        self.logger.info(f"User {user_id} registered in Observer mode")
        print(f"✅ Welcome to the Progressive Trust System!")
        print(f"   Starting in Observer Mode (7-day learning period)")
        print(f"   No trading yet - focus on learning and observation")
        
        return True
    
    def assess_user_progress(self, user_id: str) -> Dict[str, Any]:
        """Assess user progress for potential level advancement"""
        if user_id not in self.user_progress:
            return {"error": "User not registered"}
        
        progress = self.user_progress[user_id]
        current_checkpoint = self.trust_checkpoints[progress.current_level]
        
        # Calculate days since registration
        days_since_start = (time.time() - progress.start_date) / (24 * 3600)
        progress.days_at_current_level = int(days_since_start)
        
        # Check if user can advance to next level
        can_advance, next_level, reasons = self._check_advancement_eligibility(progress)
        
        # Prepare assessment report
        assessment = {
            "user_id": user_id,
            "current_level": progress.current_level.name,
            "days_at_level": progress.days_at_current_level,
            "days_since_start": int(days_since_start),
            "can_advance": can_advance,
            "next_level": next_level.name if next_level else None,
            "advancement_reasons": reasons,
            "performance_summary": self._calculate_performance_metrics(progress),
            "educational_status": self._check_educational_requirements(progress),
            "safety_status": self._check_safety_requirements(progress),
            "violations": len(progress.violations),
            "last_assessment": progress.last_assessment
        }
        
        progress.last_assessment = time.time()
        self._save_user_progress()
        
        return assessment
    
    def _check_advancement_eligibility(self, progress: UserProgress) -> Tuple[bool, Optional[TrustLevel], List[str]]:
        """Check if user is eligible for advancement"""
        current_level = progress.current_level
        reasons = []
        
        # Special case: RESTRICTED users need manual review
        if current_level == TrustLevel.RESTRICTED:
            return False, None, ["Manual review required for restricted users"]
        
        # Find next level
        level_order = [
            TrustLevel.OBSERVER,
            TrustLevel.PAPER_BEGINNER,
            TrustLevel.PAPER_ADVANCED,
            TrustLevel.MICRO_LIVE_1,
            TrustLevel.MICRO_LIVE_2
        ]
        
        try:
            current_index = level_order.index(current_level)
            if current_index >= len(level_order) - 1:
                return False, None, ["Already at maximum trust level"]
            
            next_level = level_order[current_index + 1]
        except ValueError:
            return False, None, ["Invalid current trust level"]
        
        next_checkpoint = self.trust_checkpoints[next_level]
        
        # Check minimum days requirement
        if progress.days_at_current_level < next_checkpoint.min_days:
            days_remaining = next_checkpoint.min_days - progress.days_at_current_level
            reasons.append(f"Need {days_remaining} more days at current level")
        
        # Check performance requirements
        performance_metrics = self._calculate_performance_metrics(progress)
        for metric, required_value in next_checkpoint.performance_requirements.items():
            if metric not in performance_metrics:
                reasons.append(f"Missing performance metric: {metric}")
                continue
            
            actual_value = performance_metrics[metric]
            
            # Check if requirement is met
            if metric in ['win_rate', 'sharpe_ratio', 'profit_factor']:
                if actual_value < required_value:
                    reasons.append(f"{metric}: {actual_value:.2f} < {required_value} (required)")
            elif metric == 'max_drawdown':
                if actual_value > required_value:
                    reasons.append(f"{metric}: {actual_value:.2f}% > {required_value}% (maximum)")
            elif metric in ['min_trades']:
                if actual_value < required_value:
                    reasons.append(f"{metric}: {actual_value} < {required_value} (required)")
        
        # Check educational requirements (simplified for MVP)
        educational_complete = len(next_checkpoint.educational_requirements) == 0  # Assume complete for MVP
        if not educational_complete:
            reasons.append("Educational requirements not completed")
        
        # Check safety requirements
        if len(progress.violations) > 0:
            recent_violations = [
                v for v in progress.violations 
                if time.time() - v['timestamp'] < 30 * 24 * 3600  # 30 days
            ]
            if recent_violations:
                reasons.append(f"{len(recent_violations)} recent safety violations")
        
        # Special check for live trading levels
        if next_level in [TrustLevel.MICRO_LIVE_1, TrustLevel.MICRO_LIVE_2]:
            if self.educational_mode_only:
                reasons.append("System in educational mode only - live trading disabled")
            
            if not self.require_explicit_consent:
                reasons.append("Explicit consent required for live trading")
        
        can_advance = len(reasons) == 0
        return can_advance, next_level, reasons
    
    def _calculate_performance_metrics(self, progress: UserProgress) -> Dict[str, float]:
        """Calculate user performance metrics"""
        if progress.total_trades == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "sharpe_ratio": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0
            }
        
        win_rate = (progress.successful_trades / progress.total_trades) * 100
        
        # Simplified metrics for MVP
        sharpe_ratio = max(0, win_rate / 50.0)  # Simplified calculation
        profit_factor = max(0.1, progress.total_pnl / max(1, progress.total_trades))
        
        return {
            "total_trades": progress.total_trades,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe_ratio,
            "profit_factor": profit_factor,
            "max_drawdown": progress.max_drawdown,
            "total_pnl": progress.total_pnl
        }
    
    def _check_educational_requirements(self, progress: UserProgress) -> Dict[str, Any]:
        """Check educational requirements status"""
        current_checkpoint = self.trust_checkpoints[progress.current_level]
        
        # Simplified for MVP - assume educational requirements are met
        return {
            "total_requirements": len(current_checkpoint.educational_requirements),
            "completed": len(current_checkpoint.educational_requirements),  # Assume all complete
            "completion_rate": 100.0,
            "outstanding": []
        }
    
    def _check_safety_requirements(self, progress: UserProgress) -> Dict[str, Any]:
        """Check safety requirements status"""
        recent_violations = [
            v for v in progress.violations 
            if time.time() - v['timestamp'] < 30 * 24 * 3600  # 30 days
        ]
        
        return {
            "total_violations": len(progress.violations),
            "recent_violations": len(recent_violations),
            "safety_score": max(0, 100 - len(recent_violations) * 10),
            "emergency_stops": 0,  # Track emergency stops
            "compliance_status": "good" if len(recent_violations) == 0 else "warning"
        }
    
    def advance_user_level(self, user_id: str, explicit_consent: bool = False) -> bool:
        """Advance user to next trust level"""
        assessment = self.assess_user_progress(user_id)
        
        if not assessment.get("can_advance", False):
            reasons = assessment.get("advancement_reasons", [])
            print(f"❌ Cannot advance user {user_id}:")
            for reason in reasons:
                print(f"   • {reason}")
            return False
        
        progress = self.user_progress[user_id]
        next_level_name = assessment["next_level"]
        next_level = TrustLevel[next_level_name]
        
        # Special consent check for live trading
        if next_level in [TrustLevel.MICRO_LIVE_1, TrustLevel.MICRO_LIVE_2]:
            if not explicit_consent:
                print(f"⚠️  Advancing to {next_level_name} requires explicit consent")
                print(f"   This enables live trading with real money")
                print(f"   Call advance_user_level(user_id, explicit_consent=True)")
                return False
            
            if self.educational_mode_only:
                print(f"🚫 Live trading disabled in educational mode")
                return False
        
        # Advance user
        old_level = progress.current_level
        progress.current_level = next_level
        progress.days_at_current_level = 0
        
        self._save_user_progress()
        
        # Log advancement
        self.logger.info(f"User {user_id} advanced from {old_level.name} to {next_level.name}")
        
        # Notify user
        checkpoint = self.trust_checkpoints[next_level]
        print(f"🎉 Congratulations! Advanced to {next_level_name}")
        print(f"   Maximum capital: ${checkpoint.max_capital:,.2f}")
        print(f"   Paper trading: {next_level.value < 3}")
        
        if next_level in [TrustLevel.MICRO_LIVE_1, TrustLevel.MICRO_LIVE_2]:
            print(f"⚠️  LIVE TRADING ENABLED with ${checkpoint.max_capital} limit")
            print(f"   Use this responsibly and within educational constraints")
        
        return True
    
    def record_trade(self, user_id: str, trade_result: Dict[str, Any]) -> bool:
        """Record trade result for user"""
        if user_id not in self.user_progress:
            return False
        
        progress = self.user_progress[user_id]
        
        # Update trade statistics
        progress.total_trades += 1
        
        if trade_result.get("successful", False):
            progress.successful_trades += 1
            progress.consecutive_losses = 0
        else:
            progress.consecutive_losses += 1
        
        # Update P&L
        pnl = trade_result.get("pnl", 0.0)
        progress.total_pnl += pnl
        
        # Update max drawdown
        if pnl < 0:
            progress.max_drawdown = max(progress.max_drawdown, abs(pnl))
        
        # Check for safety violations
        self._check_trade_safety_violations(user_id, trade_result)
        
        self._save_user_progress()
        return True
    
    def _check_trade_safety_violations(self, user_id: str, trade_result: Dict[str, Any]):
        """Check for safety violations in trade"""
        progress = self.user_progress[user_id]
        
        # Check consecutive losses
        if progress.consecutive_losses >= PerformanceThreshold.CONSECUTIVE_LOSSES.value:
            violation = {
                "timestamp": time.time(),
                "type": "consecutive_losses",
                "severity": "major",
                "description": f"Exceeded {PerformanceThreshold.CONSECUTIVE_LOSSES.value} consecutive losses",
                "automated_response": "Temporary trading suspension recommended"
            }
            progress.violations.append(violation)
        
        # Check drawdown limits
        if progress.max_drawdown > PerformanceThreshold.MAX_DRAWDOWN.value:
            violation = {
                "timestamp": time.time(),
                "type": "max_drawdown",
                "severity": "critical",
                "description": f"Drawdown {progress.max_drawdown:.2f}% exceeds limit {PerformanceThreshold.MAX_DRAWDOWN.value}%",
                "automated_response": "Emergency stop triggered"
            }
            progress.violations.append(violation)
            
            # Trigger emergency stop
            self._trigger_emergency_stop(user_id, "Max drawdown exceeded")
    
    def _trigger_emergency_stop(self, user_id: str, reason: str):
        """Trigger emergency stop for user"""
        if user_id in self.user_progress:
            progress = self.user_progress[user_id]
            old_level = progress.current_level
            progress.current_level = TrustLevel.RESTRICTED
            
            self.logger.critical(f"EMERGENCY STOP: User {user_id} restricted - {reason}")
            print(f"🚨 EMERGENCY STOP ACTIVATED")
            print(f"   User: {user_id}")
            print(f"   Reason: {reason}")
            print(f"   Status: Moved from {old_level.name} to RESTRICTED")
            print(f"   All trading suspended pending review")
    
    def get_user_status(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user status"""
        if user_id not in self.user_progress:
            return {"error": "User not registered"}
        
        progress = self.user_progress[user_id]
        checkpoint = self.trust_checkpoints[progress.current_level]
        
        return {
            "user_id": user_id,
            "current_level": progress.current_level.name,
            "max_capital": checkpoint.max_capital,
            "days_at_level": progress.days_at_current_level,
            "total_trades": progress.total_trades,
            "performance": self._calculate_performance_metrics(progress),
            "violations": len(progress.violations),
            "status": "active" if progress.current_level != TrustLevel.RESTRICTED else "restricted",
            "can_trade": progress.current_level != TrustLevel.RESTRICTED and not self.emergency_stop_enabled,
            "paper_trading_only": progress.current_level.value < 3 or self.educational_mode_only,
            "last_assessment": progress.last_assessment
        }

# Example usage
if __name__ == "__main__":
    trust_system = ProgressiveTrustSystem()
    
    # Register new user
    user_id = "educational_user_001"
    trust_system.register_user(user_id)
    
    # Check initial status
    status = trust_system.get_user_status(user_id)
    print(f"Initial Status: {status}")
    
    # Simulate some time passing and trading
    progress = trust_system.user_progress[user_id]
    progress.days_at_current_level = 8  # Simulate 8 days
    
    # Assess progress
    assessment = trust_system.assess_user_progress(user_id)
    print(f"Assessment: {assessment}")
    
    # Try to advance
    success = trust_system.advance_user_level(user_id)
    print(f"Advancement successful: {success}")