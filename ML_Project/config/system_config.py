"""
Enhanced Universal Trading Syndicate - System Configuration
Educational Trading System with Quantum-Encrypted Security
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class TradingMode(Enum):
    OBSERVER = "observer"  # Days 1-7: No API keys, observation only
    PAPER = "paper"        # Days 8-30: Virtual portfolio with real prices
    MICRO_LIVE = "micro"   # Day 31+: Small capital with oversight

class SecurityLevel(Enum):
    EDUCATIONAL = "educational"  # Default: No live trading
    PAPER_ONLY = "paper_only"    # Paper trading enabled
    MICRO_LIVE = "micro_live"    # Small capital live trading
    
@dataclass
class VaultConfig:
    """Quantum-encrypted security vault configuration"""
    vault_path: str = "./vault/data"
    encryption_algorithm: str = "AES-256-GCM"
    key_rotation_hours: int = 24
    audit_log_enabled: bool = True
    emergency_lockdown: bool = True
    multi_signature_required: bool = True

@dataclass
class ProgressiveTrustConfig:
    """Progressive trust system configuration"""
    observer_days: int = 7
    paper_trading_days: int = 23  # Days 8-30
    micro_live_threshold: int = 31
    max_micro_capital: float = 1000.0  # Maximum for micro-live mode
    
@dataclass
class SystemConfig:
    """Main system configuration"""
    # Core Settings
    trading_mode: TradingMode = TradingMode.OBSERVER
    security_level: SecurityLevel = SecurityLevel.EDUCATIONAL
    
    # Educational Constraints
    live_trading_enabled: bool = False  # CRITICAL: Default to False
    require_explicit_consent: bool = True
    max_position_size: float = 100.0  # Educational limit
    
    # Vault Configuration
    vault: VaultConfig = VaultConfig()
    
    # Progressive Trust
    trust: ProgressiveTrustConfig = ProgressiveTrustConfig()
    
    # Performance Targets
    target_latency_ms: int = 100
    target_throughput_tps: int = 500
    target_uptime: float = 99.9
    target_win_rate: float = 60.0
    
    # DAA Competition
    tournament_types: List[str] = None
    reward_algorithm: str = "shapley_value"
    fitness_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.tournament_types is None:
            self.tournament_types = ["daily", "weekly", "monthly"]
        
        if self.fitness_weights is None:
            self.fitness_weights = {
                "alpha": 0.3,
                "sharpe": 0.25,
                "max_drawdown": 0.25,
                "win_rate": 0.1,
                "profit_factor": 0.1
            }
    
    def enforce_educational_constraints(self) -> bool:
        """Enforce strict educational trading constraints"""
        if self.live_trading_enabled and not self.require_explicit_consent:
            raise ValueError("Live trading requires explicit user consent")
        
        if self.security_level == SecurityLevel.EDUCATIONAL:
            self.live_trading_enabled = False
            
        return True
    
    def validate_config(self) -> bool:
        """Validate system configuration for safety"""
        # Critical safety checks
        if self.live_trading_enabled and self.security_level == SecurityLevel.EDUCATIONAL:
            raise ValueError("Cannot enable live trading in educational mode")
        
        if self.max_position_size > 10000:
            raise ValueError("Position size exceeds educational limits")
        
        return True

# Global configuration instance
CONFIG = SystemConfig()

# Ensure educational constraints are enforced
CONFIG.enforce_educational_constraints()
CONFIG.validate_config()