"""
DAA Competition Engine
Neural Network Trading Strategies Competition
Educational Trading System with Performance-Based Rewards
"""

import asyncio
import json
import time
import random
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging
import numpy as np
from collections import defaultdict, deque

from ..config.system_config import CONFIG, TradingMode
from ..connectors.universal_broker import MarketData, OrderRequest, OrderResponse
from ..observatory.dashboard import PerformanceMetric, TradingDecision, SyndicateType

class CompetitionType(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"  
    MONTHLY = "monthly"
    CONTINUOUS = "continuous"

class DAAStatus(Enum):
    INACTIVE = "inactive"
    TRAINING = "training"
    COMPETING = "competing"
    SUSPENDED = "suspended"
    CHAMPION = "champion"

@dataclass
class DAAAgent:
    """Decentralized Autonomous Agent for trading"""
    agent_id: str
    name: str
    strategy_type: str
    neural_network_config: Dict[str, Any]
    performance_metrics: Dict[str, float]
    status: DAAStatus
    created_at: float
    last_active: float
    total_trades: int
    successful_trades: int
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    risk_score: float
    educational_mode: bool = True

@dataclass
class CompetitionRound:
    """Competition round tracking"""
    round_id: str
    competition_type: CompetitionType
    start_time: float
    end_time: float
    participants: List[str]  # DAA agent IDs
    market_conditions: Dict[str, Any]
    results: Dict[str, Dict[str, float]]
    winner: Optional[str] = None
    completed: bool = False

@dataclass
class TournamentBracket:
    """Swiss tournament bracket system"""
    tournament_id: str
    total_rounds: int
    current_round: int
    participants: List[str]
    pairings: Dict[int, List[Tuple[str, str]]]
    scores: Dict[str, int]
    results: Dict[int, Dict[str, str]]  # round -> {participant: result}

class ShapleyValueCalculator:
    """Shapley value calculation for fair reward distribution"""
    
    @staticmethod
    def calculate_contributions(agents: List[DAAAgent], 
                              market_performance: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate Shapley values for agent contributions
        Simplified version for educational system
        """
        if not agents:
            return {}
        
        n = len(agents)
        shapley_values = {}
        
        for agent in agents:
            agent_id = agent.agent_id
            
            # Base contribution from individual performance
            individual_performance = market_performance.get(agent_id, 0.0)
            
            # Marginal contribution (simplified calculation)
            marginal_contrib = individual_performance * (1.0 / n)
            
            # Adjust based on agent performance metrics
            performance_multiplier = (
                agent.sharpe_ratio * 0.3 +
                agent.win_rate / 100.0 * 0.25 +
                (1.0 - min(agent.max_drawdown / 20.0, 1.0)) * 0.25 +
                agent.profit_factor * 0.2
            )
            
            shapley_values[agent_id] = marginal_contrib * performance_multiplier
        
        # Normalize to sum to 1.0
        total_value = sum(shapley_values.values())
        if total_value > 0:
            shapley_values = {
                agent_id: value / total_value 
                for agent_id, value in shapley_values.items()
            }
        
        return shapley_values
    
    @staticmethod
    def distribute_rewards(shapley_values: Dict[str, float],
                          total_rewards: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Distribute rewards based on Shapley values"""
        agent_rewards = defaultdict(dict)
        
        for reward_type, total_amount in total_rewards.items():
            for agent_id, shapley_value in shapley_values.items():
                agent_rewards[agent_id][reward_type] = total_amount * shapley_value
        
        return dict(agent_rewards)

class FitnessEvaluator:
    """Fitness evaluation for DAA agents"""
    
    def __init__(self):
        self.fitness_weights = CONFIG.fitness_weights
    
    def calculate_fitness(self, agent: DAAAgent, 
                         market_conditions: Dict[str, Any] = None) -> float:
        """
        Calculate comprehensive fitness score for DAA agent
        
        Fitness = α*alpha + β*sharpe + γ*(1-drawdown) + δ*win_rate + ε*profit_factor
        """
        if agent.total_trades == 0:
            return 0.0
        
        # Normalize metrics to 0-1 range
        alpha = min(max(agent.total_pnl / 1000.0, 0), 1)  # Normalize PnL
        sharpe = min(max(agent.sharpe_ratio / 3.0, 0), 1)  # Normalize Sharpe
        drawdown_score = max(0, 1 - agent.max_drawdown / 20.0)  # 20% max drawdown
        win_rate_score = agent.win_rate / 100.0
        profit_factor_score = min(agent.profit_factor / 2.0, 1)  # Cap at 2.0
        
        # Calculate weighted fitness
        fitness = (
            self.fitness_weights['alpha'] * alpha +
            self.fitness_weights['sharpe'] * sharpe +
            self.fitness_weights['max_drawdown'] * drawdown_score +
            self.fitness_weights['win_rate'] * win_rate_score +
            self.fitness_weights['profit_factor'] * profit_factor_score
        )
        
        # Educational bonus for consistent performance
        if agent.educational_mode and agent.total_trades >= 50:
            consistency_bonus = min(agent.win_rate / 100.0, 0.1)  # Max 10% bonus
            fitness += consistency_bonus
        
        return min(fitness, 1.0)  # Cap at 1.0
    
    def rank_agents(self, agents: List[DAAAgent]) -> List[Tuple[str, float]]:
        """Rank agents by fitness score"""
        agent_fitness = [
            (agent.agent_id, self.calculate_fitness(agent))
            for agent in agents
        ]
        
        return sorted(agent_fitness, key=lambda x: x[1], reverse=True)

class DACompetitionEngine:
    """
    Main DAA Competition Engine
    Manages competitions, tournaments, and performance tracking
    """
    
    def __init__(self):
        self.logger = logging.getLogger('DACompetition')
        
        # Competition state
        self.agents: Dict[str, DAAAgent] = {}
        self.active_competitions: Dict[str, CompetitionRound] = {}
        self.tournaments: Dict[str, TournamentBracket] = {}
        self.competition_history: List[CompetitionRound] = []
        
        # Competition settings
        self.max_agents_per_competition = 16
        self.competition_duration = {
            CompetitionType.DAILY: 24 * 3600,      # 24 hours
            CompetitionType.WEEKLY: 7 * 24 * 3600,  # 7 days  
            CompetitionType.MONTHLY: 30 * 24 * 3600, # 30 days
            CompetitionType.CONTINUOUS: float('inf')  # Ongoing
        }
        
        # Evaluation components
        self.fitness_evaluator = FitnessEvaluator()
        self.shapley_calculator = ShapleyValueCalculator()
        
        # Educational mode enforcement
        self.educational_mode = CONFIG.trading_mode == TradingMode.OBSERVER
        
    def register_agent(self, agent_config: Dict[str, Any]) -> str:
        """Register new DAA agent in competition system"""
        agent_id = f"DAA_{int(time.time())}_{random.randint(1000, 9999)}"
        
        agent = DAAAgent(
            agent_id=agent_id,
            name=agent_config.get('name', f'Agent_{agent_id[-4:]}'),
            strategy_type=agent_config.get('strategy_type', 'neural_network'),
            neural_network_config=agent_config.get('neural_config', {}),
            performance_metrics={},
            status=DAAStatus.INACTIVE,
            created_at=time.time(),
            last_active=time.time(),
            total_trades=0,
            successful_trades=0,
            total_pnl=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            win_rate=0.0,
            profit_factor=1.0,
            risk_score=0.5,
            educational_mode=self.educational_mode
        )
        
        self.agents[agent_id] = agent
        
        self.logger.info(f"Registered DAA agent: {agent_id}")
        print(f"🤖 DAA Agent Registered: {agent.name}")
        print(f"   ID: {agent_id}")
        print(f"   Strategy: {agent.strategy_type}")
        print(f"   Educational Mode: {agent.educational_mode}")
        
        return agent_id
    
    def create_competition(self, competition_type: CompetitionType,
                          participant_limit: int = None) -> str:
        """Create new competition round"""
        competition_id = f"COMP_{competition_type.value}_{int(time.time())}"
        
        # Select participants
        eligible_agents = [
            agent_id for agent_id, agent in self.agents.items()
            if agent.status in [DAAStatus.COMPETING, DAAStatus.INACTIVE]
        ]
        
        limit = participant_limit or self.max_agents_per_competition
        participants = eligible_agents[:limit]
        
        if len(participants) < 2:
            self.logger.warning("Not enough agents for competition")
            return ""
        
        # Create competition round
        competition = CompetitionRound(
            round_id=competition_id,
            competition_type=competition_type,
            start_time=time.time(),
            end_time=time.time() + self.competition_duration[competition_type],
            participants=participants,
            market_conditions=self._get_current_market_conditions(),
            results={}
        )
        
        self.active_competitions[competition_id] = competition
        
        # Update agent status
        for agent_id in participants:
            self.agents[agent_id].status = DAAStatus.COMPETING
        
        self.logger.info(f"Created {competition_type.value} competition: {competition_id}")
        print(f"🏆 Competition Created: {competition_type.value.upper()}")
        print(f"   Competition ID: {competition_id}")
        print(f"   Participants: {len(participants)}")
        print(f"   Duration: {self.competition_duration[competition_type] / 3600:.1f} hours")
        
        return competition_id
    
    def _get_current_market_conditions(self) -> Dict[str, Any]:
        """Get current market conditions for competition context"""
        return {
            "timestamp": time.time(),
            "volatility": random.uniform(0.1, 0.8),  # Simulated
            "trend": random.choice(["bullish", "bearish", "sideways"]),
            "volume": random.uniform(0.5, 2.0),  # Relative volume
            "sentiment": random.uniform(-1.0, 1.0)  # Market sentiment
        }
    
    def simulate_trading_round(self, competition_id: str, 
                             duration_minutes: int = 60) -> Dict[str, Any]:
        """
        Simulate trading round for educational system
        In production, this would connect to real neural networks
        """
        if competition_id not in self.active_competitions:
            return {"error": "Competition not found"}
        
        competition = self.active_competitions[competition_id]
        
        self.logger.info(f"Simulating trading round for {competition_id}")
        
        # Simulate trading performance for each participant
        round_results = {}
        
        for agent_id in competition.participants:
            agent = self.agents[agent_id]
            
            # Simulate neural network trading decisions
            performance = self._simulate_agent_performance(agent, duration_minutes)
            
            # Update agent statistics
            self._update_agent_performance(agent, performance)
            
            round_results[agent_id] = performance
        
        # Store round results
        competition.results[int(time.time())] = round_results
        
        return {
            "competition_id": competition_id,
            "round_duration": duration_minutes,
            "participants": len(competition.participants),
            "results": round_results
        }
    
    def _simulate_agent_performance(self, agent: DAAAgent, 
                                  duration_minutes: int) -> Dict[str, float]:
        """
        Simulate agent trading performance
        Educational version with realistic but simulated results
        """
        # Base performance influenced by agent's historical performance
        base_win_rate = max(0.3, agent.win_rate / 100.0) if agent.total_trades > 0 else 0.5
        
        # Simulate trades during the period
        num_trades = max(1, int(duration_minutes / 30))  # ~1 trade per 30 minutes
        
        trades_won = 0
        total_pnl = 0.0
        max_loss = 0.0
        
        for _ in range(num_trades):
            # Simulate trade outcome
            win_probability = base_win_rate + random.uniform(-0.1, 0.1)
            trade_successful = random.random() < win_probability
            
            if trade_successful:
                trades_won += 1
                trade_pnl = random.uniform(10, 50)  # Educational profit range
            else:
                trade_pnl = -random.uniform(5, 30)  # Educational loss range
                max_loss = max(max_loss, abs(trade_pnl))
            
            total_pnl += trade_pnl
        
        # Calculate performance metrics
        win_rate = (trades_won / num_trades) * 100 if num_trades > 0 else 0
        profit_factor = max(0.1, total_pnl / max(abs(total_pnl - sum([10] * trades_won)), 1))
        sharpe_ratio = total_pnl / max(max_loss, 1) if max_loss > 0 else 0
        
        return {
            "trades": num_trades,
            "wins": trades_won,
            "pnl": total_pnl,
            "max_loss": max_loss,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "timestamp": time.time()
        }
    
    def _update_agent_performance(self, agent: DAAAgent, 
                                performance: Dict[str, float]):
        """Update agent performance statistics"""
        # Update cumulative statistics
        agent.total_trades += performance["trades"]
        agent.successful_trades += performance["wins"]
        agent.total_pnl += performance["pnl"]
        agent.max_drawdown = max(agent.max_drawdown, performance["max_loss"])
        agent.last_active = time.time()
        
        # Recalculate derived metrics
        if agent.total_trades > 0:
            agent.win_rate = (agent.successful_trades / agent.total_trades) * 100
            agent.profit_factor = max(0.1, agent.total_pnl / max(agent.total_trades, 1))
            agent.sharpe_ratio = agent.total_pnl / max(agent.max_drawdown, 1)
    
    def evaluate_competition(self, competition_id: str) -> Dict[str, Any]:
        """Evaluate competition results and distribute rewards"""
        if competition_id not in self.active_competitions:
            return {"error": "Competition not found"}
        
        competition = self.active_competitions[competition_id]
        
        # Get all participants and their current performance
        participants = [self.agents[agent_id] for agent_id in competition.participants]
        
        # Calculate fitness scores
        fitness_rankings = self.fitness_evaluator.rank_agents(participants)
        
        # Calculate Shapley values for fair reward distribution
        market_performance = {
            agent_id: agent.total_pnl 
            for agent_id, agent in self.agents.items()
            if agent_id in competition.participants
        }
        
        shapley_values = self.shapley_calculator.calculate_contributions(
            participants, market_performance
        )
        
        # Define rewards (educational system)
        total_rewards = {
            "computational_resources": 100.0,  # Resource allocation units
            "data_access_priority": 50.0,     # Priority access points
            "capital_allocation": 1000.0      # Educational capital units
        }
        
        # Distribute rewards
        agent_rewards = self.shapley_calculator.distribute_rewards(
            shapley_values, total_rewards
        )
        
        # Determine winner
        winner_id, winner_fitness = fitness_rankings[0] if fitness_rankings else (None, 0)
        competition.winner = winner_id
        competition.completed = True
        
        # Update agent status
        for agent_id in competition.participants:
            if agent_id == winner_id:
                self.agents[agent_id].status = DAAStatus.CHAMPION
            else:
                self.agents[agent_id].status = DAAStatus.INACTIVE
        
        # Store in history
        self.competition_history.append(competition)
        del self.active_competitions[competition_id]
        
        evaluation_results = {
            "competition_id": competition_id,
            "competition_type": competition.competition_type.value,
            "winner": winner_id,
            "winner_fitness": winner_fitness,
            "fitness_rankings": fitness_rankings,
            "shapley_values": shapley_values,
            "reward_distribution": agent_rewards,
            "participants": len(competition.participants),
            "completed_at": time.time()
        }
        
        self.logger.info(f"Competition {competition_id} completed")
        self._print_competition_results(evaluation_results)
        
        return evaluation_results
    
    def _print_competition_results(self, results: Dict[str, Any]):
        """Print educational competition results"""
        print(f"\n🏆 COMPETITION RESULTS")
        print(f"Competition: {results['competition_id']}")
        print(f"Type: {results['competition_type'].upper()}")
        print(f"Winner: {results['winner']} (Fitness: {results['winner_fitness']:.3f})")
        print(f"\n📊 LEADERBOARD:")
        
        for i, (agent_id, fitness) in enumerate(results['fitness_rankings'][:5], 1):
            agent = self.agents[agent_id]
            print(f"  {i}. {agent.name} (ID: {agent_id[-6:]})")
            print(f"     Fitness: {fitness:.3f} | Win Rate: {agent.win_rate:.1f}% | PnL: ${agent.total_pnl:.2f}")
        
        print(f"\n💰 REWARD DISTRIBUTION (Top 3):")
        for i, (agent_id, fitness) in enumerate(results['fitness_rankings'][:3], 1):
            rewards = results['reward_distribution'].get(agent_id, {})
            print(f"  {i}. Agent {agent_id[-6:]}:")
            for reward_type, amount in rewards.items():
                print(f"     {reward_type}: {amount:.2f}")
        
        print("─" * 60)
    
    def get_leaderboard(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get current agent leaderboard"""
        all_agents = list(self.agents.values())
        fitness_rankings = self.fitness_evaluator.rank_agents(all_agents)
        
        leaderboard = []
        for i, (agent_id, fitness) in enumerate(fitness_rankings[:limit]):
            agent = self.agents[agent_id]
            leaderboard.append({
                "rank": i + 1,
                "agent_id": agent_id,
                "name": agent.name,
                "fitness": fitness,
                "win_rate": agent.win_rate,
                "total_pnl": agent.total_pnl,
                "sharpe_ratio": agent.sharpe_ratio,
                "total_trades": agent.total_trades,
                "status": agent.status.value
            })
        
        return leaderboard
    
    def get_competition_stats(self) -> Dict[str, Any]:
        """Get comprehensive competition statistics"""
        return {
            "total_agents": len(self.agents),
            "active_competitions": len(self.active_competitions),
            "completed_competitions": len(self.competition_history),
            "agents_by_status": {
                status.value: len([
                    a for a in self.agents.values() 
                    if a.status == status
                ])
                for status in DAAStatus
            },
            "average_fitness": np.mean([
                self.fitness_evaluator.calculate_fitness(agent)
                for agent in self.agents.values()
            ]) if self.agents else 0.0,
            "total_trades": sum(agent.total_trades for agent in self.agents.values()),
            "educational_mode": self.educational_mode
        }

# Example usage and testing
if __name__ == "__main__":
    async def main():
        competition_engine = DACompetitionEngine()
        
        # Register some DAA agents
        agent_configs = [
            {"name": "AlphaTrader", "strategy_type": "momentum"},
            {"name": "BetaBot", "strategy_type": "mean_reversion"},
            {"name": "GammaAI", "strategy_type": "neural_network"},
            {"name": "DeltaSystem", "strategy_type": "arbitrage"}
        ]
        
        agent_ids = []
        for config in agent_configs:
            agent_id = competition_engine.register_agent(config)
            agent_ids.append(agent_id)
        
        # Create daily competition
        comp_id = competition_engine.create_competition(CompetitionType.DAILY)
        
        if comp_id:
            # Simulate trading rounds
            for round_num in range(3):
                print(f"\n🔄 Simulating trading round {round_num + 1}")
                results = competition_engine.simulate_trading_round(comp_id, 60)
                print(f"Round results: {len(results.get('results', {}))} agents participated")
                
                # Wait between rounds (simulated)
                await asyncio.sleep(1)
            
            # Evaluate competition
            evaluation = competition_engine.evaluate_competition(comp_id)
            
            # Show leaderboard
            leaderboard = competition_engine.get_leaderboard()
            
            # Show stats
            stats = competition_engine.get_competition_stats()
            print(f"\n📈 COMPETITION STATISTICS:")
            print(f"Total Agents: {stats['total_agents']}")
            print(f"Total Trades: {stats['total_trades']}")
            print(f"Average Fitness: {stats['average_fitness']:.3f}")
    
    asyncio.run(main())