"""
vision/state_machine.py - Poker Table State Machine

Maintains complete poker table state with rule-based validation:
- Player states (active, folded, all-in)
- Card states (hole cards, community cards)
- Chip stacks and pot amounts
- Game phase tracking (preflop, flop, turn, river)
- Action history

Enforces poker rules to prevent impossible state transitions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
from copy import deepcopy

logger = logging.getLogger(__name__)


class Street(Enum):
    """Poker game phases."""
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"
    UNKNOWN = "unknown"


class PlayerAction(Enum):
    """Possible player actions."""
    UNKNOWN = "unknown"
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"
    ALL_IN = "all_in"
    BET = "bet"


@dataclass
class PlayerState:
    """State of a single player."""
    player_id: int
    position: str = ""  # BTN, SB, BB, UTG, MP, CO
    stack: float = 0.0
    bet: float = 0.0
    is_active: bool = True
    is_dealer: bool = False
    hole_cards: List[str] = field(default_factory=list)
    last_action: PlayerAction = PlayerAction.UNKNOWN
    action_history: List[PlayerAction] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "position": self.position,
            "stack": self.stack,
            "bet": self.bet,
            "is_active": self.is_active,
            "is_dealer": self.is_dealer,
            "hole_cards": self.hole_cards,
            "last_action": self.last_action.value,
            "action_history": [a.value for a in self.action_history],
        }


@dataclass
class CardState:
    """State of a card on the table."""
    card_id: int
    rank: str
    suit: str
    position: str  # "board", "hole_p1", "hole_p2", etc.
    is_visible: bool = True
    confidence: float = 1.0
    track_id: Optional[int] = None
    
    @property
    def short_name(self) -> str:
        suit_symbol = {"spades": "s", "hearts": "h", "diamonds": "d", "clubs": "c"}
        return f"{self.rank}{suit_symbol.get(self.suit, '?')}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "card_id": self.card_id,
            "rank": self.rank,
            "suit": self.suit,
            "short_name": self.short_name,
            "position": self.position,
            "is_visible": self.is_visible,
            "confidence": self.confidence,
            "track_id": self.track_id,
        }


@dataclass
class PokerTableState:
    """
    Complete state of the poker table.
    
    This is the main output of the vision system.
    """
    # Game state
    street: Street = Street.UNKNOWN
    pot: float = 0.0
    side_pots: List[float] = field(default_factory=list)
    current_bet: float = 0.0
    min_raise: float = 0.0
    
    # Players
    players: Dict[int, PlayerState] = field(default_factory=dict)
    dealer_position: int = -1
    active_player: int = -1
    
    # Cards
    board_cards: List[CardState] = field(default_factory=list)
    
    # Metadata
    frame_number: int = 0
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    
    # History
    previous_state: Optional['PokerTableState'] = None
    action_log: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "street": self.street.value,
            "pot": self.pot,
            "side_pots": self.side_pots,
            "current_bet": self.current_bet,
            "players": {k: v.to_dict() for k, v in self.players.items()},
            "dealer_position": self.dealer_position,
            "active_player": self.active_player,
            "board_cards": [c.to_dict() for c in self.board_cards],
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
        }
    
    def get_hole_cards(self, player_id: int) -> List[str]:
        """Get hole cards for a specific player."""
        player = self.players.get(player_id)
        if player:
            return player.hole_cards
        return []
    
    def get_board_cards(self) -> List[str]:
        """Get community cards as string list."""
        return [c.short_name for c in self.board_cards if c.is_visible]


class StateMachine:
    """
    State machine for poker table state management.
    
    Features:
    - Rule-based state validation
    - Temporal consistency enforcement
    - Rollback capability
    - Anomaly detection
    """
    
    # Valid street transitions
    VALID_TRANSITIONS = {
        Street.UNKNOWN: [Street.PREFLOP, Street.FLOP],
        Street.PREFLOP: [Street.FLOP, Street.UNKNOWN],
        Street.FLOP: [Street.TURN, Street.UNKNOWN],
        Street.TURN: [Street.RIVER, Street.UNKNOWN],
        Street.RIVER: [Street.SHOWDOWN, Street.UNKNOWN],
        Street.SHOWDOWN: [Street.PREFLOP, Street.UNKNOWN],  # New hand
    }
    
    # Maximum cards per street
    MAX_CARDS = {
        Street.PREFLOP: 0,
        Street.FLOP: 3,
        Street.TURN: 4,
        Street.RIVER: 5,
        Street.SHOWDOWN: 5,
    }
    
    def __init__(
        self,
        max_players: int = 9,
        enable_rollback: bool = True,
        history_size: int = 30,
    ):
        """
        Initialize state machine.
        
        Args:
            max_players: Maximum number of players at table
            enable_rollback: Enable state rollback on invalid transitions
            history_size: Number of states to keep in history
        """
        self.max_players = max_players
        self.enable_rollback = enable_rollback
        self.history_size = history_size
        
        # Current state
        self.current_state = PokerTableState()
        
        # State history for rollback
        self.state_history: List[PokerTableState] = []
        
        # Statistics
        self.transition_count = 0
        self.rollback_count = 0
        self.anomaly_count = 0
        
        logger.info(f"StateMachine initialized (max_players={max_players})")
    
    def update(
        self,
        fused_detections: List[Any],
        frame_number: int = 0,
    ) -> PokerTableState:
        """
        Update state based on new detections.
        
        Args:
            fused_detections: List of FusedDetection from fusion engine
            frame_number: Current frame number
            
        Returns:
            Updated PokerTableState
        """
        # Save current state for potential rollback
        if self.enable_rollback:
            self._save_state()
        
        # Create new state
        new_state = PokerTableState()
        new_state.frame_number = frame_number
        new_state.previous_state = self.current_state
        
        # Update from detections
        self._update_from_detections(new_state, fused_detections)
        
        # Validate transition
        errors = self._validate_transition(self.current_state, new_state)
        new_state.validation_errors = errors
        
        if errors:
            self.anomaly_count += len(errors)
            logger.warning(f"State transition anomalies: {errors}")
            
            if self.enable_rollback and self._is_critical_error(errors):
                self.rollback_count += 1
                logger.info("Rolling back state due to critical error")
                return self.current_state
        
        # Update confidence based on validation
        new_state.confidence = self._compute_state_confidence(new_state)
        new_state.is_valid = len(errors) == 0
        
        # Commit new state
        self.current_state = new_state
        self.transition_count += 1
        
        return self.current_state
    
    def _save_state(self) -> None:
        """Save current state to history."""
        state_copy = deepcopy(self.current_state)
        self.state_history.append(state_copy)
        
        # Trim history
        if len(self.state_history) > self.history_size:
            self.state_history = self.state_history[-self.history_size:]
    
    def _update_from_detections(
        self,
        state: PokerTableState,
        fused_detections: List[Any],
    ) -> None:
        """Update state from fused detections."""
        for det in fused_detections:
            class_name = getattr(det, 'class_name', '')
            
            if class_name == 'card':
                self._process_card_detection(state, det)
            elif class_name == 'chip':
                self._process_chip_detection(state, det)
            elif class_name == 'player':
                self._process_player_detection(state, det)
            elif class_name == 'dealer_button':
                self._process_dealer_button(state, det)
            elif class_name == 'pot':
                self._process_pot_detection(state, det)
    
    def _process_card_detection(self, state: PokerState, det: Any) -> None:
        """Process card detection."""
        rank = getattr(det, 'rank', '')
        suit = getattr(det, 'suit', '')
        confidence = getattr(det, 'fused_confidence', 0.5)
        track_id = getattr(det, 'track_id', None)
        bbox = getattr(det, 'bbox', (0, 0, 0, 0))
        
        if not rank or not suit:
            return  # Incomplete classification
        
        # Determine card position based on location
        cx, cy = det.center
        position = self._infer_card_position(cx, cy, state)
        
        # Check if card already exists
        existing = self._find_existing_card(state, rank, suit, position)
        
        if existing:
            # Update existing card
            existing.confidence = max(existing.confidence, confidence)
            existing.track_id = track_id
        else:
            # Add new card
            card = CardState(
                card_id=len(state.board_cards) + len([p for p in state.players.values()]),
                rank=rank,
                suit=suit,
                position=position,
                confidence=confidence,
                track_id=track_id,
            )
            
            if position == 'board':
                state.board_cards.append(card)
            else:
                # Hole card - assign to player
                player_id = self._get_player_from_position(position)
                if player_id and player_id in state.players:
                    state.players[player_id].hole_cards.append(card.short_name)
    
    def _infer_card_position(
        self,
        cx: float,
        cy: float,
        state: PokerTableState,
    ) -> str:
        """Infer card position from coordinates."""
        # Simplified logic - would need calibration for real table
        # Board cards are typically in center
        # Hole cards are near players
        
        # For now, assume center region is board
        return 'board'  # Default
    
    def _find_existing_card(
        self,
        state: PokerTableState,
        rank: str,
        suit: str,
        position: str,
    ) -> Optional[CardState]:
        """Find existing card matching criteria."""
        for card in state.board_cards:
            if card.rank == rank and card.suit == suit and card.position == position:
                return card
        return None
    
    def _process_chip_detection(self, state: PokerTableState, det: Any) -> None:
        """Process chip detection (estimate stacks/pot)."""
        # Would need chip value recognition for accurate amounts
        pass
    
    def _process_player_detection(self, state: PokerTableState, det: Any) -> None:
        """Process player detection."""
        track_id = getattr(det, 'track_id', None)
        confidence = getattr(det, 'fused_confidence', 0.5)
        
        if track_id is not None and track_id not in state.players:
            player = PlayerState(player_id=track_id)
            state.players[track_id] = player
    
    def _process_dealer_button(self, state: PokerTableState, det: Any) -> None:
        """Process dealer button detection."""
        track_id = getattr(det, 'track_id', None)
        if track_id is not None:
            state.dealer_position = track_id
    
    def _process_pot_detection(self, state: PokerTableState, det: Any) -> None:
        """Process pot area detection."""
        # Would need OCR for exact amount
        pass
    
    def _validate_transition(
        self,
        old_state: PokerTableState,
        new_state: PokerTableState,
    ) -> List[str]:
        """
        Validate state transition against poker rules.
        
        Returns list of validation errors (empty if valid).
        """
        errors = []
        
        # Check street transition validity
        if old_state.street != new_state.street:
            valid_next = self.VALID_TRANSITIONS.get(old_state.street, [])
            if new_state.street not in valid_next:
                errors.append(
                    f"Invalid street transition: {old_state.street.value} -> {new_state.street.value}"
                )
        
        # Check card count consistency
        max_cards = self.MAX_CARDS.get(new_state.street, 5)
        if len(new_state.board_cards) > max_cards:
            errors.append(
                f"Too many board cards for {new_state.street.value}: "
                f"{len(new_state.board_cards)} > {max_cards}"
            )
        
        # Check for duplicate cards
        all_cards = set()
        for card in new_state.board_cards:
            card_key = f"{card.rank}{card.suit}"
            if card_key in all_cards:
                errors.append(f"Duplicate card detected: {card_key}")
            all_cards.add(card_key)
        
        # Check player count
        if len(new_state.players) > self.max_players:
            errors.append(f"Too many players: {len(new_state.players)} > {self.max_players}")
        
        # Check stack consistency (should not increase without reason)
        for pid, player in new_state.players.items():
            old_player = old_state.players.get(pid)
            if old_player and player.stack > old_player.stack:
                # Stack increased - only valid if we missed some info
                if player.stack > old_player.stack * 1.1:  # 10% tolerance
                    errors.append(
                        f"Player {pid} stack increased unexpectedly: "
                        f"{old_player.stack} -> {player.stack}"
                    )
        
        return errors
    
    def _is_critical_error(self, errors: List[str]) -> bool:
        """Determine if errors are critical enough to trigger rollback."""
        critical_keywords = ["duplicate card", "invalid street"]
        return any(kw in err.lower() for err in errors for kw in critical_keywords)
    
    def _compute_state_confidence(self, state: PokerTableState) -> float:
        """Compute overall state confidence."""
        if not state.board_cards:
            return 0.5
        
        # Average card confidence
        card_confs = [c.confidence for c in state.board_cards if c.confidence > 0]
        if not card_confs:
            return 0.5
        
        avg_card_conf = sum(card_confs) / len(card_confs)
        
        # Penalty for validation errors
        error_penalty = len(state.validation_errors) * 0.1
        
        return max(0.0, min(1.0, avg_card_conf - error_penalty))
    
    def get_street_from_card_count(self, card_count: int) -> Street:
        """Determine street from number of board cards."""
        if card_count == 0:
            return Street.PREFLOP
        elif card_count <= 3:
            return Street.FLOP
        elif card_count == 4:
            return Street.TURN
        elif card_count >= 5:
            return Street.RIVER
        return Street.UNKNOWN
    
    def rollback(self, steps: int = 1) -> Optional[PokerTableState]:
        """
        Rollback to previous state.
        
        Args:
            steps: Number of states to rollback
            
        Returns:
            Rolled back state or None if not available
        """
        if not self.state_history:
            return None
        
        steps = min(steps, len(self.state_history))
        self.current_state = self.state_history[-steps]
        self.state_history = self.state_history[:-steps]
        
        logger.info(f"Rolled back {steps} state(s)")
        return self.current_state
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.current_state = PokerTableState()
        self.state_history.clear()
        logger.info("StateMachine reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get state machine statistics."""
        return {
            "transition_count": self.transition_count,
            "rollback_count": self.rollback_count,
            "anomaly_count": self.anomaly_count,
            "history_size": len(self.state_history),
            "current_street": self.current_state.street.value,
            "board_cards": len(self.current_state.board_cards),
            "players": len(self.current_state.players),
        }
